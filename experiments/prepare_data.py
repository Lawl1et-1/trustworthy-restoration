"""Converts MSD Task01 (T2w channel) and IXI-T2 into 128x128 axial slice datasets.

Reads straight from the downloaded .tar archives (no extraction). Each volume is scaled by the
99.8th percentile of its own foreground: a per-volume rescaling, nothing is fitted across patients.

  python -m experiments.prepare_data --msd data/raw/msd/Task01_BrainTumour.tar --ixi data/raw/ixi/IXI-T2.tar
"""
import argparse
import gzip
import io
import json
import pathlib
import tarfile
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.transform import resize

SIZE = 128
SLICES_PER_VOL = 8
MSD_T2 = 3  # dataset.json modality order: FLAIR, T1w, t1gd, T2w


def _nifti(tf, member):
    raw = tf.extractfile(member).read()
    return nib.Nifti1Image.from_bytes(gzip.decompress(raw))


def _square(a, fill=0):
    """Centre-pad (or crop) the in-plane axes to a square, then resize to SIZE."""
    h, w = a.shape
    s = max(h, w)
    out = np.full((s, s), fill, a.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    return out


def _pick_slices(fg_area):
    """SLICES_PER_VOL evenly spaced axial slices among those with >= 50 % of the max brain area."""
    ok = np.nonzero(fg_area >= 0.5 * fg_area.max())[0]
    return ok[np.linspace(0, len(ok) - 1, SLICES_PER_VOL).round().astype(int)]


_TAR = {}


def _open_tar(path):
    """One open archive and member index per process (workers each open their own)."""
    if path not in _TAR:
        tf = tarfile.open(path, "r:")
        _TAR[path] = (tf, {m.name: m for m in tf.getmembers()})
    return _TAR[path]


def _msd_volume(args):
    tar_path, name = args
    tf, members = _open_tar(tar_path)
    vol = np.asarray(_nifti(tf, members[name]).dataobj[..., MSD_T2], dtype=np.float32)  # (240,240,155)
    lab = np.asarray(_nifti(tf, members[name.replace("imagesTr", "labelsTr")]).dataobj) > 0
    fg = vol > 0
    scale = np.percentile(vol[fg], 99.8)
    out = []
    for z in _pick_slices(fg.sum((0, 1))):
        out.append((resize(_square(np.rot90(vol[:, :, z])), (SIZE, SIZE), anti_aliasing=True, preserve_range=True) / scale,
                    resize(_square(np.rot90(lab[:, :, z]).astype(np.float32)), (SIZE, SIZE), order=0) > 0.5,
                    resize(_square(np.rot90(fg[:, :, z]).astype(np.float32)), (SIZE, SIZE), order=0) > 0.5,
                    pathlib.Path(name).name.split(".")[0]))
    return out


def prepare_msd(tar_path, out, workers=6):
    tf, members = _open_tar(tar_path)
    # the archive also holds macOS "._" resource-fork files, e.g. "._dataset.json" before "dataset.json"
    meta = json.loads(tf.extractfile([m for m in members.values() if pathlib.PurePosixPath(m.name).name == "dataset.json"][0]).read())
    assert meta["modality"]["3"].upper().startswith("T2"), meta["modality"]
    names = sorted(n for n in members if "/imagesTr/" in n and n.endswith(".nii.gz") and "/._" not in n)
    imgs, labs, fgs, pids = [], [], [], []
    # volumes are independent; pool.map keeps the sorted order, so the output is identical to a serial run
    with ProcessPoolExecutor(workers) as pool:
        for i, slices in enumerate(pool.map(_msd_volume, [(tar_path, n) for n in names], chunksize=4)):
            for im, lb, fg, pid in slices:
                imgs.append(im); labs.append(lb); fgs.append(fg); pids.append(pid)
            if i % 50 == 0:
                print(f"msd {i}/{len(names)}", flush=True)
    np.savez_compressed(out, images=np.clip(np.stack(imgs), 0, 1).astype(np.float32), labels=np.stack(labs),
                        fg=np.stack(fgs), pid=np.array(pids), site=np.array(["msd"] * len(pids)))
    print("wrote", out, len(pids), "slices from", len(names), "volumes")


IXI_BRAIN_FRAC = 0.45  # skip the lowest 45 % of the head's axial extent (jaw, neck, skull base)


def ixi_head_mask(vol):
    """Head mask, slice by slice: threshold at half Otsu, close, fill holes, keep the largest component.

    The first version (Otsu, 3D opening and fill_holes on the whole volume) traced internal tissue edges
    instead of the head outline: dark T2 tissue fell below Otsu, and 3D hole filling fails because the
    interior connects to the outside through the open top and bottom of the slab (LOG.md, M2).
    """
    thr = 0.5 * threshold_otsu(vol)
    out = np.zeros(vol.shape, bool)
    for z in range(vol.shape[2]):
        m = ndimage.binary_fill_holes(ndimage.binary_closing(vol[:, :, z] > thr, iterations=3))
        lab, n = ndimage.label(m)
        if n:
            out[:, :, z] = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return out


def _pick_ixi_slices(fg_area):
    """Brain-level slices, comparable to MSD: drop the lowest IXI_BRAIN_FRAC of the head's axial extent,
    then use the same >= 50 % max-area rule as MSD."""
    zs = np.nonzero(fg_area > 0)[0]
    lo = zs[0] + int(IXI_BRAIN_FRAC * (zs[-1] - zs[0]))
    area = np.where(np.arange(len(fg_area)) >= lo, fg_area, 0)
    return _pick_slices(area)


def prepare_ixi(tar_path, out, per_site):
    """IXI is healthy subjects and not skull-stripped; the evaluation region is the head mask."""
    tf = tarfile.open(tar_path, "r:")
    members = sorted((m for m in tf.getmembers() if m.name.endswith(".nii.gz")), key=lambda m: m.name)
    counts, imgs, labs, fgs, pids, sites = {}, [], [], [], [], []
    for m in members:
        site = m.name.split("-")[1]  # IXI002-Guys-0828-T2.nii.gz
        if counts.get(site, 0) >= per_site:
            continue
        img = nib.as_closest_canonical(_nifti(tf, m))  # RAS: axis 2 is axial
        vol = np.asarray(img.dataobj, dtype=np.float32)
        fg = ixi_head_mask(vol)
        scale = np.percentile(vol[fg], 99.8)
        for z in _pick_ixi_slices(fg.sum((0, 1))):
            imgs.append(resize(_square(np.rot90(vol[:, :, z])), (SIZE, SIZE), anti_aliasing=True, preserve_range=True) / scale)
            fgs.append(resize(_square(np.rot90(fg[:, :, z]).astype(np.float32)), (SIZE, SIZE), order=0) > 0.5)
            labs.append(np.zeros((SIZE, SIZE), bool))
            pids.append(pathlib.Path(m.name).name.split("-T2")[0]); sites.append(site)
        counts[site] = counts.get(site, 0) + 1
    np.savez_compressed(out, images=np.clip(np.stack(imgs), 0, 1).astype(np.float32), labels=np.stack(labs),
                        fg=np.stack(fgs), pid=np.array(pids), site=np.array(sites))
    print("wrote", out, counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--msd")
    ap.add_argument("--ixi")
    ap.add_argument("--ixi-per-site", type=int, default=40)
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.msd:
        prepare_msd(a.msd, f"{a.out}/msd_t2_{SIZE}.npz")
    if a.ixi:
        prepare_ixi(a.ixi, f"{a.out}/ixi_t2_{SIZE}.npz", a.ixi_per_site)
