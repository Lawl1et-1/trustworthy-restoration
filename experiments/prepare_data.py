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


def prepare_msd(tar_path, out):
    tf = tarfile.open(tar_path, "r:")
    members = {m.name: m for m in tf.getmembers() if m.name.endswith(".nii.gz") and "/._" not in m.name}
    meta = json.loads(tf.extractfile([m for m in tf.getmembers() if m.name.endswith("dataset.json")][0]).read())
    assert meta["modality"]["3"].upper().startswith("T2"), meta["modality"]
    imgs, labs, fgs, pids = [], [], [], []
    names = sorted(n for n in members if "/imagesTr/" in n)
    for i, name in enumerate(names):
        vol = np.asarray(_nifti(tf, members[name]).dataobj[..., MSD_T2], dtype=np.float32)  # (240,240,155)
        lab = np.asarray(_nifti(tf, members[name.replace("imagesTr", "labelsTr")]).dataobj) > 0
        fg = vol > 0
        scale = np.percentile(vol[fg], 99.8)
        for z in _pick_slices(fg.sum((0, 1))):
            im = np.rot90(vol[:, :, z])
            imgs.append(resize(_square(im), (SIZE, SIZE), anti_aliasing=True, preserve_range=True) / scale)
            labs.append(resize(_square(np.rot90(lab[:, :, z]).astype(np.float32)), (SIZE, SIZE), order=0) > 0.5)
            fgs.append(resize(_square(np.rot90(fg[:, :, z]).astype(np.float32)), (SIZE, SIZE), order=0) > 0.5)
            pids.append(pathlib.Path(name).name.split(".")[0])
        if i % 50 == 0:
            print(f"msd {i}/{len(names)}", flush=True)
    np.savez_compressed(out, images=np.clip(np.stack(imgs), 0, 1).astype(np.float32), labels=np.stack(labs),
                        fg=np.stack(fgs), pid=np.array(pids), site=np.array(["msd"] * len(pids)))
    print("wrote", out, len(pids), "slices from", len(names), "volumes")


def prepare_ixi(tar_path, out, per_site):
    """IXI is not skull-stripped; the evaluation region is an Otsu foreground with holes filled."""
    tf = tarfile.open(tar_path, "r:")
    members = sorted((m for m in tf.getmembers() if m.name.endswith(".nii.gz")), key=lambda m: m.name)
    counts, imgs, labs, fgs, pids, sites = {}, [], [], [], [], []
    for m in members:
        site = m.name.split("-")[1]  # IXI002-Guys-0828-T2.nii.gz
        if counts.get(site, 0) >= per_site:
            continue
        img = nib.as_closest_canonical(_nifti(tf, m))  # RAS: axis 2 is axial
        vol = np.asarray(img.dataobj, dtype=np.float32)
        thr = threshold_otsu(vol)
        fg = ndimage.binary_fill_holes(ndimage.binary_opening(vol > thr, iterations=2))
        scale = np.percentile(vol[fg], 99.8)
        for z in _pick_slices(fg.sum((0, 1))):
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
