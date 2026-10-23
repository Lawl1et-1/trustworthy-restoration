"""Example scan for the page's "Enhance a scan" section, at native resolution.

  python -m experiments.enhance_example [--dicom path.dcm]

One axial T2 slice of IXI002-Guys (an IXI test subject: a healthy volunteer from a hospital the networks never
saw), picked by the prepare_data.py rule (middle of the brain-level slices), scaled by the volume's 99.8th
percentile inside the head mask, saved as an 8-bit PNG to docs/demo/enhance_example.png. IXI: CC BY-SA 3.0.
--dicom also writes the same slice as an uncompressed 16-bit DICOM file (for testing the page's reader).
"""
import argparse
import pathlib
import struct
import tarfile

import nibabel as nib
import numpy as np
from PIL import Image

from experiments.prepare_data import _nifti, _pick_ixi_slices, ixi_head_mask

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUBJECT = "IXI002-Guys-0828-T2.nii.gz"


def ixi_slice(tar_path):
    tf = tarfile.open(tar_path, "r:")
    m = next(x for x in tf.getmembers() if x.name.endswith(SUBJECT))
    vol = np.asarray(nib.as_closest_canonical(_nifti(tf, m)).dataobj, dtype=np.float32)
    fg = ixi_head_mask(vol)
    zs = _pick_ixi_slices(fg.sum((0, 1)))
    z = zs[len(zs) // 2]
    return np.rot90(vol[:, :, z]), float(np.percentile(vol[fg], 99.8)), z


def write_dicom(path, img16):
    """Minimal explicit-VR little-endian DICOM (Secondary Capture) with a 16-bit greyscale image."""
    def el(g, e, vr, val):
        long = vr in ("OB", "OW", "UN", "SQ", "UT")
        head = struct.pack("<HH", g, e) + vr.encode()
        return head + (b"\0\0" + struct.pack("<I", len(val)) if long else struct.pack("<H", len(val))) + val
    pad = lambda b: b + (b"\0" if len(b) % 2 else b"")
    ts = pad(b"1.2.840.10008.1.2.1")
    meta = el(0x0002, 0x0010, "UI", ts)
    body = el(0x0008, 0x0016, "UI", pad(b"1.2.840.10008.5.1.4.1.1.7")) + \
        el(0x0028, 0x0002, "US", struct.pack("<H", 1)) + el(0x0028, 0x0004, "CS", b"MONOCHROME2 ") + \
        el(0x0028, 0x0010, "US", struct.pack("<H", img16.shape[0])) + el(0x0028, 0x0011, "US", struct.pack("<H", img16.shape[1])) + \
        el(0x0028, 0x0100, "US", struct.pack("<H", 16)) + el(0x0028, 0x0101, "US", struct.pack("<H", 16)) + \
        el(0x0028, 0x0102, "US", struct.pack("<H", 15)) + el(0x0028, 0x0103, "US", struct.pack("<H", 0)) + \
        el(0x7FE0, 0x0010, "OW", img16.astype("<u2").tobytes())
    glen = el(0x0002, 0x0000, "UL", struct.pack("<I", len(meta)))
    pathlib.Path(path).write_bytes(b"\0" * 128 + b"DICM" + glen + meta + body)


def main(a):
    sl, scale, z = ixi_slice(ROOT / "data" / "raw" / "ixi" / "IXI-T2.tar")
    u8 = (np.clip(sl / scale, 0, 1) * 255).round().astype(np.uint8)
    out = ROOT / "docs" / "demo" / "enhance_example.png"
    Image.fromarray(u8).save(out, optimize=True)
    print("wrote", out.relative_to(ROOT), u8.shape, "slice", z)
    if a.dicom:
        write_dicom(a.dicom, np.clip(sl, 0, 65535))
        print("wrote", a.dicom)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom")
    main(ap.parse_args())
