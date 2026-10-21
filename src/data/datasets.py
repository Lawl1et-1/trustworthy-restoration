"""Slice datasets with patient ids, as used by every experiment.

  phantom  synthetic Shepp-Logan-style brains with known lesions (each image = one patient)
  msd      MSD Task01 BrainTumour, T2w channel (from experiments/prepare_data.py)
  ixi      IXI T2, three sites; out-of-distribution test set only, never split or trained on

Intensities are already in [0, 1]: phantoms by construction, MRI by a per-volume rescaling in
prepare_data.py. Nothing is fitted across patients, so there is no normalisation to leak.
"""
import pathlib
from dataclasses import dataclass, field

import numpy as np

from src.phantoms.phantom import phantom_dataset

from .splits import get_or_make_split, split_indices

ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass
class SliceData:
    images: np.ndarray            # (N, H, W) float32 in [0, 1]
    labels: np.ndarray            # (N, H, W) bool lesion / tumour mask
    region: np.ndarray            # (N, H, W) bool evaluation region (brain / foreground)
    pid: np.ndarray               # (N,) patient id per slice
    site: np.ndarray              # (N,) acquisition site
    lesions: list = field(default=None)  # phantom only: per-image list of lesion dicts

    def __len__(self):
        return len(self.images)

    def take(self, idx):
        idx = np.asarray(idx)
        les = None if self.lesions is None else [self.lesions[i] for i in idx]
        return SliceData(self.images[idx], self.labels[idx], self.region[idx], self.pid[idx], self.site[idx], les)

    def head(self, n):
        """First slices of the first patients in sorted-id order: a deterministic subset that keeps
        whole patients together (used for smoke runs and for tuning on a slice of val)."""
        if n is None or n >= len(self):
            return self
        order = np.lexsort((np.arange(len(self)), self.pid))
        return self.take(np.sort(order[:n]))

    def center_crop(self, size):
        H = self.images.shape[-1]
        if size is None or size >= H:
            return self
        a = (H - size) // 2
        sl = (slice(None), slice(a, a + size), slice(a, a + size))
        return SliceData(self.images[sl], self.labels[sl], self.region[sl], self.pid, self.site, self.lesions)


def _phantoms(cfg):
    imgs, masks, brains, les = phantom_dataset(cfg["n_images"], cfg["size"], seed=cfg.get("seed", 1234),
                                               lesion_rate=cfg.get("lesion_rate", 0.7))
    pid = np.array([f"ph{i:05d}" for i in range(len(imgs))])
    return SliceData(imgs, masks, brains, pid, np.array(["phantom"] * len(imgs)), les)


def _npz(path):
    path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: see DATA.md (download, then experiments/prepare_data.py)")
    z = np.load(path)
    return SliceData(z["images"], z["labels"], z["fg"], z["pid"].astype(str), z["site"].astype(str))


def load_dataset(cfg, split_dir="results/splits"):
    """cfg: the `data` block of a YAML config. Returns {split: SliceData}.

    IXI is returned as {"test": ...} only. For split datasets the saved split file is reused when it
    exists, so a smoke run and a full run on the same data use the same patients.
    """
    name = cfg["name"]
    if name == "phantom":
        data = _phantoms(cfg)
    elif name in ("msd", "ixi"):
        data = _npz(cfg["path"])
    else:
        raise ValueError(name)
    if name == "ixi":
        sites = cfg.get("sites")
        if sites:
            data = data.take(np.nonzero(np.isin(data.site, sites))[0])
        out = {"test": data}
    else:
        tag = cfg.get("split_name", f"{name}_n{cfg['n_images']}" if name == "phantom" else name)
        split = get_or_make_split(ROOT / split_dir / f"{tag}.json", data.pid, cfg.get("fracs"),
                                  seed=cfg.get("split_seed", 0), meta={"dataset": name})
        out = {s: data.take(i) for s, i in split_indices(data.pid, split).items()}
    limits = cfg.get("max_slices", {})
    return {s: d.head(limits.get(s)).center_crop(cfg.get("crop")) for s, d in out.items()}
