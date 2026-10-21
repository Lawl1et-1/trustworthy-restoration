"""Classical restorers. Regularisation weights are tuned on the validation split (experiments/run_task.py)."""
import numpy as np
import pywt
import torch
from skimage.restoration import denoise_tv_chambolle, inpaint_biharmonic


def _soft(c, t):
    return np.sign(c) * np.maximum(np.abs(c) - t, 0)


def _wavelet_prox(x, t, wavelet="db4", level=3):
    coeffs = pywt.wavedec2(x, wavelet, level=level, mode="periodization")
    coeffs = [coeffs[0]] + [tuple(_soft(c, t) for c in d) for d in coeffs[1:]]
    return pywt.waverec2(coeffs, wavelet, mode="periodization")[: x.shape[0], : x.shape[1]]


def _ista(inp, op, y, prox, iters=30):
    """Proximal gradient on 0.5||Ax - y||^2 + R(x); step 1 because ||A|| = 1 for both operators."""
    x = inp.clone()
    aty = op.AT(y)
    for _ in range(iters):
        z = (x - op.P_R(x) + aty).numpy()
        x = torch.from_numpy(np.stack([[prox(zz[0])] for zz in z]).astype(np.float32))
    return x


def restore(name, d, w):
    """d: degradation dict from degrade(); w: regularisation weight. Returns (B,1,H,W) tensor."""
    inp, op, y = d["inp"], d["op"], d["y"]
    if name in ("zero_filled", "noisy", "masked"):
        return inp.clone()
    if name == "tv":
        if op is None:
            return torch.from_numpy(np.stack([[denoise_tv_chambolle(a[0].numpy(), weight=w)] for a in inp]).astype(np.float32))
        return _ista(inp, op, y, lambda z: denoise_tv_chambolle(z, weight=w))
    if name == "wavelet":
        return _ista(inp, op, y, lambda z: _wavelet_prox(z, w))
    if name == "bm3d":
        import bm3d
        return torch.from_numpy(np.stack([[bm3d.bm3d(a[0].numpy(), sigma_psd=w)] for a in inp]).astype(np.float32))
    if name == "biharmonic":
        out = [inpaint_biharmonic(a[0].numpy(), ~m[0].numpy().astype(bool)) for a, m in zip(inp, op.mask)]
        return torch.from_numpy(np.stack(out)[:, None].astype(np.float32))
    raise ValueError(name)


CLASSICAL = {"mri": ["zero_filled", "tv", "wavelet"], "denoise": ["noisy", "tv", "bm3d"],
             "inpaint": ["masked", "biharmonic", "tv"]}
WEIGHT_GRID = {"tv": [0.005, 0.01, 0.02, 0.04, 0.08], "wavelet": [0.005, 0.01, 0.02, 0.04],
               "bm3d": [0.03, 0.06, 0.1, 0.15], "zero_filled": [0], "noisy": [0], "masked": [0], "biharmonic": [0]}
