"""Classical restorers. Regularisation weights are tuned on the validation split (experiments/run_task.py)."""
import numpy as np
import pywt
import torch
from skimage.restoration import denoise_tv_chambolle, inpaint_biharmonic


def _rician_mean_table(n=2000, amax=60.0):
    """E[M] / sigma as a function of nu = A / sigma for a Rician magnitude M (monotone in nu)."""
    from scipy.special import ive
    nu = np.linspace(0, amax, n)
    x = -nu ** 2 / 2
    # Laguerre L_{1/2}(x) = e^{x/2}[(1-x) I0(-x/2) - x I1(-x/2)], written with scaled Bessels for stability
    lag = (1 - x) * ive(0, -x / 2) - x * ive(1, -x / 2)
    return nu, np.sqrt(np.pi / 2) * lag


_NU, _MEAN = _rician_mean_table()


def rician_unbias(m, sigma):
    """First-moment bias correction: the A with E[M | A, sigma] = m (0 where m < sigma*sqrt(pi/2)).

    Applied to the output of every classical denoiser: a smoother estimates E[M], which is biased
    upwards by Rician noise (the zero background goes to ~1.25 sigma). Learned restorers absorb the
    bias during training, so they do not need it.
    """
    r = np.asarray(m) / sigma
    return np.where(r > _MEAN[-1], np.sqrt(np.maximum(r ** 2 - 1, 0)), np.interp(r, _MEAN, _NU)) * sigma


def _soft(c, t):
    return np.sign(c) * np.maximum(np.abs(c) - t, 0)


def _wavelet_prox(x, t, wavelet="db4", level=3):
    coeffs = pywt.wavedec2(x, wavelet, level=level, mode="periodization")
    coeffs = [coeffs[0]] + [tuple(_soft(c, t) for c in d) for d in coeffs[1:]]
    return pywt.waverec2(coeffs, wavelet, mode="periodization")[: x.shape[0], : x.shape[1]]


def _spun_wavelet_prox(rng, t):
    """Cycle spinning: soft-threshold after a random circular shift, then undo it. The decimated
    wavelet-L1 minimiser itself has blocky artefacts and lost PSNR on R=4 phantoms as the solver
    converged (LOG.md, M2); shift averaging over iterations removes most of that."""
    def prox(z):
        s = rng.integers(0, 8, 2)
        return np.roll(_wavelet_prox(np.roll(z, s, (0, 1)), t), -s, (0, 1))
    return prox


def _fista(inp, op, y, prox, iters=100):
    """FISTA on 0.5||Ax - y||^2 + R(x); step 1 because ||A|| = 1 for both operators.

    30 plain ISTA steps (the first version) were far from converged: TV at R=4 gained ~1 dB over
    zero-filling instead of ~10 dB, which would have flattered every learned method.
    """
    x = inp.clone()
    z, t = x.clone(), 1.0
    aty = op.AT(y)
    for _ in range(iters):
        g = (z - op.P_R(z) + aty).numpy()
        xn = torch.from_numpy(np.stack([[prox(a[0])] for a in g]).astype(np.float32))
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z, x, t = xn + ((t - 1) / tn) * (xn - x), xn, tn
    return x


def restore(name, d, w):
    """d: degradation dict from degrade(); w: regularisation weight. Returns (B,1,H,W) tensor."""
    out = _restore(name, d, w)
    if d["op"] is None and name != "noisy":  # denoising of Rician magnitude data
        out = torch.from_numpy(rician_unbias(out.numpy(), d["sigma"]).astype(np.float32))
    return out


def _restore(name, d, w):
    inp, op, y = d["inp"], d["op"], d["y"]
    if name in ("zero_filled", "noisy", "masked"):
        return inp.clone()
    if name == "tv":
        if op is None:
            return torch.from_numpy(np.stack([[denoise_tv_chambolle(a[0].numpy(), weight=w)] for a in inp]).astype(np.float32))
        return _fista(inp, op, y, lambda z: denoise_tv_chambolle(z, weight=w))
    if name == "wavelet":
        if op is None:  # denoising: A = I, so the minimiser is a single proximal step
            return torch.from_numpy(np.stack([[_wavelet_prox(a[0].numpy(), w)] for a in inp]).astype(np.float32))
        return _fista(inp, op, y, _spun_wavelet_prox(np.random.default_rng(0), w))
    if name == "bm3d":
        import bm3d
        return torch.from_numpy(np.stack([[bm3d.bm3d(a[0].numpy(), sigma_psd=w)] for a in inp]).astype(np.float32))
    if name == "biharmonic":
        out = [inpaint_biharmonic(a[0].numpy(), ~m[0].numpy().astype(bool)) for a, m in zip(inp, op.mask)]
        return torch.from_numpy(np.stack(out)[:, None].astype(np.float32))
    raise ValueError(name)


CLASSICAL = {"mri": ["zero_filled", "tv", "wavelet"], "denoise": ["noisy", "tv", "wavelet", "bm3d"],
             "inpaint": ["masked", "biharmonic", "tv"]}
# Wide enough that the tuned value is interior for every severity; run_task.py logs a warning when the
# chosen weight sits on the edge of its grid.
WEIGHT_GRID = {"tv": [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32],
               "wavelet": [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32],
               "bm3d": [0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3],
               "zero_filled": [0], "noisy": [0], "masked": [0], "biharmonic": [0]}
