import numpy as np
import torch

from src.degradations.degrade import degrade, task_generator
from src.eval.metrics import psnr
from src.eval.stats import bootstrap_ci, paired_bootstrap
from src.models.classical import restore
from src.phantoms.phantom import phantom_dataset

X = torch.from_numpy(phantom_dataset(4, 64, seed=0)[0])[:, None]


def _psnr(a, b):
    return psnr(a[:, 0].numpy(), b[:, 0].clamp(0, 1).numpy()).mean()


def test_denoisers_beat_noisy_input():
    d = degrade(X, {"type": "denoise", "sigma": 0.1}, task_generator("t", "test", 0))
    base = _psnr(X, d["inp"])
    for name, w in (("tv", 0.04), ("wavelet", 0.08), ("bm3d", 0.075)):  # grid optima at sigma = 0.1
        assert _psnr(X, restore(name, d, w)) > base + 3, name


def test_rician_unbias_inverts_the_mean():
    from src.degradations.degrade import rician
    from src.models.classical import rician_unbias
    g = torch.Generator().manual_seed(0)
    for A in (0.05, 0.1, 0.3, 0.8):
        m = rician(torch.full((200_000,), A), 0.1, g).mean().item()
        assert abs(float(rician_unbias(m, 0.1)) - A) < 0.003, A


def test_linear_restorers_beat_pseudo_inverse_and_fit_data():
    for spec, names in (({"type": "mri", "R": 4, "sigma": 0.02}, (("tv", 0.01), ("wavelet", 0.01))),
                        ({"type": "inpaint", "sigma": 0.01}, (("tv", 0.01), ("biharmonic", 0)))):
        d = degrade(X, spec, task_generator("t", "test", 1))
        base = _psnr(X, d["inp"])
        for name, w in names:
            xh = restore(name, d, w)
            assert _psnr(X, xh) > base, (spec["type"], name)


def test_inpaint_tv_keeps_measured_pixels():
    d = degrade(X, {"type": "inpaint", "sigma": 0.0}, task_generator("t", "test", 2))
    xh = restore("tv", d, 0.01)
    m = d["op"].mask.bool()
    # FISTA ends with a prox step, so measured pixels are only approximately kept
    assert (xh[m] - X[m]).abs().mean() < 0.02


def test_task_generator_is_paired_and_split_specific():
    a = degrade(X, {"type": "denoise", "sigma": 0.1}, task_generator("t", "test", 0))["inp"]
    b = degrade(X, {"type": "denoise", "sigma": 0.1}, task_generator("t", "test", 0))["inp"]
    c = degrade(X, {"type": "denoise", "sigma": 0.1}, task_generator("t", "val", 0))["inp"]
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_bootstrap_groups_by_patient():
    # 2 patients: one with 9 slices at 0, one with 1 slice at 10 -> patient mean 5, not slice mean 1
    v = np.r_[np.zeros(9), 10.0]
    pid = np.array(["a"] * 9 + ["b"])
    mu, lo, hi = bootstrap_ci(v, pid)
    assert mu == 5.0 and lo <= mu <= hi
    d, lo, hi, p = paired_bootstrap(np.ones(20) + np.arange(20) * 1e-3, np.zeros(20), np.arange(20))
    assert lo > 0 and p < 0.01
