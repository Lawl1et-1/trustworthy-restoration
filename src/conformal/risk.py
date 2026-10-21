"""Distribution-free calibration of a per-pixel interval scale lambda.

Per-image risk L_i(lambda) = fraction of evaluation-region pixels outside the interval; it is
non-increasing in lambda. Two procedures on a lambda grid:
  * RCPS (Bates et al. 2021) with the Hoeffding-Bentkus bound: P(E[L(lambda_hat)] <= alpha) >= 1 - delta.
  * Conformal risk control (Angelopoulos et al. 2024): E[L(lambda_hat)] <= alpha.
"""
import numpy as np
from scipy.stats import binom, norm

LAMBDA_GRID = np.concatenate([[0.0], np.logspace(-2, 2, 400)])


def _h1(a, b):
    a = np.clip(a, 1e-12, 1 - 1e-12)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def hb_pvalue(rhat, n, alpha):
    """p-value for H0: E[L] > alpha, given empirical mean risk rhat over n i.i.d. images."""
    hoeff = np.exp(-n * _h1(np.minimum(rhat, alpha), alpha))
    bentkus = np.e * binom.cdf(np.ceil(n * rhat), n, alpha)
    return np.minimum(np.minimum(hoeff, bentkus), 1.0)


def rcps_lambda(risk, alpha, delta, grid=LAMBDA_GRID):
    """risk: (n_images, len(grid)). Scans from the largest lambda down; returns the index of lambda_hat."""
    n = risk.shape[0]
    rhat = risk.mean(0)
    idx = len(grid) - 1
    for j in range(len(grid) - 1, -1, -1):
        if hb_pvalue(rhat[j], n, alpha) >= delta:
            break
        idx = j
    return idx


def crc_lambda(risk, alpha, B=1.0, grid=LAMBDA_GRID):
    n = risk.shape[0]
    ok = (n * risk.mean(0) + B) / (n + 1) <= alpha
    return int(np.argmax(ok)) if ok.any() else len(grid) - 1


def half_widths(lam, lo_spread, hi_spread):
    return lam * lo_spread, lam * hi_spread


def risk_curves(x, center, lo_spread, hi_spread, region, grid=LAMBDA_GRID, fixed=None):
    """Per-image miss fraction and mean interval width over the grid.

    Interval: [center - (fixed + lam*lo), center + (fixed + lam*hi)] where `fixed` is an optional
    lambda-independent width (the analytic range-noise term of the null-space variant).
    All arrays (B, H, W) numpy; region bool. Returns risk (B, L), width (B, L).
    """
    B = x.shape[0]
    err = x - center
    fixed = np.zeros_like(x) if fixed is None else fixed
    risk = np.empty((B, len(grid)), np.float32)
    width = np.empty((B, len(grid)), np.float32)
    for b in range(B):
        r = region[b]
        e, lo, hi, f = err[b][r], lo_spread[b][r], hi_spread[b][r], fixed[b][r]
        # pixel is covered iff -(f + lam*lo) <= e <= f + lam*hi  <=>  lam >= needed
        need = np.where(e >= 0, (e - f) / np.maximum(hi, 1e-12), (-e - f) / np.maximum(lo, 1e-12))
        need = np.sort(np.maximum(need, 0))
        risk[b] = 1 - np.searchsorted(need, grid, side="right") / max(len(need), 1)
        width[b] = 2 * f.mean() + grid * (lo.mean() + hi.mean())
    return risk, width


def gaussian_z(alpha):
    return float(norm.ppf(1 - alpha / 2))
