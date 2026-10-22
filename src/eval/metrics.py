"""HalluBench-Med metrics. Arrays are numpy (B, H, W) unless stated; intensities in [0, 1]."""
import numpy as np
import torch
from scipy import ndimage
from scipy.stats import norm
from skimage.feature import peak_local_max
from skimage.metrics import structural_similarity
from sklearn.metrics import roc_auc_score

# ---------------- fidelity ----------------

def psnr(x, xh):
    mse = ((x - xh) ** 2).mean((1, 2))
    return 10 * np.log10(1.0 / np.maximum(mse, 1e-12))


def ssim(x, xh):
    return np.array([structural_similarity(a, b, data_range=1.0) for a, b in zip(x, xh)])


def nrmse(x, xh):
    return np.sqrt(((x - xh) ** 2).sum((1, 2)) / np.maximum((x ** 2).sum((1, 2)), 1e-12))


# ---------------- perception ----------------

_LPIPS = None


def _lpips():
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).eval()
    return _LPIPS


def _rgb(a):
    t = torch.from_numpy(np.asarray(a, np.float32))[:, None]
    return (t.clamp(0, 1) * 2 - 1).repeat(1, 3, 1, 1)


@torch.no_grad()
def lpips_dist(x, xh, bs=64):
    net = _lpips()
    return np.concatenate([net(_rgb(x[i:i + bs]), _rgb(xh[i:i + bs])).flatten().numpy() for i in range(0, len(x), bs)])


@torch.no_grad()
def alex_features(x, bs=64):
    """256-d global-average-pooled conv5 features of the LPIPS AlexNet trunk (for KID)."""
    net = _lpips().net
    out = []
    for i in range(0, len(x), bs):
        h = net(_rgb(x[i:i + bs]))[-1]
        out.append(h.mean((2, 3)).numpy())
    return np.concatenate(out)


def kid(fa, fb):
    """Unbiased squared MMD with the cubic polynomial kernel (Binkowski et al. 2018)."""
    d = fa.shape[1]
    k = lambda a, b: (a @ b.T / d + 1) ** 3
    m, n = len(fa), len(fb)
    kaa, kbb, kab = k(fa, fa), k(fb, fb), k(fa, fb)
    return ((kaa.sum() - np.trace(kaa)) / (m * (m - 1)) + (kbb.sum() - np.trace(kbb)) / (n * (n - 1)) - 2 * kab.mean())


KID_MIN_N = 1000  # below this the score is reported but flagged as not meaningful


def grad_energy(x):
    gx, gy = ndimage.sobel(x, axis=1), ndimage.sobel(x, axis=2)
    return (gx ** 2 + gy ** 2).mean((1, 2))


def laplacian_var(x):
    return np.array([ndimage.laplace(a).var() for a in x])


# ---------------- hallucination ----------------

def spurious_blobs(x, xh, region, exclude, amp=0.08, sigma=1.0):
    """Counts lesion-like blobs in the residual xh - x: (fabricated bright, erased/dark) per image.

    The residual is smoothed with a Gaussian matched to small lesions; local maxima above `amp` inside
    the eroded `region` and outside the dilated `exclude` mask (true lesions) are counted.
    """
    fab, era = [], []
    for a, b, r, e in zip(x, xh, region, exclude):
        ok = ndimage.binary_erosion(r, iterations=3) & ~ndimage.binary_dilation(e, iterations=3)
        res = ndimage.gaussian_filter(b - a, sigma)
        counts = []
        for s in (res, -res):
            pk = peak_local_max(np.where(ok, s, 0), min_distance=3, threshold_abs=amp, exclude_border=False)
            counts.append(len(pk))
        fab.append(counts[0]); era.append(counts[1])
    return np.array(fab), np.array(era)


def lesion_recovery(xh, x_bg, lesions, n):
    """Fraction of the true contrast recovered for each phantom lesion (1 = perfect, < 0.5 = erased).

    x_bg is the lesion-free version of the same image, so recovery = mean over the lesion core of
    (restored - lesion-free clean) / mean over the core of the true lesion (contrast x soft profile).
    """
    from src.phantoms.phantom import lesion_profile
    out = []
    for les in lesions:
        prof = lesion_profile(n, les["cy"], les["cx"], les["r"])
        core = prof > 0.5
        out.append(float((xh - x_bg)[core].mean() / (les["contrast"] * prof[core].mean())))
    return out


def npw_statistic(img, template):
    """Non-prewhitening matched filter: t = <template, image>."""
    return (img * template).sum((-2, -1))


def detect_auc(t_present, t_absent):
    y = np.r_[np.ones(len(t_present)), np.zeros(len(t_absent))]
    return roc_auc_score(y, np.r_[t_present, t_absent])


def null_space_energy(op, x, xh):
    """||P_N (xh - x)||^2 / ||x||^2 per image (op is a LinearOp; tensors in, numpy out)."""
    d = op.P_N(torch.as_tensor(xh - x)[:, None].float())[:, 0].numpy()
    return (d ** 2).sum((1, 2)) / np.maximum((x ** 2).sum((1, 2)), 1e-12)


def dc_violation(op, y, x, xh):
    """||A xh - y||^2 / ||A x - y||^2: 1 means the restoration fits the data as well as the truth does."""
    r = (op.A(torch.as_tensor(xh)[:, None].float()) - y).abs().pow(2).flatten(1).sum(1)
    r0 = (op.A(torch.as_tensor(x)[:, None].float()) - y).abs().pow(2).flatten(1).sum(1)
    return (r / r0.clamp_min(1e-12)).numpy()


# ---------------- uncertainty quality ----------------

SPARS_FRACS = np.linspace(0, 0.95, 20)
CAL_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


def sparsification(err, unc, region):
    """Per-image normalised RMSE after removing the most-uncertain fraction (and the oracle curve)."""
    curves, oracles = [], []
    for e, u, r in zip(err, unc, region):
        e2, uu = e[r] ** 2, u[r]
        rm = []
        for order in (np.argsort(-uu), np.argsort(-e2)):
            s = e2[order]
            tot = np.sqrt(s.mean())
            rm.append([np.sqrt(s[int(f * len(s)):].mean()) / max(tot, 1e-12) for f in SPARS_FRACS])
        curves.append(rm[0]); oracles.append(rm[1])
    curves, oracles = np.array(curves), np.array(oracles)
    ause = np.trapezoid(curves - oracles, SPARS_FRACS, axis=1)
    return curves, oracles, ause


def error_auroc(err, unc, region, tau):
    out = []
    for e, u, r in zip(err, unc, region):
        lab = np.abs(e[r]) > tau
        out.append(roc_auc_score(lab, u[r]) if 0 < lab.sum() < lab.size else np.nan)
    return np.array(out)


def calibration_curve(err, sigma, region):
    """Observed fraction of |err| <= z_p * sigma for Gaussian central intervals of level p."""
    z = norm.ppf((1 + CAL_LEVELS) / 2)
    return np.array([[(np.abs(e[r]) <= zz * s[r]).mean() for zz in z] for e, s, r in zip(err, sigma, region)])


def dice(pred, gt):
    inter = (pred & gt).sum((1, 2))
    den = pred.sum((1, 2)) + gt.sum((1, 2))
    return np.where(den > 0, 2 * inter / np.maximum(den, 1), np.nan)
