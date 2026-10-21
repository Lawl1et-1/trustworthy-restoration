"""Patient-level bootstrap. Slices of one patient are correlated, so patients are resampled, not slices."""
import numpy as np


def _patient_means(values, pids):
    pids = np.asarray(pids).astype(str)
    uniq, inv = np.unique(pids, return_inverse=True)
    v = np.asarray(values, float)
    ok = ~np.isnan(v)
    s = np.bincount(inv[ok], v[ok], minlength=len(uniq))
    c = np.bincount(inv[ok], minlength=len(uniq))
    with np.errstate(invalid="ignore"):
        return s / c, c > 0


def bootstrap_ci(values, pids, n_boot=2000, level=0.95, seed=0):
    """Mean of per-patient means, with a percentile bootstrap CI over patients."""
    m, ok = _patient_means(values, pids)
    m = m[ok]
    if len(m) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = m[rng.integers(0, len(m), (n_boot, len(m)))].mean(1)
    a = (1 - level) / 2
    return float(m.mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def paired_bootstrap(values_a, values_b, pids, n_boot=2000, level=0.95, seed=0):
    """Mean per-patient difference a - b on the same test patients, CI, and a two-sided bootstrap p."""
    d, ok = _patient_means(np.asarray(values_a, float) - np.asarray(values_b, float), pids)
    d = d[ok]
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    a = (1 - level) / 2
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a)), float(min(p, 1.0))
