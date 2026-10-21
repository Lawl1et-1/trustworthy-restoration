"""Sampling and missing-region masks."""
import numpy as np


def cartesian_mask(n, R, kind, rng, center_frac=0.08):
    """Phase-encode line mask over rows in *unshifted* FFT index order, conjugate-symmetric.

    Lines are chosen as pairs {k, -k}, so the realised acceleration n / n_lines can differ from R by a
    line or two; the realised value is what the experiments report.
    """
    assert kind in ("equispaced", "random")
    target = int(round(n / R))
    c = max(1, int(round(center_frac * n / 2)))
    chosen = {0} | {k for k in range(1, c + 1)} | {(-k) % n for k in range(1, c + 1)}
    pairs = list(range(c + 1, n // 2))
    if kind == "equispaced":
        off = int(rng.integers(R))
        order = [k for k in pairs if (k + off) % R == 0] + [k for k in pairs if (k + off) % R != 0]
    else:
        order = list(rng.permutation(pairs))
    for k in order:
        if len(chosen) + 2 > target:
            break
        chosen |= {int(k), (-int(k)) % n}
    rows = np.zeros(n, dtype=bool)
    rows[list(chosen)] = True
    return np.repeat(rows[:, None], n, axis=1)


def rect_mask(n, rng, area=(0.08, 0.2)):
    m = np.ones((n, n), dtype=bool)
    target = rng.uniform(*area) * n * n
    while (~m).sum() < target:
        h, w = rng.integers(n // 10, n // 4, size=2)
        y, x = rng.integers(0, n - h), rng.integers(0, n - w)
        m[y:y + h, x:x + w] = False
    return m


def irregular_mask(n, rng, area=(0.08, 0.2)):
    """Random brush strokes (random walks with a disc brush)."""
    m = np.ones((n, n), dtype=bool)
    yy, xx = np.mgrid[:n, :n]
    target = rng.uniform(*area) * n * n
    while (~m).sum() < target:
        y, x = rng.uniform(0, n, size=2)
        r = rng.uniform(n / 64, n / 24)
        for _ in range(int(rng.integers(5, 15))):
            a = rng.uniform(0, 2 * np.pi)
            y, x = np.clip(y + 3 * r * np.sin(a), 0, n - 1), np.clip(x + 3 * r * np.cos(a), 0, n - 1)
            m &= np.hypot(yy - y, xx - x) > r
    return m


def line_mask(n, rng, frac=(0.1, 0.25)):
    """Motion-like corruption: contiguous groups of image rows are discarded."""
    m = np.ones((n, n), dtype=bool)
    target = rng.uniform(*frac) * n
    while (~m[:, 0]).sum() < target:
        y = rng.integers(0, n)
        m[y:y + int(rng.integers(1, 4)), :] = False
    return m


def inpaint_mask(n, kind, rng):
    if kind == "mixed":
        kind = ("rect", "irregular", "lines")[int(rng.integers(3))]
    return {"rect": rect_mask, "irregular": irregular_mask, "lines": line_mask}[kind](n, rng)
