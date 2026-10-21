"""Shepp-Logan-style phantoms with randomised anatomy and small lesions of known size and contrast.

The lesion list is the ground truth for the hallucination experiments: we know exactly where every
lesion is, how big it is and how bright it is, so "erased" and "fabricated" can be counted exactly.
"""
import numpy as np

# Modified Shepp-Logan (Toft 1996): amplitude, semi-axis a, semi-axis b, x0, y0, angle (deg)
_SL = [
    (1.00, .6900, .920, 0.00, 0.0000, 0),
    (-.80, .6624, .874, 0.00, -.0184, 0),
    (-.20, .1100, .310, 0.22, 0.0000, -18),
    (-.20, .1600, .410, -.22, 0.0000, 18),
    (0.10, .2100, .250, 0.00, 0.3500, 0),
    (0.10, .0460, .046, 0.00, 0.1000, 0),
    (0.10, .0460, .046, 0.00, -.1000, 0),
    (0.10, .0460, .023, -.08, -.6050, 0),
    (0.10, .0230, .023, 0.00, -.6060, 0),
    (0.10, .0230, .046, 0.06, -.6050, 0),
]


def _grid(n):
    c = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
    return np.meshgrid(c, -c)  # x to the right, y up


def _ellipse(X, Y, a, b, x0, y0, phi):
    t = np.deg2rad(phi)
    xr = (X - x0) * np.cos(t) + (Y - y0) * np.sin(t)
    yr = -(X - x0) * np.sin(t) + (Y - y0) * np.cos(t)
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1


def lesion_profile(n, cy, cx, radius, edge=0.6):
    """Smooth disc of unit height centred at pixel (cy, cx); also the observer's template."""
    yy, xx = np.mgrid[:n, :n]
    d = np.hypot(yy - cy, xx - cx)
    return 1.0 / (1.0 + np.exp((d - radius) / edge))


def random_phantom(n, rng, texture=0.03):
    """Returns (image in [0, 1], brain mask) with randomised ellipse geometry and smooth texture."""
    X, Y = _grid(n)
    img = np.zeros((n, n))
    brain = None
    for i, (A, a, b, x0, y0, phi) in enumerate(_SL):
        s = rng.uniform(0.9, 1.1) if i > 1 else rng.uniform(0.97, 1.03)
        e = _ellipse(X, Y, a * s, b * s, x0 + rng.uniform(-.02, .02), y0 + rng.uniform(-.02, .02),
                     phi + rng.uniform(-8, 8))
        img += A * (rng.uniform(0.8, 1.2) if i > 1 else 1.0) * e
        if i == 1:
            brain = e
    # low-frequency texture so the interior is not piecewise constant (otherwise TV wins trivially)
    tex = sum(rng.uniform(-1, 1) * np.sin(rng.uniform(1, 6) * np.pi * X + rng.uniform(0, 6.3))
              * np.cos(rng.uniform(1, 6) * np.pi * Y + rng.uniform(0, 6.3)) for _ in range(4))
    img = img + texture * tex * brain
    img = np.clip(img, 0, None)
    return img / max(img.max(), 1e-8), brain


def add_lesions(img, brain, rng, n_lesions, radius_range=(1.5, 4.0), contrast_range=(0.1, 0.3)):
    """Adds non-overlapping bright lesions inside the brain. Returns image, mask, lesion list."""
    n = img.shape[0]
    out = img.copy()
    mask = np.zeros_like(img, dtype=bool)
    lesions = []
    yy, xx = np.nonzero(brain)
    for _ in range(200 * max(n_lesions, 1)):
        if len(lesions) == n_lesions:
            break
        r = rng.uniform(*radius_range)
        k = rng.integers(len(yy))
        cy, cx = yy[k], xx[k]
        margin = int(np.ceil(r + 3))
        if not brain[max(cy - margin, 0):cy + margin + 1, max(cx - margin, 0):cx + margin + 1].all():
            continue
        if any(np.hypot(cy - l["cy"], cx - l["cx"]) < r + l["r"] + 4 for l in lesions):
            continue
        c = rng.uniform(*contrast_range)
        prof = lesion_profile(n, cy, cx, r)
        out += c * prof
        mask |= prof > 0.5
        lesions.append({"cy": int(cy), "cx": int(cx), "r": float(r), "contrast": float(c)})
    return np.clip(out, 0, 1), mask, lesions


def phantom_dataset(n_images, size, seed, lesion_rate=0.7, max_lesions=4, **lesion_kw):
    """Batch of phantoms. Each phantom is its own 'patient' (id = index) so splits stay clean."""
    rng = np.random.default_rng(seed)
    imgs, masks, brains, lesion_lists = [], [], [], []
    for _ in range(n_images):
        img, brain = random_phantom(size, rng)
        k = rng.integers(1, max_lesions + 1) if rng.random() < lesion_rate else 0
        img, m, les = add_lesions(img, brain, rng, k, **lesion_kw)
        imgs.append(img); masks.append(m); brains.append(brain); lesion_lists.append(les)
    return (np.stack(imgs).astype(np.float32), np.stack(masks), np.stack(brains), lesion_lists)
