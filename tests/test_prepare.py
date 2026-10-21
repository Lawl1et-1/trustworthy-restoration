import numpy as np

from experiments.prepare_data import _pick_ixi_slices, ixi_head_mask


def _head(n=64, depth=40):
    """Bright scalp ring, dark tissue inside, brighter blobs; the slab is open at both ends in z."""
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - n / 2, xx - n / 2)
    sl = np.where(r < 26, 0.15, 0.0) + np.where((r > 22) & (r < 26), 0.85, 0.0)
    sl = sl + np.where(np.hypot(yy - 28, xx - 30) < 4, 0.6, 0.0)
    vol = np.repeat(sl[:, :, None], depth, axis=2).astype(np.float32)
    return vol + 0.01 * np.random.default_rng(0).random(vol.shape, dtype=np.float32), r < 26


def test_head_mask_is_solid_despite_dark_interior():
    vol, truth = _head()
    m = ixi_head_mask(vol)
    for z in (0, 20, 39):  # including the open ends of the slab
        iou = (m[:, :, z] & truth).sum() / (m[:, :, z] | truth).sum()
        assert iou > 0.9, (z, iou)


def test_ixi_slice_pick_skips_lower_head():
    area = np.r_[np.zeros(5), np.linspace(100, 1000, 50), np.linspace(1000, 300, 45), np.zeros(5)]
    z = _pick_ixi_slices(area)
    zs = np.nonzero(area)[0]
    assert len(z) == 8 and z.min() >= zs[0] + int(0.45 * (zs[-1] - zs[0]))
    assert (area[z] >= 0.5 * area.max()).all()
