import numpy as np

from experiments.m5_hallucination import image_scores, lesion_free_twins, lesion_rows
from src.data.datasets import SliceData
from src.phantoms.phantom import phantom_dataset


def _data():
    imgs, masks, brains, les = phantom_dataset(6, 64, seed=3, lesion_rate=1.0)
    n = len(imgs)
    return SliceData(imgs, masks, brains, np.arange(n).astype(str), np.array(["phantom"] * n), les)


def test_lesion_free_twin_removes_exactly_the_lesions():
    d = _data()
    x_bg, clipped = lesion_free_twins(d)
    ref, _, _, _ = phantom_dataset(6, 64, seed=3, lesion_rate=1.0)
    assert np.allclose(ref, d.images)
    for i, les in enumerate(d.lesions):
        assert len(les) > 0
        far = ~np.isclose(d.images[i], x_bg[i], atol=1e-4)
        assert far.sum() > 0 and far[~d.region[i]].sum() == 0  # only brain pixels change
    assert all(isinstance(c, bool) for cl in clipped for c in cl)


def test_perfect_restoration_scores_full_recovery_and_no_fabrication():
    d = _data()
    x_bg, clipped = lesion_free_twins(d)
    x = d.images
    sd = np.full_like(x, 0.05)
    rows = lesion_rows(x, x_bg, x, x_bg, (sd, sd, sd), False, d.lesions, clipped, d.region)
    ok = [r for r in rows if not r["clipped"]]
    assert all(abs(r["recovery"] - 1) < 1e-3 and abs(r["t_absent"]) < 1e-6 for r in ok)
    assert all(r["core_cov90"] == 1.0 for r in rows)
    s = image_scores(x, x_bg, x, x_bg, (sd, sd, sd), d.labels, d.region)
    assert (s["fab_present"] == 0).all() and (s["fab_absent"] == 0).all() and (s["era_present"] == 0).all()


def test_erasing_restorer_is_caught():
    d = _data()
    x_bg, clipped = lesion_free_twins(d)
    rows = lesion_rows(d.images, x_bg, x_bg, x_bg, None, False, d.lesions, clipped, d.region)
    assert all(abs(r["recovery"]) < 1e-3 for r in rows if not r["clipped"])  # output = lesion-free image
