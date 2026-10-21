import numpy as np

from src.phantoms.phantom import phantom_dataset


def test_phantom_lesions_inside_brain_and_known():
    imgs, masks, brains, les = phantom_dataset(20, 64, seed=0, lesion_rate=1.0, max_lesions=3)
    assert imgs.shape == (20, 64, 64) and imgs.min() >= 0 and imgs.max() <= 1
    for m, b, l in zip(masks, brains, les):
        assert (m <= b).all()
        assert len(l) >= 1
        for d in l:
            assert b[d["cy"], d["cx"]]


def test_phantom_deterministic():
    a = phantom_dataset(3, 32, seed=5)[0]
    b = phantom_dataset(3, 32, seed=5)[0]
    assert np.array_equal(a, b)
