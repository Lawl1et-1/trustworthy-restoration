import json

import numpy as np
import pytest

from src.data.datasets import load_dataset
from src.data.splits import get_or_make_split, load_split, make_patient_split, split_indices


def _pids(n_patients=50, slices=8):
    """MSD-like: several slices per patient, interleaved so slice order != patient order."""
    return np.array([f"BRATS_{p:03d}" for _ in range(slices) for p in range(n_patients)])


def test_split_is_patient_level_and_complete():
    pids = _pids()
    split = make_patient_split(pids, seed=0)
    idx = split_indices(pids, split)
    sets = {s: set(pids[i]) for s, i in idx.items()}
    names = list(sets)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            assert not sets[names[a]] & sets[names[b]], (names[a], names[b])
    assert sorted(np.concatenate(list(idx.values()))) == list(range(len(pids)))  # every slice, once
    for s, i in idx.items():  # all slices of a patient stay together
        for p in sets[s]:
            assert (pids[i] == p).sum() == (pids == p).sum()


def test_split_fractions_and_determinism():
    pids = _pids(100)
    a, b = make_patient_split(pids, seed=3), make_patient_split(pids, seed=3)
    assert a == b
    assert {s: len(v) for s, v in a.items()} == {"train": 60, "val": 10, "cal": 15, "test": 15}
    assert make_patient_split(pids, seed=4) != a


def test_saved_split_roundtrip_and_tamper_check(tmp_path):
    pids = _pids(20)
    p = tmp_path / "s.json"
    s1 = get_or_make_split(p, pids, seed=0)
    assert get_or_make_split(p, pids, seed=99) == s1  # the saved file wins over a new seed
    doc = json.loads(p.read_text())
    doc["patients"]["test"].append(doc["patients"]["train"].pop())  # move a patient
    p.write_text(json.dumps(doc))
    with pytest.raises(AssertionError):
        load_split(p)


def test_split_rejects_overlap():
    with pytest.raises(AssertionError):
        split_indices(np.array(["a", "b"]), {"train": ["a", "b"], "test": ["b"]})


def test_phantom_loader_uses_patient_split(tmp_path):
    d = load_dataset({"name": "phantom", "n_images": 40, "size": 32}, split_dir=tmp_path)
    assert sum(len(v) for v in d.values()) == 40
    ids = [set(v.pid) for v in d.values()]
    assert all(not (a & b) for i, a in enumerate(ids) for b in ids[i + 1:])
    assert len(d["cal"]) > 0 and len(d["test"]) > 0
    t = d["test"]
    assert t.images.shape[1:] == (32, 32) and len(t.lesions) == len(t)
