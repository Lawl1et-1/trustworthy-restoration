"""Patient-level splits. Every slice of a patient goes to exactly one of train / val / cal / test.

`cal` is the conformal calibration set: disjoint from the training data (train + val, val being used
for early stopping and hyper-parameter tuning) and from test. Splits are drawn once from a fixed split
seed, independent of the training seeds, and saved to JSON so every experiment reads the same file.
"""
import hashlib
import json
import pathlib

import numpy as np

SPLITS = ("train", "val", "cal", "test")
DEFAULT_FRACS = {"train": 0.60, "val": 0.10, "cal": 0.15, "test": 0.15}


def make_patient_split(pids, fracs=None, seed=0):
    """pids: per-slice patient ids. Returns {split: sorted list of patient ids}."""
    fracs = dict(DEFAULT_FRACS if fracs is None else fracs)
    assert abs(sum(fracs.values()) - 1) < 1e-9, fracs
    uniq = np.array(sorted(set(map(str, pids))))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    n = len(uniq)
    # cumulative rounding, so the counts always add up to n
    edges = np.round(np.cumsum([0] + [fracs[s] for s in SPLITS]) * n).astype(int)
    out = {s: sorted(uniq[perm[edges[i]:edges[i + 1]]].tolist()) for i, s in enumerate(SPLITS)}
    assert_disjoint(out)
    return out


def assert_disjoint(split):
    seen = {}
    for s, ids in split.items():
        for p in ids:
            assert p not in seen, f"patient {p} is in both {seen[p]} and {s}"
            seen[p] = s


def split_indices(pids, split):
    """Maps a patient split to slice indices; asserts every slice is assigned and none twice."""
    assert_disjoint(split)
    owner = {p: s for s, ids in split.items() for p in ids}
    pids = np.asarray(pids).astype(str)
    missing = set(pids) - set(owner)
    assert not missing, f"{len(missing)} patients not in the split, e.g. {sorted(missing)[:3]}"
    return {s: np.nonzero([owner[p] == s for p in pids])[0] for s in split}


def _digest(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def save_split(path, split, meta):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"meta": meta, "counts": {s: len(v) for s, v in split.items()},
           "sha256_16": {s: _digest(v) for s, v in split.items()}, "patients": split}
    path.write_text(json.dumps(doc, indent=1))


def load_split(path):
    doc = json.loads(pathlib.Path(path).read_text())
    split = doc["patients"]
    assert_disjoint(split)
    for s, v in split.items():
        assert _digest(v) == doc["sha256_16"][s], f"split file {path} was edited ({s})"
    return split, doc["meta"]


def get_or_make_split(path, pids, fracs=None, seed=0, meta=None):
    """Loads the saved split if it exists (and checks it covers exactly these patients), else makes it."""
    path = pathlib.Path(path)
    if path.exists():
        split, _ = load_split(path)
        have = set().union(*map(set, split.values()))
        assert have == set(map(str, pids)), f"{path} does not match this dataset; delete it to re-split"
        return split
    split = make_patient_split(pids, fracs, seed)
    save_split(path, split, {"seed": seed, "fracs": dict(DEFAULT_FRACS if fracs is None else fracs), **(meta or {})})
    return split
