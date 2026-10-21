"""M2: classical baselines on every (degradation, severity), with weights tuned on the val split.

  python -m experiments.run_task --config configs/smoke.yaml

Protocol
  * Regularisation weight: grid search on `tune_val_slices` val slices (degradation seed 0, val
    stream), maximising mean PSNR. The test split is never looked at while tuning.
  * Test: for each degradation seed the test set is degraded once (task_generator) and every method
    restores the same degraded images, so per-image differences between methods are paired.
  * Outputs are clipped to [0, 1] before scoring (intensities are in [0, 1] by construction).
  * CIs: per image, metrics are averaged over seeds; then a patient-level bootstrap.

Writes results/<profile>/m2_classical/<dataset>/{per_image.csv, summary.csv, tuned_weights.json, meta.json}.
"""
import os

# One numeric thread per process: the pool already uses every core, and 7 workers x 8 BLAS threads
# made BM3D ~20x slower than its single-thread speed (LOG.md, M2). Set before numpy is imported;
# spawned workers re-import this module, so they inherit it too.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pathlib
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.datasets import ROOT, load_dataset
from src.degradations.degrade import degrade, task_generator
from src.eval import metrics as M
from src.eval.stats import bootstrap_ci
from src.models.classical import CLASSICAL, WEIGHT_GRID, restore

FIDELITY = ["psnr", "ssim", "nrmse"]
PERCEPTION = ["grad_energy", "laplacian_var"]
LINEAR = ["null_space_energy", "dc_violation"]


def _slice_d(d, idx):
    """Restricts a degradation dict to a subset of the batch."""
    from src.degradations.operators import LinearOp
    out = dict(d)
    for k in ("inp", "ind", "y"):
        if d[k] is not None:
            out[k] = d[k][idx]
    if d["op"] is not None:
        out["op"] = LinearOp(d["op"].kind, d["op"].mask[idx])
    return out


def _restore_chunk(args):
    torch.set_num_threads(1)
    name, d, w = args
    t0 = time.perf_counter()
    out = restore(name, d, w).clamp(0, 1)
    return out, time.perf_counter() - t0


def run_method(pool, name, d, w, chunk=2):
    n = d["inp"].shape[0]
    if name == "bm3d":
        # the bm3d library parallelises internally; running copies side by side made it 7-12x slower
        # per image (1.2 s with 1 worker, 8-14 s with 2-7 workers; LOG.md, M2), so it runs serially
        pool = None
    jobs = [(name, _slice_d(d, slice(i, i + chunk)), w) for i in range(0, n, chunk)]
    res = list(pool.map(_restore_chunk, jobs)) if pool else [_restore_chunk(j) for j in jobs]
    return torch.cat([r[0] for r in res]), sum(r[1] for r in res) / n


def score(x, xh, region, d):
    """Per-image metrics; x, xh numpy (B, H, W)."""
    s = {"psnr": M.psnr(x, xh), "ssim": M.ssim(x, xh), "nrmse": M.nrmse(x, xh),
         "grad_energy": M.grad_energy(xh), "laplacian_var": M.laplacian_var(xh)}
    if d["op"] is not None:
        s["null_space_energy"] = M.null_space_energy(d["op"], x, xh)
        s["dc_violation"] = M.dc_violation(d["op"], d["y"], x, xh)
    return s


def tune(pool, name, ttype, spec, task, val, n_val):
    grid = WEIGHT_GRID[name]
    if len(grid) == 1:
        return grid[0], {}
    v = val.head(n_val)
    x = torch.from_numpy(v.images)[:, None]
    d = degrade(x, spec, task_generator(task, "val", 0))
    curve = {}
    for w in grid:
        xh, _ = run_method(pool, name, d, w)
        curve[w] = float(M.psnr(v.images, xh[:, 0].numpy()).mean())
    best = max(curve, key=curve.get)
    if best in (grid[0], grid[-1]):
        print(f"  WARNING {task}/{name}: tuned weight {best} is on the edge of the grid", flush=True)
    return best, curve


def evaluate(pool, data, split_name, tasks, seeds, weights, rows, log_tag):
    x_np = data.images
    x = torch.from_numpy(x_np)[:, None]
    for task, spec in tasks.items():
        for seed in seeds:
            d = degrade(x, spec, task_generator(task, split_name, seed))
            for name in CLASSICAL[spec["type"]]:
                xh, sec = run_method(pool, name, d, weights[task][name])
                s = score(x_np, xh[:, 0].numpy(), data.region, d)
                for i in range(len(x_np)):
                    rows.append({"split": split_name, "task": task, "type": spec["type"], "method": name,
                                 "seed": seed, "weight": weights[task][name], "idx": i, "pid": data.pid[i],
                                 "site": data.site[i], "sec_per_img": sec, **{k: float(v[i]) for k, v in s.items()}})
                print(f"  [{log_tag}] {task} seed{seed} {name:12s} PSNR {s['psnr'].mean():6.2f} "
                      f"SSIM {s['ssim'].mean():.3f} ({sec * 1e3:.0f} ms/img)", flush=True)


def summarise(df):
    out = []
    metrics = [c for c in FIDELITY + PERCEPTION + LINEAR if c in df.columns]
    for (split, task, method), g in df.groupby(["split", "task", "method"], sort=False):
        per_img = g.groupby("idx").agg({**{m: "mean" for m in metrics}, "pid": "first"})
        row = {"split": split, "task": task, "method": method, "weight": g["weight"].iloc[0],
               "n_images": len(per_img), "n_patients": per_img["pid"].nunique(), "n_seeds": g["seed"].nunique()}
        for m in metrics:
            if per_img[m].notna().any():
                mu, lo, hi = bootstrap_ci(per_img[m].values, per_img["pid"].values)
                row.update({m: mu, f"{m}_lo": lo, f"{m}_hi": hi})
        out.append(row)
    return pd.DataFrame(out)


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main(cfg_path, workers):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    t_start = time.time()
    data = load_dataset(cfg["data"])
    tasks = cfg["tasks"]
    print({s: (len(d), len(set(d.pid))) for s, d in data.items()}, "(slices, patients)", flush=True)
    out_dir = ROOT / "results" / cfg["profile"] / "m2_classical" / cfg["data"].get("tag", cfg["data"]["name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        weights, curves = {}, {}
        for task, spec in tasks.items():
            weights[task], curves[task] = {}, {}
            for name in CLASSICAL[spec["type"]]:
                weights[task][name], curves[task][name] = tune(pool, name, spec["type"], spec, task, data["val"],
                                                               cfg.get("tune_val_slices", 32))
            print(f"tuned {task}: {weights[task]}", flush=True)
        rows = []
        evaluate(pool, data["test"], "test", tasks, cfg["seeds"], weights, rows, "test")
        ood = cfg.get("ood")
        if ood:  # out-of-distribution test sets: weights stay those tuned in distribution
            for o in ood:
                try:
                    od = load_dataset(o)["test"]
                except FileNotFoundError as e:
                    print("skip OOD:", e, flush=True)
                    continue
                evaluate(pool, od, "ood", tasks, cfg["seeds"], weights, rows, o.get("tag", o["name"]))
    finally:
        if pool:
            pool.shutdown()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_image.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "tuned_weights.json").write_text(json.dumps({"weights": weights, "val_psnr_curves": curves}, indent=1))
    meta = {"config": cfg, "git": _git_hash(), "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "workers": workers, "wall_seconds": round(time.time() - t_start, 1),
            "n_slices": {s: len(d) for s, d in data.items()}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    cols = ["split", "task", "method", "psnr", "psnr_lo", "psnr_hi", "ssim"]
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {out_dir} in {meta['wall_seconds']} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    main(a.config, a.workers)
