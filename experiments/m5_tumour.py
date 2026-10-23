"""M5b: real tumours (MSD test patients). Does a restorer flatten or distort a tumour, and does its uncertainty notice?

  python -m experiments.m5_tumour --config configs/m5_tumour_msd.yaml

The phantom test (m5_hallucination.py) has an exact lesion-free twin of every image; real scans do not. Here
the MSD tumour masks are used instead, on every test slice that contains tumour:
  contrast_kept   (mean restored tumour - mean restored ring) / (same on the truth); the ring is the brain within
                  `ring` px around the tumour. 1 = contrast preserved, < 1 = flattened towards the surroundings.
  err_ratio       RMSE inside the tumour / RMSE in the rest of the brain: > 1 means the method does worse on the
                  tumour than on ordinary tissue (an atypical region being "normalised").
  cov_tumour, cov_rest     share of pixels inside the stated 90 % interval, in the tumour and in the rest of the brain
  spread_ratio    mean stated spread in the tumour / in the rest of the brain (does uncertainty rise there?)
Slices are grouped by tumour area (small < 100 px, medium, large > 600 px at 128 x 128). Degradations are the
M2-M4 test streams (task_generator(task, "test", dseed)), so inputs equal those of the main tables. Methods are
loaded, not retrained (M2 weights, M3/M4 checkpoints). Blocks are cached so an interrupted run resumes.

Writes results/<profile>/m5_tumour/msd/{per_image.csv.gz, summary.csv, meta.json}.
"""
import argparse
import json
import pathlib
import platform
import shutil
import time

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import ndimage

from experiments.m3_nets import Z90, load_block, save_block, type_tasks
from experiments.m5_hallucination import classical_methods, net_methods, run
from experiments.run_task import _git_hash
from src.data.datasets import ROOT, load_dataset
from src.degradations.degrade import degrade, task_generator
from src.eval.stats import bootstrap_ci

SIZE_BINS = [(0, 100, "small"), (100, 600, "medium"), (600, 10 ** 9, "large")]


def masks(sd, ring):
    tum = sd.labels & sd.region
    rings = np.array([ndimage.binary_dilation(t, iterations=ring) & ~t & r for t, r in zip(tum, sd.region)])
    rest = sd.region & ~tum
    return tum, rings, rest


def image_rows(x, xh, unc, quant, tum, rings, rest):
    out = []
    for b in range(len(x)):
        t, rg, rs = tum[b], rings[b], rest[b]
        ct = x[b][t].mean() - x[b][rg].mean() if rg.any() else np.nan
        ch = xh[b][t].mean() - xh[b][rg].mean() if rg.any() else np.nan
        e = x[b] - xh[b]
        r = {"area": int(t.sum()),
             "contrast_kept": float(ch / ct) if rg.any() and abs(ct) > 0.02 else np.nan,
             "err_ratio": float(np.sqrt((e[t] ** 2).mean()) / max(np.sqrt((e[rs] ** 2).mean()), 1e-6)),
             "rmse_tumour": float(np.sqrt((e[t] ** 2).mean())),
             "cov_tumour": np.nan, "cov_rest": np.nan, "spread_ratio": np.nan}
        if unc is not None:
            spread, lo, hi = unc
            lam = 1.0 if quant else Z90
            inside = (e >= -lam * lo[b]) & (e <= lam * hi[b])
            r["cov_tumour"], r["cov_rest"] = float(inside[t].mean()), float(inside[rs].mean())
            r["spread_ratio"] = float(spread[b][t].mean() / max(spread[b][rs].mean(), 1e-9))
        out.append(r)
    return out


def summarise(df):
    rows = []
    df = df.assign(size=pd.cut(df.area, [b[0] for b in SIZE_BINS] + [SIZE_BINS[-1][1]], right=False,
                               labels=[b[2] for b in SIZE_BINS]).astype(str))
    for (task, method, size), g in pd.concat([df.assign(size="all"), df]).groupby(["task", "method", "size"], sort=False):
        r = {"task": task, "method": method, "size": size, "n_slices": len(g) // max(g.model_seed.nunique(), 1),
             "n_patients": g.pid.nunique()}
        for c in ["contrast_kept", "err_ratio", "rmse_tumour", "cov_tumour", "cov_rest", "spread_ratio"]:
            m, lo, hi = bootstrap_ci(g[c].values, g.pid.values)
            r[c], r[c + "_lo"], r[c + "_hi"] = m, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def main(cfg_path):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    dev = "cuda" if torch.cuda.is_available() and not cfg.get("cpu") else "cpu"
    print("device:", dev, flush=True)
    t_start = time.time()
    test = load_dataset(cfg["data"])["test"]
    keep = np.nonzero((test.labels & test.region).sum((1, 2)) >= cfg.get("min_area", 10))[0]
    test = test.take(keep)
    tum, rings, rest = masks(test, cfg.get("ring", 5))
    print(f"{len(test)} test slices with tumour from {len(np.unique(test.pid))} patients", flush=True)
    out_dir = ROOT / "results" / cfg["profile"] / cfg.get("out_name", "m5_tumour") / cfg["data"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / ".partial"
    classical = classical_methods(cfg, cfg["data"]["name"])
    bs = cfg.get("eval_batch", 32)
    x = test.images
    xt = torch.from_numpy(x)[:, None]
    rows = []
    for dtype in sorted({s["type"] for s in cfg["tasks"].values()}):
        methods = [(nm, -1, None, {"weights": w}) for nm, w in classical.get(dtype, [])] + net_methods(cfg, dtype, dev)
        for task, spec in type_tasks(cfg["tasks"], dtype).items():
            for ds in cfg["eval_dseeds"]:
                bkey = f"tumour|{task}|{ds}|" + "|".join(f"{m[0]}:{m[1]}" for m in methods)
                done = load_block(cache, bkey)
                if done is not None:
                    rows.extend(done["rows"]); print(f"  {task} d{ds}: reused from cache", flush=True); continue
                n0 = len(rows)
                d = degrade(xt, spec, task_generator(task, "test", ds))
                inp = d["inp"].clamp(0, 1)[:, 0].numpy()
                for i, r in enumerate(image_rows(x, inp, None, None, tum, rings, rest)):
                    rows.append({"task": task, "method": "input", "model_seed": -1, "seed": ds, "idx": i,
                                 "pid": test.pid[i], **r})
                for meth in methods:
                    t0 = time.time()
                    for name, (xh, unc, quant) in run(meth, d, spec, task, dev, bs, f"tumour|{task}|{ds}|{meth[1]}").items():
                        rr = image_rows(x, xh, unc, quant, tum, rings, rest)
                        for i, r in enumerate(rr):
                            rows.append({"task": task, "method": name, "model_seed": meth[1], "seed": ds, "idx": i,
                                         "pid": test.pid[i], **r})
                        g = pd.DataFrame(rr)
                        print(f"  {task} d{ds} {name:20s} m{meth[1]:2d} contrast kept {g.contrast_kept.mean():.3f} "
                              f"err ratio {g.err_ratio.mean():.2f} cov tumour {g.cov_tumour.mean():.3f} "
                              f"({time.time() - t0:.0f} s)", flush=True)
                save_block(cache, bkey, rows=rows[n0:])
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_image.csv.gz", index=False)
    summ = summarise(df)
    summ.to_csv(out_dir / "summary.csv", index=False)
    meta = {"config": cfg, "git": _git_hash(), "python": platform.python_version(), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "wall_seconds": round(time.time() - t_start, 1), "n_slices": int(len(test)),
            "n_patients": int(len(np.unique(test.pid)))}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    cols = ["task", "method", "size", "contrast_kept", "err_ratio", "cov_tumour", "cov_rest", "spread_ratio"]
    print(summ[summ["size"] == "all"][cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {out_dir} in {meta['wall_seconds']} s", flush=True)
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
