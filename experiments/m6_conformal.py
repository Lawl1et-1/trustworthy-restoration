"""M6: distribution-free calibration of every method's per-pixel intervals on held-out calibration patients.

  python -m experiments.m6_conformal --config configs/m6_full_msd.yaml

Protocol
  * Interval of a method: [mean - lam * lo, mean + lam * hi], lo / hi its stated spreads (M3 interface).
    Point predictors (unet, unrolled, gan) get lo = hi = 1, i.e. a constant-width split-conformal band.
  * Risk of an image = fraction of its brain pixels outside the interval (src/conformal/risk.py).
  * lam is chosen on the CALIBRATION patients (degradation seed 0 of the cal stream) by
      CRC  (Angelopoulos et al. 2024): expected risk <= alpha
      RCPS (Bates et al. 2021, Hoeffding-Bentkus): P(expected risk <= alpha) >= 1 - delta
    and applied unchanged to the TEST patients and to the out-of-distribution set (IXI), which were never
    used for anything else. "raw" is the method's own 90 % interval (lam = z_0.95, or 1 for quantile).
  * Guarantees assume calibration and test images are exchangeable; IXI breaks that on purpose.

Writes results/<profile>/m6_conformal/<dataset>/{per_image.csv.gz, summary.csv, lambdas.json, meta.json}.
"""
import argparse
import json
import shutil
import pathlib
import platform
import time

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.m3_nets import Z90, load_block, predict_all, save_block, type_tasks
from experiments.m5_hallucination import net_methods
from experiments.run_task import _git_hash
from src.conformal.risk import LAMBDA_GRID, crc_lambda, rcps_lambda, risk_curves
from src.data.datasets import ROOT, load_dataset
from src.degradations.degrade import degrade, task_generator
from src.eval.stats import bootstrap_ci

POINT = ("unet", "unrolled", "gan")


def intervals(model_entry, d, spec, dev, bs, key):
    """{name: (mean, lo, hi, raw_lambda)} numpy (B, H, W) for one method on one degraded batch."""
    name, mseed, m, kw = model_entry
    p = predict_all(m, d, spec, dev, kw.get("bs", bs), mc=kw.get("mc", 0), gen_kw=kw.get("gen_kw"), seed_key=key)
    mean = p["mean"].clamp(0, 1)[:, 0].numpy()
    out = {}
    if name in POINT:
        one = np.ones_like(mean)
        out[name + "_const"] = (mean, one, one, np.nan)
    else:
        raw = 1.0 if name == "quantile" else Z90
        out[name] = (mean, p["lo"][:, 0].numpy(), p["hi"][:, 0].numpy(), raw)
    if "sample" in p:  # a single diffusion draw with a constant band, the way a user would see it
        s = p["sample"].clamp(0, 1)[:, 0].numpy()
        out[name + "_sample_const"] = (s, np.ones_like(s), np.ones_like(s), np.nan)
    return out


def nearest(lam):
    return int(np.argmin(np.abs(LAMBDA_GRID - lam)))


def main(cfg_path):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    dev = "cuda" if torch.cuda.is_available() and not cfg.get("cpu") else "cpu"
    print("device:", dev, flush=True)
    alpha, delta = cfg.get("alpha", 0.1), cfg.get("delta", 0.1)
    t_start = time.time()
    data = load_dataset(cfg["data"])
    ds_tag = cfg["data"].get("tag", cfg["data"]["name"])
    splits = {"cal": data["cal"], "test": data["test"]}
    for o in cfg.get("ood", []):
        try:
            splits["ood"] = load_dataset(o)["test"]
        except FileNotFoundError as e:
            print("skip OOD:", e, flush=True)
    out_dir = ROOT / "results" / cfg["profile"] / cfg.get("out_name", "m6_conformal") / ds_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    bs = cfg.get("eval_batch", 32)
    rows, lambdas = [], {}
    cache = out_dir / ".partial"  # finished blocks; removed once the results are written
    for dtype in sorted({s["type"] for s in cfg["tasks"].values()}):
        methods = net_methods(cfg, dtype, dev)
        for task, spec in type_tasks(cfg["tasks"], dtype).items():
            bkey = f"m6|{task}|{sorted(splits)}|" + "|".join(f"{m[0]}:{m[1]}" for m in methods)
            done = load_block(cache, bkey)
            if done is not None:  # finished before an interruption
                rows.extend(done["rows"]); lambdas[task] = done["lambdas"]
                print(f"  {task}: reused from cache", flush=True)
                continue
            n_rows = len(rows)
            curves = {}  # (split, name, mseed) -> (risk, width) over the lambda grid
            for split, sd in splits.items():
                x = torch.from_numpy(sd.images)[:, None]
                d = degrade(x, spec, task_generator(task, split, 0))
                for meth in methods:
                    for name, (mean, lo, hi, raw) in intervals(meth, d, spec, dev, bs, f"m6|{task}|{split}|{meth[1]}").items():
                        curves[(split, name, meth[1])] = risk_curves(sd.images, mean, lo, hi, sd.region) + (raw,)
            for (split, name, mseed), (risk, width, raw) in curves.items():
                if split != "cal":
                    continue
                choice = {"crc": crc_lambda(risk, alpha), "rcps": rcps_lambda(risk, alpha, delta)}
                if not np.isnan(raw):
                    choice["raw"] = nearest(raw)
                lambdas.setdefault(task, {})[f"{name}|{mseed}"] = {k: float(LAMBDA_GRID[j]) for k, j in choice.items()}
                for ev in [s for s in splits if s != "cal"]:
                    r_ev, w_ev, _ = curves[(ev, name, mseed)]
                    sd = splits[ev]
                    for rule, j in choice.items():
                        for i in range(len(sd)):
                            rows.append({"split": ev, "task": task, "method": name, "model_seed": mseed, "rule": rule,
                                         "lam": float(LAMBDA_GRID[j]), "idx": i, "pid": sd.pid[i], "site": sd.site[i],
                                         "miss": float(r_ev[i, j]), "width": float(w_ev[i, j])})
                    msg = "  ".join(f"{rule} cov {1 - r_ev[:, j].mean():.3f} w {w_ev[:, j].mean():.3f}"
                                    for rule, j in choice.items())
                    print(f"  {task} {ev:4s} {name:22s} m{mseed:2d}  {msg}", flush=True)
            save_block(cache, bkey, rows=rows[n_rows:], lambdas=lambdas.get(task, {}))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_image.csv.gz", index=False)
    summ = []
    for (split, task, method, rule), g in df.groupby(["split", "task", "method", "rule"], sort=False):
        cov = bootstrap_ci(1 - g.miss.values, g.pid.values)
        summ.append({"split": split, "task": task, "method": method, "rule": rule, "lam": g.lam.mean(),
                     "coverage": cov[0], "coverage_lo": cov[1], "coverage_hi": cov[2], "width": g.width.mean(),
                     "frac_images_over_alpha": float((g.miss > alpha).mean()), "n_images": len(g) // g.model_seed.nunique(),
                     "n_model_seeds": g.model_seed.nunique()})
    summ = pd.DataFrame(summ)
    summ.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "lambdas.json").write_text(json.dumps({"alpha": alpha, "delta": delta, "lambdas": lambdas}, indent=1))
    meta = {"config": cfg, "git": _git_hash(), "python": platform.python_version(), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "wall_seconds": round(time.time() - t_start, 1), "n_slices": {s: len(d) for s, d in splits.items()}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(summ[["split", "task", "method", "rule", "coverage", "width"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {out_dir} in {meta['wall_seconds']} s", flush=True)
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
