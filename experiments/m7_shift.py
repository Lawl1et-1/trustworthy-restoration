"""M7: why do restorers and their uncertainty break when moving from MSD to IXI?

  python -m experiments.m7_shift --config configs/m7_full.yaml

Three suspected causes, each tested with the M2-M4 models unchanged (nothing is retrained):
  noise   IXI images carry their own acquisition noise, MSD's (preprocessed BraTS) do not. MSD test images get
          Rician noise at IXI's level (estimated from IXI backgrounds, worst site) and are used as both the
          source and the reference, as IXI images are. If noise explained the shift, gains would drop as on IXI.
  skull   IXI is not skull-stripped and its evaluation region is the whole head. Every split is scored on the
          full region and on its interior (region eroded by `erode` px, which removes skull and scalp on IXI).
          If the skull explained the shift, the IXI gains would recover on the interior.
  site    gains and coverage per IXI hospital; and per-site recalibration: lam re-chosen by CRC on n_cal
          subjects of one site and tested on that site's other subjects (an ablation that uses IXI data for
          calibration, kept separate from every main result), over `n_splits` random subject splits.

Writes results/<profile>/m7_shift/{per_image.csv.gz, region.csv, site.csv, recal.csv, noise.json, meta.json}.
"""
import argparse
import json
import pathlib
import platform
import time

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import ndimage

from experiments.m3_nets import Z90, type_tasks
from experiments.m5_hallucination import classical_methods, net_methods, run
from experiments.run_task import _git_hash
from src.conformal.risk import LAMBDA_GRID, crc_lambda, risk_curves
from src.data.datasets import ROOT, SliceData, load_dataset
from src.degradations.degrade import degrade, rician, task_generator


def background_sigma(sd, dilate=6, border=4):
    """Per-site median Rayleigh-sigma of the background (outside the dilated head mask)."""
    out = {}
    for s in np.unique(sd.site):
        vals = []
        for i in np.nonzero(sd.site == s)[0]:
            bg = ~ndimage.binary_dilation(sd.region[i], iterations=dilate)
            bg[:border] = bg[-border:] = False
            bg[:, :border] = bg[:, -border:] = False
            v = sd.images[i][bg]
            if v.size > 200:
                vals.append(np.sqrt((v.astype(np.float64) ** 2).mean() / 2))
        out[str(s)] = float(np.median(vals))
    return out


def with_acquisition_noise(sd, sigma, seed):
    g = torch.Generator().manual_seed(seed)
    x = rician(torch.from_numpy(sd.images)[:, None], sigma, g)[:, 0].clamp(0, 1).numpy().astype(np.float32)
    return SliceData(x, sd.labels, sd.region, sd.pid, sd.site)


def psnr_in(x, xh, mask):
    mse = np.array([((a - b)[m] ** 2).mean() if m.any() else np.nan for a, b, m in zip(x, xh, mask)])
    return 10 * np.log10(1 / np.maximum(mse, 1e-12))


def cov_in(x, mean, unc, quant, mask):
    if unc is None:
        return np.full(len(x), np.nan)
    _, lo, hi = unc
    lam = 1.0 if quant else Z90
    e = x - mean
    return np.array([(((ee >= -lam * l) & (ee <= lam * h))[m]).mean() if m.any() else np.nan
                     for ee, l, h, m in zip(e, lo, hi, mask)])


def keep(methods, max_seed):
    """One model seed per method (seed 0), the ensemble of all hetero seeds, no MC-dropout."""
    return [m for m in methods if m[0] in ("ensemble",) or (m[0] != "mcdropout" and m[0] != "hetero" and m[1] <= max_seed)]


def recalibrate(curves, sd, alpha, n_cals, n_splits, seed):
    """Per-site CRC on n_cal subjects, coverage on the site's remaining subjects."""
    rng = np.random.default_rng(seed)
    rows = []
    for site in np.unique(sd.site):
        idx_site = np.nonzero(sd.site == site)[0]
        subj = np.unique(sd.pid[idx_site])
        for n_cal in n_cals:
            if n_cal >= len(subj):
                continue
            for k in range(n_splits):
                cal_s = rng.choice(subj, n_cal, replace=False)
                cal = idx_site[np.isin(sd.pid[idx_site], cal_s)]
                ev = idx_site[~np.isin(sd.pid[idx_site], cal_s)]
                for (task, method), (risk, width) in curves.items():
                    j = crc_lambda(risk[cal], alpha)
                    rows.append({"site": site, "task": task, "method": method, "n_cal": n_cal, "split": k,
                                 "lam": float(LAMBDA_GRID[j]), "coverage": float(1 - risk[ev, j].mean()),
                                 "width": float(width[ev, j].mean())})
    return pd.DataFrame(rows)


def main(cfg_path):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    dev = "cuda" if torch.cuda.is_available() and not cfg.get("cpu") else "cpu"
    print("device:", dev, flush=True)
    t_start = time.time()
    msd = load_dataset(cfg["data"])["test"]
    ixi = load_dataset(cfg["ood"])["test"]
    noise = background_sigma(ixi)
    sig = max(noise.values()) * cfg.get("noise_factor", 1.0)
    print("IXI background sigma per site:", {k: round(v, 4) for k, v in noise.items()}, "-> MSD + noise", round(sig, 4),
          flush=True)
    splits = {"msd": msd, "msd_acqnoise": with_acquisition_noise(msd, sig, cfg.get("noise_seed", 7)), "ixi": ixi}
    interior = {k: np.array([ndimage.binary_erosion(r, iterations=cfg.get("erode", 8)) for r in v.region])
                for k, v in splits.items()}
    out_dir = ROOT / "results" / cfg["profile"] / cfg.get("out_name", "m7_shift")
    out_dir.mkdir(parents=True, exist_ok=True)
    classical = classical_methods(cfg, cfg["data"].get("tag", cfg["data"]["name"]))
    bs = cfg.get("eval_batch", 32)
    rows, curves = [], {}
    for dtype in sorted({s["type"] for s in cfg["tasks"].values()}):
        methods = [(nm, -1, None, {"weights": w}) for nm, w in classical.get(dtype, [])]
        methods += keep(net_methods(cfg, dtype, dev), cfg.get("max_model_seed", 0))
        for task, spec in type_tasks(cfg["tasks"], dtype).items():
            for sname, sd in splits.items():
                x = sd.images
                gen_split = "ood" if sname == "ixi" else "test"  # the M2-M4 test / OOD degradation streams
                d = degrade(torch.from_numpy(x)[:, None], spec, task_generator(task, gen_split, 0))
                inp = d["inp"].clamp(0, 1)[:, 0].numpy()
                in_full, in_int = psnr_in(x, inp, sd.region), psnr_in(x, inp, interior[sname])
                for meth in methods:
                    t0 = time.time()
                    for name, (xh, unc, quant) in run(meth, d, spec, task, dev, bs, f"m7|{task}|{sname}|{meth[1]}").items():
                        pf, pi = psnr_in(x, xh, sd.region), psnr_in(x, xh, interior[sname])
                        cf, ci = cov_in(x, xh, unc, quant, sd.region), cov_in(x, xh, unc, quant, interior[sname])
                        for i in range(len(x)):
                            rows.append({"split": sname, "task": task, "method": name, "model_seed": meth[1],
                                         "idx": i, "pid": sd.pid[i], "site": sd.site[i],
                                         "psnr_full": pf[i], "psnr_interior": pi[i],
                                         "gain_full": pf[i] - in_full[i], "gain_interior": pi[i] - in_int[i],
                                         "cov_full": cf[i], "cov_interior": ci[i]})
                        if sname == "ixi" and not name.endswith("_sample"):
                            lo = unc[1] if unc is not None else np.ones_like(xh)
                            hi = unc[2] if unc is not None else np.ones_like(xh)
                            label = name if unc is not None else name + "_const"
                            curves[(task, label)] = risk_curves(x, xh, lo, hi, sd.region)
                        print(f"  {task} {sname:12s} {name:16s} gain full {np.nanmean(pf - in_full):6.2f} "
                              f"interior {np.nanmean(pi - in_int):6.2f} cov {np.nanmean(cf):.3f} "
                              f"({time.time() - t0:.0f} s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_image.csv.gz", index=False)
    g = df.groupby(["split", "task", "method"], sort=False)
    region = g[["gain_full", "gain_interior", "cov_full", "cov_interior"]].mean().reset_index()
    region.to_csv(out_dir / "region.csv", index=False)
    site = df[df.split == "ixi"].groupby(["site", "task", "method"], sort=False)[
        ["gain_full", "gain_interior", "cov_full"]].mean().reset_index()
    site.to_csv(out_dir / "site.csv", index=False)
    rec = recalibrate(curves, ixi, cfg.get("alpha", 0.1), cfg.get("n_cals", [3, 5, 10, 20]), cfg.get("n_splits", 50),
                      cfg.get("recal_seed", 0))
    recs = rec.groupby(["site", "task", "method", "n_cal"]).agg(
        coverage=("coverage", "mean"), coverage_p10=("coverage", lambda v: np.percentile(v, 10)),
        width=("width", "mean")).reset_index()
    recs.to_csv(out_dir / "recal.csv", index=False)
    (out_dir / "noise.json").write_text(json.dumps({"ixi_background_sigma": noise, "msd_added_sigma": sig}, indent=1))
    meta = {"config": cfg, "git": _git_hash(), "python": platform.python_version(), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "wall_seconds": round(time.time() - t_start, 1), "n_slices": {k: len(v) for k, v in splits.items()}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(region.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {out_dir} in {meta['wall_seconds']} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
