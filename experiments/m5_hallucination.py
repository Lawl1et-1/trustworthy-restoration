"""M5: do restorers erase real lesions or invent new ones, and does their uncertainty notice?

  python -m experiments.m5_hallucination --config configs/m5_full_phantom.yaml

Paired lesion test on phantoms (exact ground truth)
  * Every test phantom with lesions is paired with its lesion-free twin x_bg = x - sum_k c_k * profile_k
    (the lesion list stores centre, radius and contrast; lesions whose pixels were clipped at 1 are flagged).
  * Both versions are degraded with the SAME generator (task_generator(task, "test", dseed)), so they see
    identical noise and masks, and the lesion-present version is exactly the M2-M4 test input.
  * Both are restored by every method; the methods are the M2 classical baselines (weights tuned in M2),
    the M3 networks and the M4 generative restorers, loaded from their checkpoints (nothing is retrained).

Per lesion (one row per lesion x degradation seed x method)
  recovery      mean over the lesion core of (restored - x_bg) / mean over the core of the true lesion
                (contrast x its soft profile); 1 = perfect, < 0.5 = erased
  t_present     mean over the core of (restored present - x_bg); t_absent: the same on the lesion-free twin.
                AUC(t_present vs t_absent) over all lesions = how detectable lesions remain after restoration
                (signal and background known exactly; the degraded input gives the reference AUC).
  core_spread   mean stated uncertainty over the core, relative to the image's median spread in the brain
  core_cov90    fraction of core pixels whose truth lies inside the stated 90 % interval
Per image
  fab_present / era_present   lesion-like blobs in (restored - truth) outside the true lesions
  fab_absent                  lesion-like blobs fabricated anywhere in the restored lesion-free twin
  ause, err_auroc             sparsification error and |error| > 0.05 detection AUROC of the uncertainty map

Writes results/<profile>/m5_halluc/<dataset>/{per_lesion.csv.gz, per_image.csv.gz, summary.csv, meta.json}.
"""
import argparse
import json
import pathlib
import platform
import time
import zlib

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.m3_nets import Z90, load_model, method_tcfg, predict_all, type_tasks
from experiments.run_task import _git_hash
from src.data.datasets import ROOT, load_dataset
from src.degradations.degrade import degrade, task_generator
from src.eval import metrics as M
from src.eval.stats import bootstrap_ci
from src.models.classical import restore
from src.phantoms.phantom import lesion_profile

ERR_TAU = 0.05  # |error| above 5 % of the intensity range counts as a "large" error for err_auroc


# ------------------------------------------------------------------ data

def lesion_free_twins(data):
    """x_bg for every image (lesions subtracted) and, per lesion, whether clipping at 1 touched its core."""
    n = data.images.shape[-1]
    x_bg = data.images.astype(np.float64).copy()
    clipped = []
    for i, les in enumerate(data.lesions):
        cl = []
        for l in les:
            prof = lesion_profile(n, l["cy"], l["cx"], l["r"])
            core = prof > 0.5
            cl.append(bool((data.images[i][core] >= 1.0 - 1e-6).any()))
            x_bg[i] -= l["contrast"] * prof
        clipped.append(cl)
    return np.clip(x_bg, 0, 1).astype(np.float32), clipped


# ------------------------------------------------------------------ methods

def classical_methods(cfg, ds_tag):
    """{type: [(name, weights_by_task)]} from the M2 tuned weights of the same profile and dataset."""
    path = ROOT / "results" / cfg["profile"] / "m2_classical" / ds_tag / "tuned_weights.json"
    w = json.loads(path.read_text())["weights"]
    out = {}
    for dtype, names in cfg.get("classical", {}).items():
        out[dtype] = [(nm, {t: w[t][nm] for t in w if nm in w[t]}) for nm in names]
    return out


def net_methods(cfg, dtype, dev):
    """[(name, model_seed, model or list, kw)] from the M3/M4 checkpoints named by cfg['nets']."""
    out = []
    for src in cfg.get("nets", []):
        scfg = yaml.safe_load((ROOT / src).read_text())
        ckdir = ROOT / "checkpoints" / scfg["profile"] / scfg["data"].get("tag", scfg["data"]["name"])
        hetero = []
        for method, seeds in scfg["methods"].items():
            if method == "unrolled" and dtype != "mri":
                continue
            mt = method_tcfg(scfg, method)
            for seed in seeds[: cfg.get("max_model_seeds", len(seeds))]:
                ck = ckdir / f"{dtype}_{method}_s{seed}.pt"
                if not ck.exists():
                    print(f"  missing {ck.name}, skipped", flush=True)
                    continue
                m = load_model(method, ck, mt, dev)
                kw = {}
                if method == "diffusion":
                    kw = {"gen_kw": {"n_samples": mt.get("n_samples", 8), "steps": mt.get("sample_steps", 25)},
                          "bs": mt.get("sample_batch", 64)}
                if method != "diffusion" or mt.get("plain", True):
                    out.append((method, seed, m, kw))
                if method == "diffusion" and mt.get("dc_eta") and dtype in ("mri", "inpaint"):
                    out.append(("diffusion_dc", seed, m, {**kw, "gen_kw": {**kw["gen_kw"], "dc_eta": mt["dc_eta"]}}))
                if method == "unet":
                    out.append(("mcdropout", seed, m, {"mc": mt.get("mc_samples", 16)}))
                if method == "hetero":
                    hetero.append(m)
        if len(hetero) >= 2:
            out.append(("ensemble", -1, hetero, {}))
    return out


def run(method, d, spec, task, dev, bs, key):
    """Returns {name: (restored (B,H,W) numpy clipped to [0,1], spread (B,H,W) or None, quantile flag)}."""
    name, mseed, m, kw = method
    if m is None:  # classical
        w = kw["weights"][task]
        xh = restore(name, d, w).clamp(0, 1)[:, 0].numpy()
        return {name: (xh, None, None)}
    p = predict_all(m, d, spec, dev, kw.get("bs", bs), mc=kw.get("mc", 0), gen_kw=kw.get("gen_kw"), seed_key=key)
    spread = ((p["lo"] + p["hi"]) / 2)[:, 0].numpy()
    lo, hi = p["lo"][:, 0].numpy(), p["hi"][:, 0].numpy()
    has_u = name not in ("unet", "unrolled", "gan")
    out = {name: (p["mean"].clamp(0, 1)[:, 0].numpy(), (spread, lo, hi) if has_u else None, name == "quantile")}
    if "sample" in p:
        out[name + "_sample"] = (p["sample"].clamp(0, 1)[:, 0].numpy(), None, None)
    return out


# ------------------------------------------------------------------ scoring

def lesion_rows(x, x_bg, xp, xa, unc, quant, lesions, clipped, region):
    n = x.shape[-1]
    rows = []
    for b, les in enumerate(lesions):
        med = float(np.median(unc[0][b][region[b]])) if unc is not None else np.nan
        for k, l in enumerate(les):
            prof = lesion_profile(n, l["cy"], l["cx"], l["r"])
            core = prof > 0.5
            height = l["contrast"] * float(prof[core].mean())  # true mean lesion height over the core
            tp = float((xp[b] - x_bg[b])[core].mean())
            ta = float((xa[b] - x_bg[b])[core].mean())
            r = {"lesion": k, "img": b, "radius": l["r"], "contrast": l["contrast"], "clipped": clipped[b][k],
                 "recovery": tp / height, "t_present": tp, "t_absent": ta,
                 "core_spread": np.nan, "core_cov90": np.nan}
            if unc is not None:
                spread, lo, hi = unc
                lam = 1.0 if quant else Z90
                e = x[b][core] - xp[b][core]
                r["core_spread"] = float(spread[b][core].mean() / max(med, 1e-12))
                r["core_cov90"] = float(((e >= -lam * lo[b][core]) & (e <= lam * hi[b][core])).mean())
            rows.append(r)
    return rows


def image_scores(x, x_bg, xp, xa, unc, labels, region):
    fab_p, era_p = M.spurious_blobs(x, xp, region, labels)
    fab_a, _ = M.spurious_blobs(x_bg, xa, region, np.zeros_like(labels))
    s = {"fab_present": fab_p, "era_present": era_p, "fab_absent": fab_a,
         "ause": np.full(len(x), np.nan), "err_auroc": np.full(len(x), np.nan)}
    if unc is not None:
        err = x - xp
        _, _, s["ause"] = M.sparsification(err, unc[0], region)
        s["err_auroc"] = M.error_auroc(err, unc[0], region, ERR_TAU)
    return s


def summarise(les, img):
    rows = []
    usable = les[~les.clipped]
    for (task, method), g in usable.groupby(["task", "method"], sort=False):
        gi = img[(img.task == task) & (img.method == method)]
        auc = M.detect_auc(g.t_present.values, g.t_absent.values)
        rec_ci = bootstrap_ci(g.recovery.values, g.pid.values)
        r = {"task": task, "method": method, "n_lesions": len(g), "n_patients": g.pid.nunique(),
             "recovery": rec_ci[0], "recovery_lo": rec_ci[1], "recovery_hi": rec_ci[2],
             "erased_frac": float((g.recovery < 0.5).mean()), "detect_auc": auc,
             "fab_present": gi.fab_present.mean(), "era_present": gi.era_present.mean(),
             "fab_absent": gi.fab_absent.mean(), "ause": gi.ause.mean(), "err_auroc": gi.err_auroc.mean(),
             "core_spread": g.core_spread.mean(), "core_cov90": g.core_cov90.mean()}
        # does the stated uncertainty single out the lesions the method erased?
        er = g.recovery < 0.5
        if g.core_spread.notna().all() and 0 < er.sum() < len(g):
            from sklearn.metrics import roc_auc_score
            r["erased_flag_auroc"] = roc_auc_score(er, g.core_spread)
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main

def main(cfg_path):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    dev = "cuda" if torch.cuda.is_available() and not cfg.get("cpu") else "cpu"
    print("device:", dev, flush=True)
    t_start = time.time()
    data = load_dataset(cfg["data"])
    ds_tag = cfg["data"].get("tag", cfg["data"]["name"])
    test = data["test"]
    keep = np.array([len(l) > 0 for l in test.lesions])
    test = test.take(np.nonzero(keep)[0])
    x = test.images
    x_bg, clipped = lesion_free_twins(test)
    print(f"{len(x)} test phantoms with lesions, {sum(map(len, test.lesions))} lesions, "
          f"{sum(map(sum, clipped))} with clipped cores (excluded from summaries)", flush=True)
    out_dir = ROOT / "results" / cfg["profile"] / cfg.get("out_name", "m5_halluc") / ds_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    classical = classical_methods(cfg, ds_tag)
    bs = cfg.get("eval_batch", 32)
    xt, xbt = torch.from_numpy(x)[:, None], torch.from_numpy(x_bg)[:, None]

    les_rows, img_rows = [], []
    for dtype in sorted({s["type"] for s in cfg["tasks"].values()}):
        methods = [(nm, -1, None, {"weights": w}) for nm, w in classical.get(dtype, [])] + net_methods(cfg, dtype, dev)
        for task, spec in type_tasks(cfg["tasks"], dtype).items():
            for ds in cfg["eval_dseeds"]:
                dp = degrade(xt, spec, task_generator(task, "test", ds))
                da = degrade(xbt, spec, task_generator(task, "test", ds))
                for meth in methods:
                    t0 = time.time()
                    key = f"m5|{task}|{ds}|{meth[1]}"
                    rp = run(meth, dp, spec, task, dev, bs, key + "|p")
                    ra = run(meth, da, spec, task, dev, bs, key + "|a")
                    for name in rp:
                        xp, unc, quant = rp[name]
                        xa = ra[name][0]
                        for r in lesion_rows(x, x_bg, xp, xa, unc, quant, test.lesions, clipped, test.region):
                            les_rows.append({"task": task, "method": name, "model_seed": meth[1], "seed": ds,
                                             "pid": test.pid[r["img"]], **r})
                        s = image_scores(x, x_bg, xp, xa, unc, test.labels, test.region)
                        for i in range(len(x)):
                            img_rows.append({"task": task, "method": name, "model_seed": meth[1], "seed": ds,
                                             "idx": i, "pid": test.pid[i], **{k: float(v[i]) for k, v in s.items()}})
                        g = pd.DataFrame([r for r in les_rows if r["task"] == task and r["method"] == name and r["seed"] == ds
                                          and r["model_seed"] == meth[1]])
                        print(f"  {task} d{ds} {name:16s} m{meth[1]:2d} recovery {g.recovery.mean():5.2f} "
                              f"erased {(g.recovery < 0.5).mean():4.2f} fab_absent {s['fab_absent'].mean():5.2f} "
                              f"({time.time() - t0:.0f} s)", flush=True)
        # the degraded input itself is the reference for detectability
        for task, spec in type_tasks(cfg["tasks"], dtype).items():
            for ds in cfg["eval_dseeds"]:
                dp = degrade(xt, spec, task_generator(task, "test", ds))
                da = degrade(xbt, spec, task_generator(task, "test", ds))
                xp, xa = dp["inp"].clamp(0, 1)[:, 0].numpy(), da["inp"].clamp(0, 1)[:, 0].numpy()
                for r in lesion_rows(x, x_bg, xp, xa, None, None, test.lesions, clipped, test.region):
                    les_rows.append({"task": task, "method": "input", "model_seed": -1, "seed": ds,
                                     "pid": test.pid[r["img"]], **r})
                s = image_scores(x, x_bg, xp, xa, None, test.labels, test.region)
                for i in range(len(x)):
                    img_rows.append({"task": task, "method": "input", "model_seed": -1, "seed": ds, "idx": i,
                                     "pid": test.pid[i], **{k: float(v[i]) for k, v in s.items()}})

    les, img = pd.DataFrame(les_rows), pd.DataFrame(img_rows)
    les.to_csv(out_dir / "per_lesion.csv.gz", index=False)
    img.to_csv(out_dir / "per_image.csv.gz", index=False)
    summ = summarise(les, img)
    summ.to_csv(out_dir / "summary.csv", index=False)
    meta = {"config": cfg, "git": _git_hash(), "python": platform.python_version(), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "wall_seconds": round(time.time() - t_start, 1), "n_images": int(len(x)),
            "n_lesions": int(sum(map(len, test.lesions))), "n_clipped": int(sum(map(sum, clipped)))}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    cols = ["task", "method", "recovery", "erased_frac", "detect_auc", "fab_absent", "core_cov90"]
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {out_dir} in {meta['wall_seconds']} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
