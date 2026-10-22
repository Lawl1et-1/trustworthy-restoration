"""M3: train and evaluate the learned restorers (src/models/restorers.py).

  python -m experiments.m3_nets --config configs/m3_smoke.yaml

Protocol
  * One network per (degradation type, method, model seed), trained on the train split with fresh
    degradations every step: a random severity of that type per batch, random flips.
  * Model selection: the checkpoint with the best mean PSNR on a fixed degraded val subset (every
    severity, degradation seed 0, val stream). Test, cal and OOD data are never used for training or selection.
  * Test: the same degradations as M2 (task_generator(task, "test", dseed)), so learned and classical
    methods are compared on identical inputs. Outputs clipped to [0, 1] before scoring, as in M2.
  * Checkpoints go to checkpoints/<profile>/<dataset>/; an existing checkpoint is reused, so an
    interrupted run resumes where it stopped.

Writes results/<profile>/m3_nets/<dataset>/{per_image.csv.gz, summary.csv, seeds.csv, train_logs.json, meta.json}.
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

from experiments.run_task import FIDELITY, PERCEPTION, LINEAR, _git_hash, score, summarise
from src.data.datasets import ROOT, load_dataset
from src.degradations.degrade import degrade, task_generator
from src.degradations.operators import LinearOp
from src.models.restorers import Restorer, condition, ensemble

UNCERT = ["cov90", "mean_spread", "spread_err_spearman"]
Z90 = 1.6448536269514722  # two-sided 90 % Gaussian interval


def to_dev(d, dev):
    out = dict(d)
    for k in ("inp", "ind", "y"):
        if d[k] is not None:
            out[k] = d[k].to(dev)
    if d["op"] is not None:
        out["op"] = LinearOp(d["op"].kind, d["op"].mask.to(dev))
    return out


def slice_d(d, sl):
    out = dict(d)
    for k in ("inp", "ind", "y"):
        if d[k] is not None:
            out[k] = d[k][sl]
    if d["op"] is not None:
        out["op"] = LinearOp(d["op"].kind, d["op"].mask[sl])
    return out


def type_tasks(tasks, dtype):
    return {k: v for k, v in tasks.items() if v["type"] == dtype}


# ------------------------------------------------------------------ training

def _val_set(val, tasks, n_val):
    v = val.head(n_val)
    x = torch.from_numpy(v.images)[:, None]
    return x, {t: degrade(x, s, task_generator(t, "val", 0)) for t, s in tasks.items()}


@torch.no_grad()
def val_psnr(model, x, vd, tasks, dev, bs):
    model.eval()
    out = []
    for t, d in vd.items():
        means = []
        for i in range(0, len(x), bs):
            di = to_dev(slice_d(d, slice(i, i + bs)), dev)
            means.append(model.predict(di, condition(di, tasks[t]))["mean"].clamp(0, 1).cpu())
        m = torch.cat(means)
        mse = ((m - x) ** 2).flatten(1).mean(1)
        out.append(float((10 * torch.log10(1 / mse.clamp_min(1e-12))).mean()))
    return float(np.mean(out))


def train_one(method, seed, tasks, data, tcfg, dev, ckpt):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(10_000 + seed)
    model = Restorer(method, base=tcfg["base"], p_drop=tcfg.get("p_drop", 0.1), nll_mode=tcfg.get("nll_mode", "plain"), nll_beta=tcfg.get("nll_beta", 0.5)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    steps, bs = tcfg["steps"], tcfg["batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], total_steps=steps, pct_start=0.05)
    train = torch.from_numpy(data["train"].images)[:, None]
    xv, vd = _val_set(data["val"], tasks, tcfg["n_val"])
    names = list(tasks)
    warm = int(tcfg.get("warmup_frac", 0.1) * steps)
    best, best_state, curve, losses = -np.inf, None, [], []
    t0 = time.time()
    for step in range(steps):
        model.train()
        idx = torch.randint(0, len(train), (bs,), generator=gen)
        x = train[idx]
        if torch.rand(1, generator=gen) < 0.5:
            x = x.flip(-1)
        if torch.rand(1, generator=gen) < 0.5:
            x = x.flip(-2)
        task = names[int(torch.randint(0, len(names), (1,), generator=gen))]
        d = to_dev(degrade(x, tasks[task], gen), dev)
        out = model(d, condition(d, tasks[task]))
        loss = model.loss(out, x.to(dev), step, warm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % tcfg["eval_every"] == 0 or step + 1 == steps:
            p = val_psnr(model, xv, vd, tasks, dev, tcfg.get("eval_batch", 32))
            curve.append({"step": step + 1, "val_psnr": p, "train_loss": float(np.mean(losses[-tcfg["eval_every"]:]))})
            if p > best and (method != "hetero" or step >= warm):  # select only after the NLL phase starts
                best, best_state = p, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"    step {step + 1:6d} loss {curve[-1]['train_loss']:.4f} val PSNR {p:.2f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    log = {"method": method, "seed": seed, "best_val_psnr": best, "curve": curve,
           "seconds": round(time.time() - t0, 1), "steps": steps}
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "log": log, "tcfg": tcfg}, ckpt)
    return log


def load_model(method, ckpt, tcfg, dev):
    m = Restorer(method, base=tcfg["base"], p_drop=tcfg.get("p_drop", 0.1), nll_mode=tcfg.get("nll_mode", "plain"), nll_beta=tcfg.get("nll_beta", 0.5)).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev)["state"])
    return m.eval()


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def predict_all(models, d, spec, dev, bs, mc=0):
    """Batched prediction; `models` is one model or a list (ensemble). Returns CPU tensors."""
    outs = []
    for i in range(0, d["inp"].shape[0], bs):
        di = to_dev(slice_d(d, slice(i, i + bs)), dev)
        c = condition(di, spec)
        p = ensemble([m.predict(di, c) for m in models]) if isinstance(models, list) else models.predict(di, c, mc)
        outs.append({k: v.cpu() for k, v in p.items()})
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def uncert_scores(x, p, region):
    """Sanity-level uncertainty numbers (the full suite is M5): 90 % interval coverage in the region,
    mean spread, and the per-image Spearman correlation between spread and |error|."""
    from scipy.stats import spearmanr
    mean, lo, hi = p["mean"][:, 0].numpy(), p["lo"][:, 0].numpy(), p["hi"][:, 0].numpy()
    lam = 1.0 if p.get("quantile") else Z90
    err = x - mean
    cov = np.array([(((err[b] >= -lam * lo[b]) & (err[b] <= lam * hi[b]))[region[b]]).mean() for b in range(len(x))])
    spread = (lo + hi) / 2
    msp = np.array([spread[b][region[b]].mean() for b in range(len(x))])
    rho = np.array([spearmanr(spread[b][region[b]], np.abs(err[b][region[b]]))[0] if spread[b][region[b]].std() > 0
                    else np.nan for b in range(len(x))])
    return {"cov90": cov, "mean_spread": msp, "spread_err_spearman": rho}


def evaluate(cfg, data_split, split_name, dtype, models, dseeds, dev, rows, tag):
    tasks = type_tasks(cfg["tasks"], dtype)
    x_np = data_split.images
    x = torch.from_numpy(x_np)[:, None]
    bs = cfg["train"].get("eval_batch", 32)
    mc = cfg["train"].get("mc_samples", 16)
    for task, spec in tasks.items():
        for ds in dseeds:
            d = degrade(x, spec, task_generator(task, split_name, ds))
            for name, mseed, m, kw in models:
                t0 = time.time()
                p = predict_all(m, d, spec, dev, bs, mc=kw.get("mc", 0))
                p["quantile"] = name == "quantile"
                sec = (time.time() - t0) / len(x_np)
                xh = p["mean"].clamp(0, 1)[:, 0].numpy()
                s = score(x_np, xh, data_split.region, d)
                if name != "unet" and name != "unrolled":
                    s.update(uncert_scores(x_np, p, data_split.region))
                for i in range(len(x_np)):
                    rows.append({"split": split_name, "task": task, "type": dtype, "method": name, "seed": ds,
                                 "model_seed": mseed, "idx": i, "pid": data_split.pid[i], "site": data_split.site[i],
                                 "sec_per_img": sec, **{k: float(v[i]) for k, v in s.items()}})
                print(f"  [{tag}] {task} d{ds} {name:10s} m{mseed} PSNR {s['psnr'].mean():6.2f} "
                      f"SSIM {s['ssim'].mean():.3f}" + (f" cov90 {np.nanmean(s['cov90']):.3f}" if "cov90" in s else ""),
                      flush=True)


def summarise_nets(df):
    """Patient-bootstrap summary (averaging each image over model and degradation seeds), plus a
    per-model-seed table: patient bootstraps alone hide the variation between training runs."""
    summ = summarise(df.rename(columns={"model_seed": "_ms"}).assign(weight=np.nan))
    extra = [c for c in UNCERT if c in df.columns]
    for c in extra:
        g = df.groupby(["split", "task", "method"], sort=False)[c].mean().rename(c)
        summ = summ.merge(g, on=["split", "task", "method"], how="left")
    seeds = (df.groupby(["split", "task", "method", "model_seed"], sort=False)[["psnr", "ssim"] + extra].mean()
             .reset_index())
    sd = seeds.groupby(["split", "task", "method"], sort=False)["psnr"].agg(["std", "count"]).reset_index()
    summ = summ.merge(sd.rename(columns={"std": "psnr_sd_over_model_seeds", "count": "n_model_seeds"}),
                      on=["split", "task", "method"], how="left")
    return summ, seeds


def main(cfg_path):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() and not cfg.get("cpu") else "cpu"
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
    t_start = time.time()
    data = load_dataset(cfg["data"])
    ds_tag = cfg["data"].get("tag", cfg["data"]["name"])
    ckdir = ROOT / "checkpoints" / cfg["profile"] / ds_tag
    out_dir = ROOT / "results" / cfg["profile"] / "m3_nets" / ds_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tcfg = cfg["train"]
    types = sorted({s["type"] for s in cfg["tasks"].values()})
    ood = []
    for o in cfg.get("ood", []):
        try:
            ood.append((o.get("tag", o["name"]), load_dataset(o)["test"]))
        except FileNotFoundError as e:
            print("skip OOD:", e, flush=True)

    logs, rows = [], []
    for dtype in types:
        tasks = type_tasks(cfg["tasks"], dtype)
        trained = {}
        for method, seeds in cfg["methods"].items():
            if method == "unrolled" and dtype != "mri":
                continue
            for seed in seeds:
                ck = ckdir / f"{dtype}_{method}_s{seed}.pt"
                if ck.exists():
                    print(f"[{dtype}] {method} seed {seed}: reuse {ck.name}", flush=True)
                    logs.append(torch.load(ck, map_location="cpu")["log"] | {"type": dtype, "reused": True})
                else:
                    print(f"[{dtype}] train {method} seed {seed}", flush=True)
                    logs.append(train_one(method, seed, tasks, data, tcfg, dev, ck) | {"type": dtype})
                trained.setdefault(method, []).append((seed, load_model(method, ck, tcfg, dev)))
        models = []
        for method, ms in trained.items():
            for seed, m in ms:
                models.append((method, seed, m, {}))
                if method == "unet" and cfg.get("mc_dropout", True):
                    models.append(("mcdropout", seed, m, {"mc": tcfg.get("mc_samples", 16)}))
        if len(trained.get("hetero", [])) >= 2:
            models.append(("ensemble", -1, [m for _, m in trained["hetero"]], {}))
        evaluate(cfg, data["test"], "test", dtype, models, cfg["eval_dseeds"], dev, rows, "test")
        for tag, od in ood:
            evaluate(cfg, od, "ood", dtype, models, cfg.get("ood_dseeds", [0]), dev, rows, tag)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_image.csv.gz", index=False)
    summ, seeds = summarise_nets(df)
    summ.to_csv(out_dir / "summary.csv", index=False)
    seeds.to_csv(out_dir / "seeds.csv", index=False)
    (out_dir / "train_logs.json").write_text(json.dumps(logs, indent=1))
    meta = {"config": cfg, "git": _git_hash(), "python": platform.python_version(), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "wall_seconds": round(time.time() - t_start, 1), "n_slices": {s: len(d) for s, d in data.items()}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    cols = ["split", "task", "method", "psnr", "psnr_lo", "psnr_hi", "psnr_sd_over_model_seeds"] + \
           [c for c in ("cov90",) if c in summ.columns]
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {out_dir} in {meta['wall_seconds']} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
