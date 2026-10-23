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

Writes results/<profile>/<out_name>/<dataset>/{per_image.csv.gz, summary.csv, seeds.csv, train_logs.json, meta.json},
out_name = "m3_nets" by default; the M4 configs (diffusion, gan) set "m4_gen".

M4 generative methods use the same protocol. Per-method training settings come from `train_overrides`.
Diffusion is selected on val PSNR of one DDIM sample (`val_steps` steps) and evaluated with `n_samples`
draws of `sample_steps` steps; its mean is scored as "diffusion" and its first draw as "diffusion_sample".
With `dc_eta` > 0 the same network is also sampled with data consistency at every step (MRI, inpainting)
and scored as "diffusion_dc" / "diffusion_dc_sample"; `plain: false` skips the plain variant.
"""
import argparse
import hashlib
import os
import shutil
import zlib
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
def val_psnr(model, x, vd, tasks, dev, bs, tcfg=None):
    model.eval()
    kw = {}
    out = []
    for t, d in vd.items():
        means = []
        for i in range(0, len(x), bs):
            di = to_dev(slice_d(d, slice(i, i + bs)), dev)
            if model.method == "diffusion":
                kw = {"n_samples": 1, "steps": tcfg.get("val_steps", 20),
                      "gen": torch.Generator(device=dev).manual_seed(zlib.crc32(f"val|{t}|{i}".encode()))}
            means.append(model.predict(di, condition(di, tasks[t]), **kw)["mean"].clamp(0, 1).cpu())
        m = torch.cat(means)
        mse = ((m - x) ** 2).flatten(1).mean(1)
        out.append(float((10 * torch.log10(1 / mse.clamp_min(1e-12))).mean()))
    return float(np.mean(out))


def atomic_save(obj, path):
    """torch.save to a temporary file, then rename: a power cut never leaves a half-written file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_block(cache, key):
    """Rows (list of dicts) and extra JSON of a finished block saved by save_block, or None."""
    if cache is None:
        return None
    f = cache / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".json")
    return json.loads(f.read_text()) if f.exists() else None


def save_block(cache, key, **parts):
    """Atomically stores a finished block (any JSON-serialisable parts) so an interrupted run can skip it."""
    if cache is None:
        return
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".json")
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(parts, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    os.replace(tmp, f)


def make_model(method, tcfg):
    return Restorer(method, base=tcfg["base"], p_drop=tcfg.get("p_drop", 0.1), nll_mode=tcfg.get("nll_mode", "plain"),
                    nll_beta=tcfg.get("nll_beta", 0.5), gan_l1=tcfg.get("gan_l1", 100.0))


def method_tcfg(cfg, method):
    return {**cfg["train"], **cfg.get("train_overrides", {}).get(method, {})}


def train_one(method, seed, tasks, data, tcfg, dev, ckpt):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(10_000 + seed)
    model = make_model(method, tcfg).to(dev)
    opt = torch.optim.Adam(model.net.parameters() if method == "gan" else model.parameters(), lr=tcfg["lr"])
    opt_d = (torch.optim.Adam(model.disc.parameters(), lr=tcfg.get("lr_d", tcfg["lr"]), betas=(0.5, 0.999))
             if method == "gan" else None)
    steps, bs = tcfg["steps"], tcfg["batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], total_steps=steps, pct_start=0.05)
    train = torch.from_numpy(data["train"].images)[:, None]
    xv, vd = _val_set(data["val"], tasks, tcfg["n_val"])
    names = list(tasks)
    warm = int(tcfg.get("warmup_frac", 0.1) * steps)
    best, best_state, curve, losses = -np.inf, None, [], []
    t0, start, spent = time.time(), 0, 0.0
    # resumable: a ".partial.pt" next to the checkpoint holds everything needed to continue (written atomically
    # at every validation step); an interrupted run restarts from there instead of from step 0
    partial = ckpt.with_name(ckpt.stem + ".partial.pt")
    if partial.exists():
        r = torch.load(partial, map_location=dev, weights_only=False)
        model.load_state_dict(r["model"]); opt.load_state_dict(r["opt"]); sched.load_state_dict(r["sched"])
        if opt_d is not None:
            opt_d.load_state_dict(r["opt_d"])
        gen.set_state(r["gen"]); torch.set_rng_state(r["torch_rng"])
        if dev == "cuda" and r.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(r["cuda_rng"])
        best, best_state, curve, start, spent = r["best"], r["best_state"], r["curve"], r["step"], r["seconds"]
        print(f"    resumed from step {start} ({partial.name})", flush=True)
    for step in range(start, steps):
        model.train()
        idx = torch.randint(0, len(train), (bs,), generator=gen)
        x = train[idx]
        if torch.rand(1, generator=gen) < 0.5:
            x = x.flip(-1)
        if torch.rand(1, generator=gen) < 0.5:
            x = x.flip(-2)
        task = names[int(torch.randint(0, len(names), (1,), generator=gen))]
        d = to_dev(degrade(x, tasks[task], gen), dev)
        c = condition(d, tasks[task])
        if method == "diffusion":
            loss = model.diffusion_loss(d, c, x.to(dev), gen)
        elif method == "gan":
            fake = model(d, c)
            opt_d.zero_grad(set_to_none=True)
            model.gan_d_loss(d, c, fake, x.to(dev)).backward()
            opt_d.step()
            loss = model.gan_g_loss(d, c, fake, x.to(dev))  # D's grads from this are cleared at its next step
        else:
            loss = model.loss(model(d, c), x.to(dev), step, warm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters() if method == "gan" else model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % tcfg["eval_every"] == 0 or step + 1 == steps:
            p = val_psnr(model, xv, vd, tasks, dev, tcfg.get("eval_batch", 32), tcfg)
            curve.append({"step": step + 1, "val_psnr": p, "train_loss": float(np.mean(losses[-tcfg["eval_every"]:]))})
            if p > best and (method != "hetero" or step >= warm):  # select only after the NLL phase starts
                best, best_state = p, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"    step {step + 1:6d} loss {curve[-1]['train_loss']:.4f} val PSNR {p:.2f} "
                  f"({spent + time.time() - t0:.0f} s)", flush=True)
            if step + 1 < steps:
                atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                             "opt_d": opt_d.state_dict() if opt_d is not None else None,
                             "gen": gen.get_state(), "torch_rng": torch.get_rng_state(),
                             "cuda_rng": torch.cuda.get_rng_state() if dev == "cuda" else None,
                             "best": best, "best_state": best_state, "curve": curve, "step": step + 1,
                             "seconds": spent + time.time() - t0}, partial)
    log = {"method": method, "seed": seed, "best_val_psnr": best, "curve": curve,
           "seconds": round(spent + time.time() - t0, 1), "steps": steps, "resumed_at": start or None}
    atomic_save({"state": best_state, "log": log, "tcfg": tcfg}, ckpt)
    partial.unlink(missing_ok=True)
    return log


def load_model(method, ckpt, tcfg, dev):
    m = make_model(method, tcfg).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev)["state"])
    return m.eval()


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def predict_all(models, d, spec, dev, bs, mc=0, gen_kw=None, seed_key=""):
    """Batched prediction; `models` is one model or a list (ensemble). Returns CPU tensors.
    gen_kw (diffusion): n_samples, steps; each batch gets its own generator seeded from seed_key,
    so samples are reproducible across reruns."""
    outs = []
    for i in range(0, d["inp"].shape[0], bs):
        di = to_dev(slice_d(d, slice(i, i + bs)), dev)
        c = condition(di, spec)
        if gen_kw:
            g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"{seed_key}|{i}".encode()))
            p = models.predict(di, c, gen=g, **gen_kw)
        else:
            if mc:  # MC-dropout: seeded per batch, so results do not depend on what ran before (or on resuming)
                torch.manual_seed(zlib.crc32(f"{seed_key}|mc|{i}".encode()))
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


def evaluate(cfg, data_split, split_name, dtype, models, dseeds, dev, rows, tag, cache=None):
    tasks = type_tasks(cfg["tasks"], dtype)
    x_np = data_split.images
    x = torch.from_numpy(x_np)[:, None]
    bs = cfg["train"].get("eval_batch", 32)
    mc = cfg["train"].get("mc_samples", 16)
    for task, spec in tasks.items():
        for ds in dseeds:
            d = degrade(x, spec, task_generator(task, split_name, ds))
            for name, mseed, m, kw in models:
                key = f"{split_name}|{tag}|{task}|{ds}|{name}|{mseed}|{json.dumps(kw.get('gen_kw'), sort_keys=True)}"
                cfile = cache / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".csv.gz") if cache else None
                if cfile is not None and cfile.exists():  # finished before an interruption: reuse
                    old = pd.read_csv(cfile, dtype={"pid": str, "site": str})
                    rows.extend(old.to_dict("records"))
                    print(f"  [{tag}] {task} d{ds} {name:10s} m{mseed} reused from cache", flush=True)
                    continue
                n_before = len(rows)
                t0 = time.time()
                p = predict_all(m, d, spec, dev, kw.get("bs", bs), mc=kw.get("mc", 0), gen_kw=kw.get("gen_kw"),
                                seed_key=f"{task}|{split_name}|{ds}|{mseed}")
                p["quantile"] = name == "quantile"
                sec = (time.time() - t0) / len(x_np)
                variants = [(name, p["mean"])] + ([(name + "_sample", p["sample"])] if "sample" in p else [])
                for vname, img in variants:
                    xh = img.clamp(0, 1)[:, 0].numpy()
                    s = score(x_np, xh, data_split.region, d)
                    if vname not in ("unet", "unrolled", "gan") and not vname.endswith("_sample"):
                        s.update(uncert_scores(x_np, p, data_split.region))
                    for i in range(len(x_np)):
                        rows.append({"split": split_name, "task": task, "type": dtype, "method": vname, "seed": ds,
                                     "model_seed": mseed, "idx": i, "pid": data_split.pid[i],
                                     "site": data_split.site[i], "sec_per_img": sec,
                                     **{k: float(v[i]) for k, v in s.items()}})
                    print(f"  [{tag}] {task} d{ds} {vname:10s} m{mseed} PSNR {s['psnr'].mean():6.2f} "
                          f"SSIM {s['ssim'].mean():.3f}"
                          + (f" cov90 {np.nanmean(s['cov90']):.3f}" if "cov90" in s else "")
                          + f" ({sec * 1000:.0f} ms/img)", flush=True)
                if cfile is not None:
                    cfile.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cfile.with_name(cfile.name + ".tmp.csv.gz")
                    pd.DataFrame(rows[n_before:]).to_csv(tmp, index=False)
                    os.replace(tmp, cfile)


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
    out_dir = ROOT / "results" / cfg["profile"] / cfg.get("out_name", "m3_nets") / ds_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / ".partial"  # finished evaluation blocks; removed once the run has written its results
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
                    logs.append(train_one(method, seed, tasks, data, method_tcfg(cfg, method), dev, ck) | {"type": dtype})
                trained.setdefault(method, []).append((seed, load_model(method, ck, method_tcfg(cfg, method), dev)))
        models = []
        for method, ms in trained.items():
            for seed, m in ms:
                mt = method_tcfg(cfg, method)
                kw = ({"gen_kw": {"n_samples": mt.get("n_samples", 8), "steps": mt.get("sample_steps", 50)},
                       "bs": mt.get("sample_batch", mt.get("eval_batch", 32))} if method == "diffusion" else {})
                if method != "diffusion" or mt.get("plain", True):
                    models.append((method, seed, m, kw))
                if method == "diffusion" and mt.get("dc_eta") and dtype in ("mri", "inpaint"):
                    # the same network sampled with data consistency at every DDIM step
                    models.append(("diffusion_dc", seed, m, {**kw, "gen_kw": {**kw["gen_kw"], "dc_eta": mt["dc_eta"]}}))
                if method == "unet" and cfg.get("mc_dropout", True):
                    models.append(("mcdropout", seed, m, {"mc": tcfg.get("mc_samples", 16)}))
        if len(trained.get("hetero", [])) >= 2:
            models.append(("ensemble", -1, [m for _, m in trained["hetero"]], {}))
        if cfg.get("eval_split", "test") == "val":
            # pilots: score on validation patients only, so no test number is seen while choosing settings
            evaluate(cfg, data["val"], "val", dtype, models, cfg["eval_dseeds"], dev, rows, "val", cache)
            continue
        evaluate(cfg, data["test"], "test", dtype, models, cfg["eval_dseeds"], dev, rows, "test", cache)
        for tag, od in ood:
            evaluate(cfg, od, "ood", dtype, models, cfg.get("ood_dseeds", [0]), dev, rows, tag, cache)

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
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
