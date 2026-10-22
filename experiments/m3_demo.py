"""Demo assets for the project page (docs/): real test images restored by every method, with error,
uncertainty and interval-miss maps.

  python -m experiments.m3_demo --config configs/m3_full_phantom.yaml
  python -m experiments.m3_demo --config configs/m3_full_msd.yaml

Image choice is a fixed rule decided before looking at any output, never a pick of good-looking cases:
  phantom   the first 9 test phantoms (sorted id) that contain at least one lesion
  msd       the first 9 test patients (sorted id), each at its slice with the largest tumour area
  ixi       the first 3 subjects of each site (sorted id), each at its middle slice
Every image is degraded exactly as in the evaluation (degradation seed 0, same stream), and the PSNR of each
exported output is checked against the evaluation's per_image.csv.gz, M3 and (for the classical reference)
M2 (CPU vs GPU, so a
few hundredths of a dB of float noise are tolerated; MC-dropout is stochastic and is reported, not checked).

Writes docs/demo/<dataset>/<split>_<image>_<task>.png (one strip of 128 px grey tiles, order in the JSON),
docs/demo/<dataset>/<split>_thumbs.png (the clean images of the gallery, in manifest order) and
docs/demo/<dataset>/manifest.json.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from experiments.m3_nets import Z90, load_model, slice_d, type_tasks
from experiments.run_task import ROOT
from src.data.datasets import load_dataset
from src.degradations.degrade import degrade, task_generator
from src.eval import metrics as M
from src.models.classical import restore
from src.models.restorers import condition, ensemble

DEMO_TASKS = ["denoise_s0.10", "mri_R4", "mri_R8", "inpaint_mixed"]
REF = {"denoise": "bm3d", "mri": "tv", "inpaint": "biharmonic"}
EMAX = 0.15          # error and half-width maps share one fixed scale: 0 -> black, >= EMAX -> white
TOL_DB = 0.05
N_PICK = 9


def pick(split_name, data, ds, n=N_PICK):
    pids = np.unique(data.pid)
    out = []
    if ds == "phantom":
        for p in pids:
            i = int(np.nonzero(data.pid == p)[0][0])
            if data.labels[i].any():
                out.append(i)
            if len(out) == n:
                break
    elif split_name == "ood":
        sites = sorted(set(data.site))
        for site in sites:
            for p in sorted(set(data.pid[data.site == site]))[:n // len(sites)]:
                idx = np.nonzero(data.pid == p)[0]
                out.append(int(idx[len(idx) // 2]))
    else:
        for p in pids[:n]:
            idx = np.nonzero(data.pid == p)[0]
            out.append(int(idx[np.argmax(data.labels[idx].reshape(len(idx), -1).sum(1))]))
    return out


def u8(a, scale=1.0):
    return (np.clip(np.asarray(a, np.float64) / scale, 0, 1) * 255 + 0.5).astype(np.uint8)


@torch.no_grad()
def run(cfg_path, out_root):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    torch.manual_seed(0)
    data = load_dataset(cfg["data"])
    ds = cfg["data"].get("tag", cfg["data"]["name"])
    tcfg = cfg["train"]
    ckdir = ROOT / "checkpoints" / cfg["profile"] / ds
    weights = json.loads((ROOT / "results" / cfg["profile"] / "m2_classical" / ds / "tuned_weights.json").read_text())["weights"]
    m3img = pd.read_csv(ROOT / "results" / cfg["profile"] / "m3_nets" / ds / "per_image.csv.gz")
    m2img = pd.read_csv(ROOT / "results" / cfg["profile"] / "m2_classical" / ds / "per_image.csv.gz")
    out = pathlib.Path(out_root) / ds
    out.mkdir(parents=True, exist_ok=True)
    splits = [("test", data["test"])]
    for o in cfg.get("ood", []):
        try:
            splits.append(("ood", load_dataset(o)["test"]))
        except FileNotFoundError as e:
            print("skip OOD:", e)
    manifest = {"dataset": ds, "emax": EMAX, "tile": int(data["test"].images.shape[-1]), "rule": __doc__.split("Image choice")[1].split("Every image")[0].strip(),
                "items": []}
    problems = []
    for task in [t for t in DEMO_TASKS if t in cfg["tasks"]] or list(cfg["tasks"]):
        spec = cfg["tasks"][task]
        ttype = spec["type"]
        models = {}
        for method, seeds in cfg["methods"].items():
            if method == "unrolled" and ttype != "mri":
                continue
            use = seeds if method == "hetero" else seeds[:1]  # the ensemble needs every hetero seed; others show seed 0
            models[method] = [load_model(method, ckdir / f"{ttype}_{method}_s{s}.pt", tcfg, "cpu") for s in use]
        for split_name, sd in splits:
            sel = pick(split_name, sd, ds)
            x_all = torch.from_numpy(sd.images)[:, None]
            d_all = degrade(x_all, spec, task_generator(task, split_name, 0))  # the evaluation's own degradation
            for i in sel:
                d = slice_d(d_all, slice(i, i + 1))
                x = sd.images[i]
                reg = sd.region[i]
                c = condition(d, spec)
                preds = {"unet": models["unet"][0].predict(d, c),
                         "mcdropout": models["unet"][0].predict(d, c, tcfg.get("mc_samples", 16)),
                         "hetero": models["hetero"][0].predict(d, c),
                         "ensemble": ensemble([m.predict(d, c) for m in models["hetero"]]),
                         "quantile": models["quantile"][0].predict(d, c)}
                if "unrolled" in models:
                    preds["unrolled"] = models["unrolled"][0].predict(d, c)
                ref = REF[ttype]
                ref_img = restore(ref, d, weights[task][ref]).clamp(0, 1)[0, 0].numpy()
                inp = d["inp"][0, 0].clamp(0, 1).numpy()
                tiles = [("clean", u8(x)), ("input", u8(inp)), (ref, u8(ref_img)), (f"{ref}_err", u8(np.abs(x - ref_img), EMAX))]
                stats = {"input": {"psnr": float(M.psnr(x[None], inp[None])[0])},
                         ref: {"psnr": float(M.psnr(x[None], ref_img[None])[0])}}
                r2 = m2img[(m2img.split == split_name) & (m2img.task == task) & (m2img.method == ref) &
                           (m2img.seed == 0) & (m2img.idx == i)]
                if len(r2) != 1 or abs(r2.psnr.iloc[0] - stats[ref]["psnr"]) > TOL_DB:
                    problems.append(f"{split_name}/{task}/{ref}/idx{i}: exported {stats[ref]['psnr']:.3f} dB, "
                                    f"M2 evaluated {r2.psnr.iloc[0] if len(r2) else 'missing'}")
                for name, p in preds.items():
                    mean = p["mean"].clamp(0, 1)[0, 0].numpy()
                    err = x - p["mean"][0, 0].numpy()
                    tiles += [(name, u8(mean)), (f"{name}_err", u8(np.abs(x - mean), EMAX))]
                    s = {"psnr": float(M.psnr(x[None], mean[None])[0])}
                    if name not in ("unet", "unrolled"):
                        lam = 1.0 if name == "quantile" else Z90
                        lo, hi = lam * p["lo"][0, 0].numpy(), lam * p["hi"][0, 0].numpy()
                        miss = ~((err >= -lo) & (err <= hi)) & reg
                        tiles += [(f"{name}_halfwidth", u8((lo + hi) / 2, EMAX)), (f"{name}_miss", u8(miss))]
                        s["cov90"] = float(1 - miss[reg].mean())
                    stats[name] = s
                    ms = 0 if name not in ("ensemble",) else -1
                    if name != "mcdropout":
                        r = m3img[(m3img.split == split_name) & (m3img.task == task) & (m3img.method == name) &
                                   (m3img.seed == 0) & (m3img.model_seed == ms) & (m3img.idx == i)]
                        if len(r) != 1:
                            problems.append(f"{split_name}/{task}/{name}/idx{i}: {len(r)} rows in per_image.csv.gz")
                        elif abs(r.psnr.iloc[0] - s["psnr"]) > TOL_DB:
                            problems.append(f"{split_name}/{task}/{name}/idx{i}: exported {s['psnr']:.3f} dB, "
                                            f"evaluated {r.psnr.iloc[0]:.3f} dB")
                tiles.append(("region", u8(reg)))
                if sd.labels[i].any():
                    tiles.append(("lesion", u8(sd.labels[i])))
                fname = f"{split_name}_{i}_{task}.png"
                Image.fromarray(np.concatenate([t for _, t in tiles], 1)).save(out / fname, optimize=True)
                manifest["items"].append({"split": split_name, "idx": i, "pid": str(sd.pid[i]), "site": str(sd.site[i]),
                                          "task": task, "type": ttype, "file": fname, "tiles": [n for n, _ in tiles],
                                          "reference": ref, "stats": stats})
                print(f"  {split_name} {task} idx {i} ({sd.pid[i]}): " +
                      " ".join(f"{k} {v['psnr']:.2f}" for k, v in stats.items()), flush=True)
    manifest["thumbs"] = {}
    for split_name, sd in splits:
        sel = pick(split_name, sd, ds)
        Image.fromarray(np.concatenate([u8(sd.images[i]) for i in sel], 1)).save(out / f"{split_name}_thumbs.png", optimize=True)
        manifest["thumbs"][split_name] = {"file": f"{split_name}_thumbs.png", "idx": sel,
                                          "pid": [str(sd.pid[i]) for i in sel], "site": [str(sd.site[i]) for i in sel]}
    manifest["checks"] = problems
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for p in problems:
        print("CHECK FAILED:", p)
    print(f"wrote {len(manifest['items'])} strips to {out}")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo"))
    a = ap.parse_args()
    raise SystemExit(1 if run(a.config, a.out) else 0)
