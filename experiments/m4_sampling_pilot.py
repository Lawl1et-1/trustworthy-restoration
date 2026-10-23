"""M4 pilot: how many DDIM steps and samples does the diffusion restorer need? Validation patients only.

  python -m experiments.m4_sampling_pilot --ckpt checkpoints/pilot_m4/phantom/denoise_diffusion_s0.pt

Scores the sample mean and a single draw for a grid of (steps, n_samples) on the first n validation
images (degradation seed 1), and reports seconds per image, so the full run can pick a setting.
"""
import argparse
import time
import zlib

import numpy as np
import torch
import yaml

from experiments.m3_nets import load_model, method_tcfg, slice_d, to_dev, uncert_scores
from src.data.datasets import load_dataset
from src.degradations.degrade import degrade, task_generator
from src.eval.metrics import psnr
from src.models.restorers import condition


def main(a):
    cfg = yaml.safe_load(open(a.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_model("diffusion", a.ckpt, method_tcfg(cfg, "diffusion"), dev)
    val = load_dataset(cfg["data"])["val"].head(a.n)
    x = torch.from_numpy(val.images)[:, None]
    tasks = {t: s for t, s in cfg["tasks"].items() if s["type"] == a.type}
    for task, spec in tasks.items():
        d = degrade(x, spec, task_generator(task, "val", 1))
        for steps in a.steps:
          for eta in a.dc:
            for k in a.samples:
                outs = []
                t0 = time.time()
                for i in range(0, len(x), a.bs):
                    di = to_dev(slice_d(d, slice(i, i + a.bs)), dev)
                    g = torch.Generator(device=dev).manual_seed(zlib.crc32(f"pilot|{task}|{i}".encode()))
                    p = m.predict(di, condition(di, spec), n_samples=k, steps=steps, gen=g, dc_eta=eta)
                    outs.append({kk: v.cpu() for kk, v in p.items()})
                if dev == "cuda":
                    torch.cuda.synchronize()
                sec = (time.time() - t0) / len(x)
                p = {kk: torch.cat([o[kk] for o in outs]) for kk in outs[0]}
                mean = psnr(val.images, p["mean"].clamp(0, 1)[:, 0].numpy()).mean()
                one = psnr(val.images, p["sample"].clamp(0, 1)[:, 0].numpy()).mean()
                cov = np.nanmean(uncert_scores(val.images, p, val.region)["cov90"]) if k > 1 else float("nan")
                print(f"{task} steps {steps:3d} dc {eta:.2f} K {k:2d}: mean PSNR {mean:6.2f}  one-sample {one:6.2f}  "
                      f"cov90 {cov:.3f}  {sec * 1000:6.1f} ms/img", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m4_pilot.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--type", default="denoise")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 25, 50])
    ap.add_argument("--samples", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--dc", type=float, nargs="+", default=[0.0], help="data-consistency strengths (MRI, inpainting)")
    main(ap.parse_args())
