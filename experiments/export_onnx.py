"""Export trained restorers to ONNX for the project page's "try your own image" demo (runs in the browser).

  python -m experiments.export_onnx --config configs/m3_full_msd.yaml

Exports, per degradation type, the heteroscedastic network of model seed 0 (mean + uncertainty), and for MRI
also the unrolled network of seed 0. The FFT inside the unrolled network is written as multiplications with the
128-point DFT matrix, which ONNX supports everywhere. Each export is checked against the PyTorch model on test
images with real degradations; the export fails if the outputs differ by more than 1e-4.

Inputs (float32, 1 x 1 x H x W): inp (A^+ y or the noisy image), cond (the condition channel of restorers.py);
unrolled also takes mask, y_re, y_im (unshifted k-space). Outputs: mean, sd (heteroscedastic) or mean (unrolled).
Writes docs/models/<dataset>/*.onnx and docs/models/<dataset>/models.json.
"""
import argparse
import json
import math
import pathlib

import numpy as np
import torch
import yaml

from experiments.m3_nets import load_model, slice_d
from src.data.datasets import ROOT, load_dataset
from src.degradations.degrade import degrade, task_generator
from src.models.restorers import condition

TOL = 1e-4


class HeteroWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, inp, cond):
        out = self.m({"inp": inp}, cond)
        return out[:, :1], torch.exp(0.5 * out[:, 1:])


class UnrolledWrap(torch.nn.Module):
    """Unrolled.forward with fft2 / ifft2 (norm="ortho") written as DFT-matrix products."""

    def __init__(self, m, n):
        super().__init__()
        self.net = m.net
        k = torch.arange(n, dtype=torch.float64)
        ang = -2 * math.pi * torch.outer(k, k) / n
        self.register_buffer("Fr", (torch.cos(ang) / math.sqrt(n)).float())
        self.register_buffer("Fi", (torch.sin(ang) / math.sqrt(n)).float())

    def fft2(self, xr, xi):  # F X F, F symmetric
        ar, ai = self.Fr @ xr - self.Fi @ xi, self.Fr @ xi + self.Fi @ xr
        return ar @ self.Fr - ai @ self.Fi, ar @ self.Fi + ai @ self.Fr

    def ifft2_real(self, yr, yi):  # Re(conj(F) Y conj(F))
        ar, ai = self.Fr @ yr + self.Fi @ yi, self.Fr @ yi - self.Fi @ yr
        return ar @ self.Fr + ai @ self.Fi

    def forward(self, inp, cond, mask, y_re, y_im):
        aty = self.ifft2_real(y_re * mask, y_im * mask)
        x = inp
        for cnn, eta in zip(self.net.cnns, self.net.eta):
            x = x + cnn(torch.cat([x, cond], 1))
            kr, ki = self.fft2(x, torch.zeros_like(x))
            x = x + torch.sigmoid(eta) * (aty - self.ifft2_real(kr * mask, ki * mask))
        return x


def export(wrap, args, names, outs, path):
    torch.onnx.export(wrap, args, str(path), input_names=names, output_names=outs, opset_version=17, dynamo=False)


def check(path, wrap, feeds, ref):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {k: v.numpy() for k, v in feeds.items()})
    err = max(float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref))
    return err


@torch.no_grad()
def main(cfg_path, out_root):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    ds = cfg["data"].get("tag", cfg["data"]["name"])
    tcfg = cfg["train"]
    ckdir = ROOT / "checkpoints" / cfg["profile"] / ds
    out = pathlib.Path(out_root) / ds
    out.mkdir(parents=True, exist_ok=True)
    test = load_dataset(cfg["data"])["test"]
    x_all = torch.from_numpy(test.images[:8])[:, None]
    n = x_all.shape[-1]
    manifest = {"dataset": ds, "size": n, "models": {}, "max_abs_err": {}}
    for ttype, task in (("denoise", "denoise_s0.10"), ("inpaint", "inpaint_mixed"), ("mri", "mri_R4")):
        spec = cfg["tasks"][task]
        d = degrade(x_all, spec, task_generator(task, "test", 0))
        jobs = [("hetero", HeteroWrap)] + ([("unrolled", None)] if ttype == "mri" else [])
        for method, W in jobs:
            ck = ckdir / f"{ttype}_{method}_s0.pt"
            if not ck.exists():
                print("skip (not trained yet):", ck.name)
                continue
            m = load_model(method, ck, tcfg, "cpu")
            wrap = (W(m) if W else UnrolledWrap(m, n)).eval()
            errs = []
            path = out / f"{ttype}_{method}.onnx"
            for i in range(len(x_all)):
                di = slice_d(d, slice(i, i + 1))
                c = condition(di, spec)
                if method == "hetero":
                    feeds = {"inp": di["inp"], "cond": c}
                    ref = list(wrap(di["inp"], c))
                    ref_model = m.predict(di, c)
                    assert torch.allclose(ref[0], ref_model["mean"], atol=1e-5) and torch.allclose(ref[1], ref_model["lo"], atol=1e-5)
                else:
                    feeds = {"inp": di["inp"], "cond": c, "mask": di["op"].mask, "y_re": di["y"].real.float(), "y_im": di["y"].imag.float()}
                    ref = [wrap(*feeds.values())]
                    ref_model = m.predict(di, c)["mean"]
                    e = float((ref[0] - ref_model).abs().max())
                    assert e < 1e-3, f"DFT rewrite differs from the torch FFT model by {e}"
                if i == 0:
                    export(wrap, tuple(feeds.values()), list(feeds), ["mean", "sd"] if method == "hetero" else ["mean"], path)
                errs.append(check(path, wrap, feeds, ref))
            err = max(errs)
            assert err < TOL, f"{path.name}: ONNX differs from PyTorch by {err}"
            manifest["models"][f"{ttype}_{method}"] = {"file": path.name, "bytes": path.stat().st_size,
                                                        "inputs": list(feeds), "checkpoint": ck.name}
            manifest["max_abs_err"][f"{ttype}_{method}"] = err
            print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB, max |ONNX - torch| = {err:.2e} over {len(x_all)} test images")
    manifest["degradations"] = {"denoise_sigmas": [0.03, 0.06, 0.10, 0.15], "mri_R": [2, 4, 8], "mri_sigma": 0.02,
                                "mri_center_frac": 0.08, "inpaint_sigma": 0.01}
    (out / "models.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=str(ROOT / "docs" / "models"))
    a = ap.parse_args()
    main(a.config, a.out)
