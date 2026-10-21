"""Degradations D1 (Rician denoising), D2 (accelerated MRI) and D3 (inpainting).

Every degradation returns the same dict so restorers are interchangeable:
  inp  (B,1,H,W)  network input: A^+ y (D2, D3) or the noisy image (D1)
  ind  (B,1,H,W)  measurement indicator: inpainting mask, ones otherwise
  y               measurements (None for D1)
  op              LinearOp (None for D1, which is non-linear)
  sigma           image-domain noise std
"""
import numpy as np
import torch

from .masks import cartesian_mask, inpaint_mask
from .operators import LinearOp

RICIAN_LEVELS = (0.03, 0.06, 0.10, 0.15)


def rician(x, sigma, gen):
    n1 = torch.randn(x.shape, generator=gen) * sigma
    n2 = torch.randn(x.shape, generator=gen) * sigma
    return torch.sqrt((x + n1) ** 2 + n2 ** 2)


def _np_rng(gen):
    return np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))


def degrade(x, spec, gen):
    """x: (B,1,H,W) clean images in [0,1]. spec: dict from the YAML config."""
    B, _, H, W = x.shape
    t = spec["type"]
    if t == "denoise":
        sigma = float(spec["sigma"])
        return dict(inp=rician(x, sigma, gen), ind=torch.ones_like(x), y=None, op=None, sigma=sigma)
    rng = _np_rng(gen)
    if t == "mri":
        kinds = spec.get("mask", "mixed")
        masks = []
        for _ in range(B):
            k = ("equispaced", "random")[int(rng.integers(2))] if kinds == "mixed" else kinds
            masks.append(cartesian_mask(H, spec["R"], k, rng))
        op = LinearOp("mri", torch.from_numpy(np.stack(masks))[:, None])
        sigma = float(spec.get("sigma", 0.02))
        # complex noise with per-component std sigma: a fully sampled image would have pixel std sigma
        noise = torch.complex(torch.randn(x.shape, generator=gen), torch.randn(x.shape, generator=gen))
        y = op.A(x) + sigma * noise * op.mask
        return dict(inp=op.AT(y), ind=torch.ones_like(x), y=y, op=op, sigma=sigma)
    if t == "inpaint":
        m = torch.from_numpy(np.stack([inpaint_mask(H, spec.get("mask", "mixed"), rng) for _ in range(B)]))[:, None]
        op = LinearOp("inpaint", m)
        sigma = float(spec.get("sigma", 0.01))
        y = op.A(x + sigma * torch.randn(x.shape, generator=gen))
        return dict(inp=op.AT(y), ind=op.mask, y=y, op=op, sigma=sigma)
    raise ValueError(t)
