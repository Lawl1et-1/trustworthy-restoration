"""Small CPU-sized networks: U-Net (optionally time-conditioned), unrolled DC network, PatchGAN."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c):
    return nn.GroupNorm(min(8, c), c)


class Block(nn.Module):
    def __init__(self, cin, cout, tdim=0, p_drop=0.0):
        super().__init__()
        self.c1, self.n1 = nn.Conv2d(cin, cout, 3, padding=1), _gn(cout)
        self.c2, self.n2 = nn.Conv2d(cout, cout, 3, padding=1), _gn(cout)
        self.t = nn.Linear(tdim, cout) if tdim else None
        self.drop = nn.Dropout2d(p_drop) if p_drop else nn.Identity()
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb=None):
        h = F.silu(self.n1(self.c1(x)))
        if self.t is not None:
            h = h + self.t(temb)[:, :, None, None]
        h = self.drop(h)
        return F.silu(self.n2(self.c2(h))) + self.skip(x)


def timestep_embedding(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class UNet(nn.Module):
    """3-level U-Net. `time=True` adds a sinusoidal timestep embedding (diffusion denoiser)."""

    def __init__(self, cin=2, cout=1, base=16, p_drop=0.0, time=False):
        super().__init__()
        ch = [base, 2 * base, 4 * base, 8 * base]
        self.tdim = 4 * base if time else 0
        if time:
            self.tmlp = nn.Sequential(nn.Linear(base, self.tdim), nn.SiLU(), nn.Linear(self.tdim, self.tdim))
        self.inc = nn.Conv2d(cin, ch[0], 3, padding=1)
        self.down = nn.ModuleList([Block(ch[i], ch[i + 1], self.tdim) for i in range(3)])
        self.mid = Block(ch[3], ch[3], self.tdim, p_drop)
        self.up = nn.ModuleList([Block(ch[i + 1] + ch[i], ch[i], self.tdim, p_drop) for i in reversed(range(3))])
        self.e0 = Block(ch[0], ch[0], self.tdim)
        self.out = nn.Conv2d(ch[0], cout, 1)
        self.base = base

    def forward(self, x, t=None):
        temb = self.tmlp(timestep_embedding(t, self.base)) if self.tdim else None
        h = self.e0(self.inc(x), temb)
        skips = [h]
        for blk in self.down:
            h = blk(F.avg_pool2d(h, 2), temb)
            skips.append(h)
        h = self.mid(skips.pop(), temb)
        for blk in self.up:
            h = blk(torch.cat([F.interpolate(h, scale_factor=2, mode="nearest"), skips.pop()], 1), temb)
        return self.out(h)


class Unrolled(nn.Module):
    """K cascades of (small CNN refinement -> learnable soft data consistency) for linear operators."""

    def __init__(self, K=5, base=8):
        super().__init__()
        self.cnns = nn.ModuleList([UNet(2, 1, base) for _ in range(K)])
        self.eta = nn.Parameter(torch.full((K,), 0.5))

    def forward(self, inp, ind, op, y):
        x = inp
        aty = op.AT(y)
        for cnn, eta in zip(self.cnns, self.eta):
            x = x + cnn(torch.cat([x, ind], 1))
            x = x + torch.sigmoid(eta) * (aty - op.P_R(x))
        return x


class PatchGAN(nn.Module):
    """pix2pix discriminator on (condition, image) pairs; ~34x34 receptive field."""

    def __init__(self, cin=3, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, base, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(base, 2 * base, 4, 2, 1), _gn(2 * base), nn.LeakyReLU(0.2),
            nn.Conv2d(2 * base, 4 * base, 4, 1, 1), _gn(4 * base), nn.LeakyReLU(0.2),
            nn.Conv2d(4 * base, 1, 4, 1, 1))

    def forward(self, x):
        return self.net(x)
