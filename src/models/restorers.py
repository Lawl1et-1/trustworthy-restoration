"""M3 learned restorers: one interface for every method.

Every model maps a degradation dict (from degrade()) to a prediction dict
  mean (B,1,H,W)        point estimate
  lo, hi (B,1,H,W)      spreads below / above the mean (>= 0); symmetric methods have lo == hi = sigma
so the conformal code (M6) can treat them all the same way: interval = [mean - lam*lo, mean + lam*hi].

Methods
  unet        U-Net, L1 loss, dropout p = 0.1 in the decoder. Deterministic output = dropout off.
  mcdropout   the same trained `unet` weights, dropout on, T samples: mean and std (Gal & Ghahramani 2016).
  hetero      U-Net with a mean and a log-variance head, Gaussian NLL (Kendall & Gal 2017), with an MSE warm-up;
              `nll_mode`: plain, beta (beta-NLL, Seitzer et al. 2022) or faithful (MSE mean + NLL variance on
              a frozen mean, Stirn et al. 2023); chosen by pilot, see LOG.md (M3).
  ensemble    5 `hetero` seeds: mixture mean, total std = sqrt(mean of variances + variance of means)
              (Lakshminarayanan et al. 2017). Built at evaluation time, not trained separately.
  quantile    U-Net with 3 outputs for the 5 / 50 / 95 % quantiles, pinball loss; sorted to avoid crossing.
  unrolled    MRI only: K cascades of a small U-Net and soft data consistency (deterministic).

M4 generative restorers
  diffusion   SR3-style conditional diffusion (Saharia et al. 2022): epsilon-prediction U-Net with a timestep
              embedding, conditioned on [degraded image, condition]; DDIM sampling (Song et al. 2021). K samples
              per input: mean = sample mean, lo = hi = sample std, plus `sample` = the first draw.
  gan         pix2pix (Isola et al. 2017): residual U-Net generator, PatchGAN discriminator on
              (degraded image, condition, image), L1 x `gan_l1` + non-saturating adversarial loss. Deterministic.

One network per degradation type covers all its severities. Input channels: [degraded image, condition],
condition = noise-level map (denoise), 1/R map (MRI), measurement mask (inpainting).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import Diffusion
from .nets import PatchGAN, UNet, Unrolled

QUANTILES = (0.05, 0.5, 0.95)
LOGVAR_MIN, LOGVAR_MAX = -14.0, 2.0


def condition(d, spec):
    """Second input channel, so one network can serve every severity of a degradation type."""
    if spec["type"] == "denoise":
        return torch.full_like(d["inp"], float(spec["sigma"]))
    if spec["type"] == "mri":
        return torch.full_like(d["inp"], 1.0 / float(spec["R"]))
    return d["ind"]


class Restorer(nn.Module):
    def __init__(self, method, base=16, p_drop=0.1, K=5, nll_mode="plain", nll_beta=0.5, gan_l1=100.0):
        super().__init__()
        assert nll_mode in ("plain", "beta", "faithful"), nll_mode
        self.method = method
        self.nll_mode, self.nll_beta = nll_mode, nll_beta
        self.faithful = method == "hetero" and nll_mode == "faithful"
        if method == "diffusion":
            self.net = UNet(3, 1, base, time=True)  # [x_t, degraded image, condition]
            self.diff = Diffusion()
        elif method == "gan":
            self.net = UNet(2, 1, base)
            self.disc = PatchGAN(cin=3, base=2 * base)
            self.gan_l1 = gan_l1
        elif method == "unrolled":
            self.net = Unrolled(K=K, base=base // 2)
        elif self.faithful:
            # Stirn et al. 2023: the U-Net only ever sees the MSE gradient; a separate variance head reads its
            # features with gradients blocked (plus the input and condition), so the mean is unaffected
            self.net = UNet(2, 1, base)
            self.var_head = nn.Sequential(nn.Conv2d(base + 2, base, 3, padding=1), nn.SiLU(),
                                          nn.Conv2d(base, base, 3, padding=1), nn.SiLU(), nn.Conv2d(base, 1, 1))
        else:
            cout = {"unet": 1, "hetero": 2, "quantile": len(QUANTILES)}[method]
            self.net = UNet(2, cout, base, p_drop=p_drop if method == "unet" else 0.0)

    def forward(self, d, cond):
        """Raw network output; the input image is added back (residual learning) for every mean channel."""
        if self.method == "unrolled":
            return self.net(d["inp"], cond, d["op"], d["y"])
        if self.faithful:
            h = self.net.features(torch.cat([d["inp"], cond], 1))
            lv = self.var_head(torch.cat([h.detach(), d["inp"], cond], 1))
            return torch.cat([d["inp"] + self.net.out(h), lv.clamp(LOGVAR_MIN, LOGVAR_MAX)], 1)
        out = self.net(torch.cat([d["inp"], cond], 1))
        if self.method == "hetero":
            return torch.cat([d["inp"] + out[:, :1], out[:, 1:].clamp(LOGVAR_MIN, LOGVAR_MAX)], 1)
        return d["inp"] + out

    # ---- M4: generative training objectives (the training loop calls these instead of forward + loss)

    @staticmethod
    def _to_pm1(v):
        return 2 * v - 1

    def diffusion_loss(self, d, cond, x, gen):
        """Epsilon-prediction loss on x in [-1, 1], conditioned on the degraded image and the condition map."""
        c = torch.cat([self._to_pm1(d["inp"]), cond], 1)
        return self.diff.loss(self.net, self._to_pm1(x), c, gen)

    def gan_d_loss(self, d, cond, fake, x):
        """pix2pix discriminator loss on (degraded image, condition, image) triples; `fake` is detached."""
        pair = torch.cat([d["inp"], cond], 1)
        logit_real = self.disc(torch.cat([pair, x], 1))
        logit_fake = self.disc(torch.cat([pair, fake.detach()], 1))
        return 0.5 * (F.binary_cross_entropy_with_logits(logit_real, torch.ones_like(logit_real)) +
                      F.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake)))

    def gan_g_loss(self, d, cond, fake, x):
        """Generator loss: L1 x gan_l1 + non-saturating adversarial term (call after the D step)."""
        logit = self.disc(torch.cat([d["inp"], cond, fake], 1))
        return self.gan_l1 * F.l1_loss(fake, x) + F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))

    @staticmethod
    def dc_step(d, eta):
        """Data consistency on the predicted clean image u in [0, 1]: u + eta * (A^+ y - P_R u).
        eta = 1 replaces the measured part by the measurement (hard DC); linear degradations only."""
        op, y = d["op"], d["y"]
        aty = op.AT(y)

        def f(x0_pm1):
            u = (x0_pm1 + 1) / 2
            return 2 * (u + eta * (aty - op.P_R(u))) - 1
        return f

    @torch.no_grad()
    def sample(self, d, cond, n_samples, steps, gen, dc_eta=0.0):
        """(n_samples, B, 1, H, W) diffusion samples in [0, 1]-scale (not clipped).
        dc_eta > 0 enforces data consistency at every DDIM step (MRI, inpainting)."""
        c = torch.cat([self._to_pm1(d["inp"]), cond], 1)
        dc = self.dc_step(d, dc_eta) if dc_eta and d["op"] is not None else None
        return torch.stack([(self.diff.sample(self.net, c, steps, gen, dc) + 1) / 2 for _ in range(n_samples)])

    def loss(self, out, x, step, warmup):
        if self.method in ("unet", "unrolled", "gan"):
            return F.l1_loss(out, x)
        if self.method == "hetero":
            mu, lv = out[:, :1], out[:, 1:]
            if step < warmup:  # fit the mean first; NLL from the start is unstable (well-known failure mode)
                return F.mse_loss(mu, x)
            if self.nll_mode == "faithful":
                # Stirn et al. 2023: MSE for the mean, NLL for the variance against a frozen mean, so the
                # variance head cannot pull the mean away from the MSE solution
                return F.mse_loss(mu, x) + (0.5 * (lv + (x - mu.detach()) ** 2 * torch.exp(-lv))).mean()
            nll = 0.5 * (lv + (x - mu) ** 2 * torch.exp(-lv))
            if self.nll_mode == "beta":  # beta-NLL (Seitzer et al. 2022): weight by stop-grad(var^beta)
                nll = nll * torch.exp(self.nll_beta * lv).detach()
            return nll.mean()
        if self.method == "quantile":
            q = torch.tensor(QUANTILES, device=out.device).view(1, -1, 1, 1)
            e = x - out
            return torch.maximum(q * e, (q - 1) * e).mean()
        raise ValueError(self.method)

    @torch.no_grad()
    def predict(self, d, cond, mc_samples=0, n_samples=8, steps=50, gen=None, dc_eta=0.0):
        """Prediction dict. mc_samples > 0 turns dropout on (unet only) for MC-dropout.
        Diffusion: n_samples DDIM draws with `steps` steps from generator `gen` (on the input's device),
        with data consistency of strength dc_eta (0 = none)."""
        if self.method == "diffusion":
            s = self.sample(d, cond, n_samples, steps, gen, dc_eta)
            sd = s.std(0) if n_samples > 1 else torch.zeros_like(s[0])
            return {"mean": s.mean(0), "lo": sd, "hi": sd, "sample": s[0]}
        if self.method == "unet" and mc_samples:
            self.net.train()  # only dropout layers depend on train/eval (GroupNorm has no running stats)
            s = torch.stack([self.forward(d, cond) for _ in range(mc_samples)])
            self.net.eval()
            sd = s.std(0)
            return {"mean": s.mean(0), "lo": sd, "hi": sd}
        out = self.forward(d, cond)
        if self.method in ("unet", "unrolled", "gan"):
            z = torch.zeros_like(out)
            return {"mean": out, "lo": z, "hi": z}
        if self.method == "hetero":
            sd = torch.exp(0.5 * out[:, 1:])
            return {"mean": out[:, :1], "lo": sd, "hi": sd}
        qs = out.sort(1).values  # no quantile crossing
        med = qs[:, 1:2]
        return {"mean": med, "lo": med - qs[:, :1], "hi": qs[:, 2:] - med}


def ensemble(preds):
    """Mixture of Gaussian members: mean of means, std = sqrt(mean variance + variance of means)."""
    mus = torch.stack([p["mean"] for p in preds])
    var = torch.stack([p["lo"] ** 2 for p in preds]).mean(0) + mus.var(0, unbiased=False)
    sd = var.sqrt()
    return {"mean": mus.mean(0), "lo": sd, "hi": sd}
