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

One network per degradation type covers all its severities. Input channels: [degraded image, condition],
condition = noise-level map (denoise), 1/R map (MRI), measurement mask (inpainting).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets import UNet, Unrolled

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
    def __init__(self, method, base=16, p_drop=0.1, K=5, nll_mode="plain", nll_beta=0.5):
        super().__init__()
        assert nll_mode in ("plain", "beta", "faithful"), nll_mode
        self.method = method
        self.nll_mode, self.nll_beta = nll_mode, nll_beta
        self.faithful = method == "hetero" and nll_mode == "faithful"
        if method == "unrolled":
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

    def loss(self, out, x, step, warmup):
        if self.method in ("unet", "unrolled"):
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
    def predict(self, d, cond, mc_samples=0):
        """Prediction dict. mc_samples > 0 turns dropout on (unet only) for MC-dropout."""
        if self.method == "unet" and mc_samples:
            self.net.train()  # only dropout layers depend on train/eval (GroupNorm has no running stats)
            s = torch.stack([self.forward(d, cond) for _ in range(mc_samples)])
            self.net.eval()
            sd = s.std(0)
            return {"mean": s.mean(0), "lo": sd, "hi": sd}
        out = self.forward(d, cond)
        if self.method in ("unet", "unrolled"):
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
