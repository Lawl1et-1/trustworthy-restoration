"""SR3-style conditional diffusion (epsilon prediction, linear schedule) with DDIM sampling."""
import torch


class Diffusion:
    def __init__(self, T=1000, beta1=1e-4, betaT=0.02):
        self.T = T
        betas = torch.linspace(beta1, betaT, T)
        self.abar = torch.cumprod(1 - betas, 0)

    def loss(self, net, x0, cond, gen=None):
        """x0 is in [-1, 1]; cond is (B, C, H, W) conditioning stacked as extra input channels.
        `gen` is a CPU generator; the draws are moved to x0's device."""
        B, dev = x0.shape[0], x0.device
        t = torch.randint(0, self.T, (B,), generator=gen).to(dev)
        eps = torch.randn(x0.shape, generator=gen).to(dev)
        a = self.abar.to(dev)[t].view(B, 1, 1, 1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        return (net(torch.cat([xt, cond], 1), t) - eps).pow(2).mean()

    @torch.no_grad()
    def sample(self, net, cond, steps, gen, dc=None):
        """Deterministic DDIM (eta = 0); sample diversity comes from the initial noise.
        `gen` must live on cond's device. `dc`, if given, maps the predicted x0 (in [-1, 1]) to a version
        consistent with the measurements; it is applied at every step before the next state is formed."""
        B, _, H, W = cond.shape
        dev = cond.device
        abar = self.abar.to(dev)
        x = torch.randn((B, 1, H, W), generator=gen, device=dev)
        ts = torch.linspace(self.T - 1, 0, steps).long().to(dev)
        for i, t in enumerate(ts):
            a = abar[t]
            eps = net(torch.cat([x, cond], 1), t.repeat(B))
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            if dc is not None:
                x0 = dc(x0).clamp(-1, 1)
            a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
            # recompute eps from the clamped x0 so the trajectory stays consistent with it
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x
