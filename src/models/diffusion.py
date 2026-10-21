"""SR3-style conditional diffusion (epsilon prediction, linear schedule) with DDIM sampling."""
import torch


class Diffusion:
    def __init__(self, T=1000, beta1=1e-4, betaT=0.02):
        self.T = T
        betas = torch.linspace(beta1, betaT, T)
        self.abar = torch.cumprod(1 - betas, 0)

    def loss(self, net, x0, cond, gen=None):
        """x0 is in [-1, 1]; cond is (B, C, H, W) conditioning stacked as extra input channels."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), generator=gen)
        eps = torch.randn(x0.shape, generator=gen)
        a = self.abar[t].view(B, 1, 1, 1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        return (net(torch.cat([xt, cond], 1), t) - eps).pow(2).mean()

    @torch.no_grad()
    def sample(self, net, cond, steps, gen):
        """Deterministic DDIM (eta = 0); sample diversity comes from the initial noise."""
        B, _, H, W = cond.shape
        x = torch.randn((B, 1, H, W), generator=gen)
        ts = torch.linspace(self.T - 1, 0, steps).long()
        for i, t in enumerate(ts):
            a = self.abar[t]
            eps = net(torch.cat([x, cond], 1), t.repeat(B))
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            a_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x
