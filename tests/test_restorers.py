import torch

from src.degradations.degrade import degrade, task_generator
from src.models.restorers import QUANTILES, Restorer, condition, ensemble

X = torch.rand(4, 1, 32, 32)
SPECS = {"denoise": {"type": "denoise", "sigma": 0.1}, "mri": {"type": "mri", "R": 4, "sigma": 0.02},
         "inpaint": {"type": "inpaint", "sigma": 0.01}}


def _d(kind):
    return degrade(X, SPECS[kind], task_generator("t", "test", 0))


def test_every_method_gives_valid_prediction_and_finite_loss():
    for method in ("unet", "hetero", "quantile", "unrolled"):
        for kind in (("mri",) if method == "unrolled" else SPECS):
            torch.manual_seed(0)
            m = Restorer(method, base=8)
            d = _d(kind)
            c = condition(d, SPECS[kind])
            out = m(d, c)
            for step in (0, 100):
                assert torch.isfinite(m.loss(out, X, step, warmup=10)), (method, kind)
            m.eval()
            p = m.predict(d, c)
            assert p["mean"].shape == X.shape and (p["lo"] >= 0).all() and (p["hi"] >= 0).all(), (method, kind)


def test_condition_channel_encodes_severity():
    assert torch.allclose(condition(_d("denoise"), SPECS["denoise"]), torch.full_like(X, 0.1))
    assert torch.allclose(condition(_d("mri"), SPECS["mri"]), torch.full_like(X, 0.25))
    d = _d("inpaint")
    assert torch.equal(condition(d, SPECS["inpaint"]), d["ind"])


def test_quantile_outputs_are_sorted_and_pinball_is_minimised_by_true_quantile():
    m = Restorer("quantile", base=8).eval()
    d = _d("denoise")
    p = m.predict(d, condition(d, SPECS["denoise"]))
    assert (p["lo"] >= 0).all() and (p["hi"] >= 0).all()
    # pinball loss of a constant prediction on N(0,1) samples is lowest at the true quantile
    torch.manual_seed(0)
    y = torch.randn(20000, 1, 1, 1)
    for k, q in enumerate(QUANTILES):
        losses = {}
        for c in (-2.0, -1.645, -1.0, 0.0, 1.0, 1.645, 2.0):
            out = torch.zeros(1, len(QUANTILES), 1, 1).expand(len(y), -1, -1, -1).clone()
            out[:, k] = c
            e = y - out[:, k:k + 1]
            losses[c] = float(torch.maximum(q * e, (q - 1) * e).mean())
        best = min(losses, key=losses.get)
        assert abs(best - {0.05: -1.645, 0.5: 0.0, 0.95: 1.645}[q]) < 1e-9, (q, best)


def test_mc_dropout_has_spread_and_restores_eval_mode():
    torch.manual_seed(0)
    m = Restorer("unet", base=8, p_drop=0.5).eval()
    d = _d("denoise")
    c = condition(d, SPECS["denoise"])
    p = m.predict(d, c, mc_samples=8)
    assert p["lo"].mean() > 0
    assert not m.net.training
    q1, q2 = m.predict(d, c), m.predict(d, c)
    assert torch.equal(q1["mean"], q2["mean"])  # deterministic with dropout off


def test_ensemble_total_variance():
    a = {"mean": torch.zeros(1, 1, 2, 2), "lo": torch.ones(1, 1, 2, 2), "hi": torch.ones(1, 1, 2, 2)}
    b = {"mean": torch.full((1, 1, 2, 2), 2.0), "lo": torch.ones(1, 1, 2, 2), "hi": torch.ones(1, 1, 2, 2)}
    e = ensemble([a, b])
    assert torch.allclose(e["mean"], torch.ones_like(e["mean"]))
    assert torch.allclose(e["lo"], torch.full_like(e["lo"], 2.0 ** 0.5))  # var 1 (aleatoric) + var 1 (spread)


def test_hetero_warmup_uses_mse_then_nll():
    m = Restorer("hetero", base=8)
    out = torch.cat([X + 0.1, torch.zeros_like(X)], 1)
    assert torch.isclose(m.loss(out, X, step=0, warmup=5), torch.tensor(0.01), atol=1e-6)
    assert torch.isclose(m.loss(out, X, step=10, warmup=5), torch.tensor(0.005), atol=1e-6)  # 0.5 * (0 + 0.01)


def test_faithful_mode_mean_gets_only_the_mse_gradient():
    m = Restorer("hetero", base=8, nll_mode="faithful")
    mu = (X + 0.1).requires_grad_(True)
    lv = torch.full_like(X, -2.0).requires_grad_(True)
    m.loss(torch.cat([mu, lv], 1), X, step=10, warmup=5).backward()
    assert torch.allclose(mu.grad, 2 * 0.1 * torch.ones_like(X) / X.numel(), atol=1e-9)  # d MSE / d mu only
    assert lv.grad.abs().sum() > 0


def test_faithful_variance_head_does_not_touch_the_unet():
    """The NLL term alone must give zero gradient to every U-Net weight (only the variance head learns)."""
    torch.manual_seed(0)
    m = Restorer("hetero", base=8, nll_mode="faithful")
    d = _d("denoise")
    out = m(d, condition(d, SPECS["denoise"]))
    mu, lv = out[:, :1], out[:, 1:]
    (0.5 * (lv + (X - mu.detach()) ** 2 * torch.exp(-lv))).mean().backward()
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in m.net.parameters())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in m.var_head.parameters())
