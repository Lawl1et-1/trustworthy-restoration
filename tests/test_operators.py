import numpy as np
import torch

from src.degradations.degrade import degrade, rician
from src.degradations.masks import cartesian_mask, inpaint_mask
from src.degradations.operators import LinearOp

N = 32


def _ops(rng):
    m_mri = torch.from_numpy(np.stack([cartesian_mask(N, 4, "random", rng), cartesian_mask(N, 4, "equispaced", rng)]))[:, None]
    m_inp = torch.from_numpy(np.stack([inpaint_mask(N, "mixed", rng) for _ in range(2)]))[:, None]
    return [LinearOp("mri", m_mri), LinearOp("inpaint", m_inp)]


def test_adjoint():
    rng = np.random.default_rng(0)
    for op in _ops(rng):
        x = torch.randn(2, 1, N, N, dtype=torch.float64)
        y = op.A(torch.randn(2, 1, N, N, dtype=torch.float64))
        if torch.is_complex(y):
            y = y + 1j * op.A(torch.randn(2, 1, N, N, dtype=torch.float64)).real
        lhs = (torch.conj(op.A(x)) * y).real.sum() if torch.is_complex(y) else (op.A(x) * y).sum()
        rhs = (x * op.AT(y)).sum()
        assert torch.allclose(lhs, rhs, rtol=1e-9, atol=1e-9)


def test_projectors():
    rng = np.random.default_rng(1)
    for op in _ops(rng):
        x = torch.randn(2, 1, N, N, dtype=torch.float64)
        pr, pn = op.P_R(x), op.P_N(x)
        assert torch.allclose(op.P_R(pr), pr, atol=1e-10)              # idempotent
        assert torch.allclose(pr + pn, x)                              # decomposition
        assert abs(float((pr * pn).sum())) < 1e-8                      # orthogonal
        assert torch.allclose(op.A(pn), torch.zeros_like(op.A(pn)), atol=1e-10)  # null space


def test_mask_symmetry_and_acceleration():
    rng = np.random.default_rng(2)
    for R in (2, 4, 8):
        for kind in ("equispaced", "random"):
            m = cartesian_mask(128, R, kind, rng)
            rows = m[:, 0]
            assert all(rows[k] == rows[(-k) % 128] for k in range(128))
            assert abs(128 / rows.sum() - R) / R < 0.1


def test_range_noise_variance():
    gen = torch.Generator().manual_seed(0)
    x = torch.zeros(400, 1, N, N)
    d = degrade(x, {"type": "mri", "R": 4, "sigma": 0.05}, gen)
    emp = d["inp"].var(dim=0).mean()
    pred = d["op"].range_noise_var(0.05).mean()
    assert abs(float(emp / pred) - 1) < 0.05


def test_data_consistency_noiseless():
    gen = torch.Generator().manual_seed(1)
    x = torch.rand(3, 1, N, N)
    for spec in ({"type": "mri", "R": 4, "sigma": 0.0}, {"type": "inpaint", "sigma": 0.0}):
        d = degrade(x, spec, gen)
        z = d["op"].data_consistency(torch.zeros_like(x), d["y"])
        assert torch.allclose(d["op"].A(z), d["y"], atol=1e-5)


def test_rician_bias_positive():
    gen = torch.Generator().manual_seed(0)
    x = torch.zeros(1, 1, 64, 64)
    assert rician(x, 0.1, gen).mean() > 0.1  # Rayleigh mean = sigma*sqrt(pi/2)
