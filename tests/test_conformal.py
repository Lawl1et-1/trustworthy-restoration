import numpy as np

from src.conformal.risk import LAMBDA_GRID, crc_lambda, hb_pvalue, rcps_lambda, risk_curves


def _synthetic(n, rng, scale=1.0):
    x = rng.normal(size=(n, 16, 16)) * scale
    sig = np.ones_like(x)
    return risk_curves(x, np.zeros_like(x), sig, sig, np.ones_like(x, bool))


def test_risk_curve_matches_gaussian():
    risk, width = _synthetic(50, np.random.default_rng(0))
    j = np.argmin(abs(LAMBDA_GRID - 1.96))
    assert abs(risk[:, j].mean() - 0.05) < 0.01
    assert np.all(np.diff(risk.mean(0)) <= 1e-9)  # monotone in lambda


def test_rcps_coverage_holds_over_repeats():
    """Over repeated cal/test draws, the test risk exceeds alpha in at most ~delta of trials."""
    rng = np.random.default_rng(1)
    alpha, delta, fails = 0.1, 0.1, 0
    for _ in range(100):
        cal, _ = _synthetic(200, rng, scale=rng.uniform(0.5, 2))
        test, _ = _synthetic(2000, rng, scale=1.0)
        cal2, _ = _synthetic(200, rng, scale=1.0)
        j = rcps_lambda(cal2, alpha, delta)
        fails += test[:, j].mean() > alpha
    assert fails <= 15


def test_crc_expected_risk():
    rng = np.random.default_rng(2)
    risks = []
    for _ in range(200):
        cal, _ = _synthetic(50, rng)
        test, _ = _synthetic(1, rng)
        risks.append(test[0, crc_lambda(cal, 0.1)])
    assert np.mean(risks) <= 0.1 + 0.01


def test_hb_pvalue_monotone():
    p = [hb_pvalue(r, 100, 0.1) for r in (0.02, 0.05, 0.08, 0.12)]
    assert all(a <= b for a, b in zip(p, p[1:]))
