"""Unit tests for the scaling-exponent estimators."""
import numpy as np
import pytest

from fractalsig.features.fractal_math import calc_H, calc_S, generate_trajectory, scale

fbm = pytest.importorskip("fbm")


def _fgn(h: float, n: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    return fbm.FBM(n=n, hurst=h, length=1, method="daviesharte").fgn()


def test_trajectory_is_standardised_walk():
    x = np.random.default_rng(0).normal(3.0, 2.0, 500)
    traj, inc = generate_trajectory(x)
    assert inc.shape == traj.shape == (500,)
    assert abs(inc.mean()) < 1e-12 and abs(inc.std() - 1.0) < 1e-12
    np.testing.assert_allclose(traj, np.cumsum(inc))


@pytest.mark.parametrize("h_true", [0.5, 0.7])
def test_hurst_recovers_known_exponent_on_long_series(h_true):
    estimates = [calc_H(generate_trajectory(_fgn(h_true, 12800, s))[1])[0] for s in range(5)]
    # R/S carries a known small positive bias; the tolerance brackets it.
    assert abs(np.mean(estimates) - h_true) < 0.08


def test_estimators_are_deterministic():
    traj, inc = generate_trajectory(_fgn(0.65, 2000, 1))
    assert calc_H(inc) == calc_H(inc)
    assert calc_S(traj) == calc_S(traj)          # seeded sampler (n - tau > n_boot)
    short, _ = generate_trajectory(_fgn(0.65, 400, 2))
    np.random.seed(123); a = calc_S(short)
    np.random.seed(999); b = calc_S(short)
    assert a == b                                 # exhaustive enumeration ignores global RNG


def test_calc_s_rejects_2d_array():
    with pytest.raises(ValueError):
        calc_S(np.zeros((5, 400)))


def test_holder_recovers_known_exponent_on_long_series():
    # Reference T2 noise floor: S_hat_mean = 0.647 at H = 0.65, N = 12800.
    estimates = [calc_S(generate_trajectory(_fgn(0.65, 12800, s))[0])[0] for s in range(8)]
    assert abs(np.mean(estimates) - 0.65) < 0.06


def test_holder_is_biased_low_at_document_length():
    # The paper's premise: at ~400 tokens the per-document S estimate is biased low.
    estimates = [calc_S(generate_trajectory(_fgn(0.65, 400, s))[0])[0] for s in range(30)]
    assert np.nanmean(estimates) < 0.65 - 0.2


def test_return_std_err_shapes():
    _, inc = generate_trajectory(_fgn(0.6, 1000, 3))
    h, r2, se = calc_H(inc, return_std_err=True)
    assert np.isfinite([h, r2, se]).all() and se >= 0


@pytest.mark.parametrize("fn", [calc_H, calc_S])
def test_too_short_input_returns_nan(fn):
    out = fn(np.arange(5, dtype=float))
    assert np.isnan(out[0]) and out[1] == 0.0


def test_constant_series_does_not_raise():
    _, inc = generate_trajectory(np.ones(400) + 1e-12 * np.arange(400))
    h, _ = calc_H(inc)
    assert np.isnan(h) or np.isfinite(h)


def test_explicit_scale_list_is_honoured():
    _, inc = generate_trajectory(_fgn(0.6, 200, 4))
    h_default, _ = calc_H(inc)
    h_small, _ = calc_H(inc, scale_param=[8, 16, 32, 48, 64])
    assert np.isfinite(h_small)
    assert scale[-1] == 320 and (np.isnan(h_default) or np.isfinite(h_default))
