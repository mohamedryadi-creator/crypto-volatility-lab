from __future__ import annotations

import numpy as np
import pytest

from crypto_vol_lab.svi import (
    NoArbitrageError,
    RawSVIParams,
    SSVIParams,
    SSVISurface,
    SVIError,
    butterfly_diagnostic,
    calendar_arbitrage_diagnostic,
    fit_raw_svi,
    fit_ssvi_surface,
    price_arbitrage_diagnostic,
    raw_svi_butterfly_g,
    raw_svi_from_unconstrained,
    raw_svi_jacobian,
    raw_svi_total_variance,
    ssvi_butterfly_g,
    ssvi_sufficient_no_arbitrage,
    ssvi_total_variance,
)


def test_raw_svi_jacobian_matches_finite_differences() -> None:
    k = np.linspace(-0.8, 0.8, 17)
    params = RawSVIParams(a=0.025, b=0.16, rho=-0.45, m=0.04, sigma=0.22)
    analytic = raw_svi_jacobian(k, params)
    base = params.as_array()
    numerical = np.empty_like(analytic)
    step = 1.0e-6
    for column in range(5):
        plus = base.copy()
        minus = base.copy()
        plus[column] += step
        minus[column] -= step
        up = raw_svi_total_variance(k, RawSVIParams(*plus))
        down = raw_svi_total_variance(k, RawSVIParams(*minus))
        numerical[:, column] = (up - down) / (2.0 * step)
    np.testing.assert_allclose(analytic, numerical, rtol=2.0e-7, atol=2.0e-9)


def test_unconstrained_transform_enforces_admissible_parameters() -> None:
    for values in (
        [-100.0, -100.0, 100.0, -3.0, -100.0],
        [2.0, 5.0, -100.0, 10.0, 3.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ):
        params = raw_svi_from_unconstrained(values)
        assert params.b >= 0.0
        assert params.sigma > 0.0
        assert abs(params.rho) < 1.0
        assert params.minimum_total_variance >= 0.0


def test_fit_raw_svi_recovers_synthetic_smile_with_quote_weights() -> None:
    k = np.linspace(-1.0, 1.0, 61)
    truth = RawSVIParams(a=0.022, b=0.13, rho=-0.37, m=0.03, sigma=0.28)
    mid = np.asarray(raw_svi_total_variance(k, truth))
    half_spread = 2.0e-4 * (1.0 + np.abs(k))
    vega = np.exp(-1.2 * k * k)
    result = fit_raw_svi(
        k,
        mid,
        total_variance_bid=mid - half_spread,
        total_variance_ask=mid + half_spread,
        vega=vega,
        loss="linear",
    )
    fitted = raw_svi_total_variance(k, result.params)

    assert result.success
    assert result.inside_spread_fraction == 1.0
    assert result.rmse < 1.0e-5
    np.testing.assert_array_less(np.abs(fitted - mid), half_spread)


def test_power_law_ssvi_surface_is_monotone_and_arbitrage_checked() -> None:
    maturities = np.asarray([7.0, 30.0, 90.0, 180.0]) / 365.0
    theta = np.asarray([0.015, 0.045, 0.095, 0.16])
    params = SSVIParams(rho=-0.35, eta=0.35, gamma=0.5)
    report = ssvi_sufficient_no_arbitrage(theta, params)
    surface = SSVISurface(maturities, theta, params)

    assert report.passed
    dense_maturities = np.linspace(0.0, 1.0, 101)
    interpolated = surface.theta(dense_maturities)
    assert np.all(np.diff(interpolated) >= -1.0e-14)
    assert surface.total_variance(0.0, maturities[2]) == pytest.approx(theta[2])

    k = np.linspace(-2.0, 2.0, 401)
    g = ssvi_butterfly_g(k, theta[2], params)
    diagnostic = butterfly_diagnostic(k, g)
    assert diagnostic.passed
    assert diagnostic.minimum_g > 0.0

    calendar = calendar_arbitrage_diagnostic(
        surface, np.linspace(0.01, 0.8, 30), np.linspace(-1.5, 1.5, 51)
    )
    assert calendar.passed


def test_ssvi_rejects_parameters_that_fail_sufficient_conditions() -> None:
    maturities = [0.1, 0.5, 1.0]
    theta = [0.04, 0.12, 0.25]
    params = SSVIParams(rho=0.9, eta=10.0, gamma=0.1)
    assert not ssvi_sufficient_no_arbitrage(theta, params).passed
    with pytest.raises(NoArbitrageError):
        SSVISurface(maturities, theta, params)


def test_fit_ssvi_surface_recovers_synthetic_global_parameters() -> None:
    atm_maturities = np.asarray([0.08, 0.2, 0.5, 1.0])
    atm_theta = np.asarray([0.018, 0.038, 0.082, 0.145])
    truth = SSVIParams(rho=-0.42, eta=0.38, gamma=0.47)
    observation_maturities = np.repeat(atm_maturities, 31)
    k = np.tile(np.linspace(-1.2, 1.2, 31), atm_maturities.size)
    theta = np.repeat(atm_theta, 31)
    observed = ssvi_total_variance(k, theta, truth)
    weights = np.exp(-0.5 * k * k)

    result = fit_ssvi_surface(
        observation_maturities,
        k,
        observed,
        atm_maturities,
        atm_theta,
        weights=weights,
        loss="linear",
    )

    assert result.success
    assert result.no_arbitrage_report.passed
    assert result.rmse < 1.0e-9
    assert result.params.rho == pytest.approx(truth.rho, abs=2.0e-6)
    assert result.params.eta == pytest.approx(truth.eta, abs=2.0e-6)
    assert result.params.gamma == pytest.approx(truth.gamma, abs=2.0e-6)


def test_raw_svi_butterfly_and_price_shape_diagnostics() -> None:
    k = np.linspace(-1.5, 1.5, 301)
    params = RawSVIParams(a=0.04, b=0.08, rho=-0.2, m=0.0, sigma=0.4)
    raw = butterfly_diagnostic(k, raw_svi_butterfly_g(k, params))
    assert raw.passed

    strikes = np.asarray([80.0, 90.0, 100.0, 110.0, 120.0])
    valid_calls = np.asarray([22.0, 14.0, 8.0, 4.0, 2.0])
    valid = price_arbitrage_diagnostic(strikes, valid_calls, option_type="call")
    assert valid.passed

    non_convex_calls = np.asarray([22.0, 14.0, 5.0, 4.0, 2.0])
    invalid = price_arbitrage_diagnostic(
        strikes, non_convex_calls, option_type="call"
    )
    assert not invalid.passed
    assert invalid.convexity_violations > 0


def test_invalid_raw_svi_inputs_fail_clearly() -> None:
    with pytest.raises(SVIError, match="b"):
        RawSVIParams(a=0.1, b=-0.1, rho=0.0, m=0.0, sigma=0.2)
    with pytest.raises(SVIError, match="rho"):
        RawSVIParams(a=0.1, b=0.1, rho=1.0, m=0.0, sigma=0.2)
