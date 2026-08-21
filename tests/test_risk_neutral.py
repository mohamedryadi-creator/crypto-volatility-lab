from __future__ import annotations

import numpy as np
import pytest

from crypto_vol_lab.pricing import black76_price
from crypto_vol_lab.risk_neutral import (
    RiskNeutralError,
    breeden_litzenberger_density,
    density_from_ssvi,
    model_free_variance,
    model_free_variance_from_chain,
)
from crypto_vol_lab.svi import SSVIParams, SSVISurface


def test_breeden_litzenberger_recovers_lognormal_moments() -> None:
    forward = 100.0
    maturity = 0.75
    volatility = 0.3
    discount = 0.98
    strikes = np.linspace(8.0, 350.0, 1_501)
    calls = black76_price(
        forward, strikes, maturity, volatility, discount, "call"
    )
    distribution = breeden_litzenberger_density(
        strikes,
        calls,
        discount_factor=discount,
        evaluation_strikes=np.linspace(8.0, 350.0, 3_001),
    )

    theoretical_variance = forward**2 * (np.exp(volatility**2 * maturity) - 1.0)
    assert distribution.raw_mass == pytest.approx(1.0, rel=2.0e-4)
    assert distribution.cdf[0] == 0.0
    assert distribution.cdf[-1] == 1.0
    assert distribution.mean == pytest.approx(forward, rel=3.0e-4)
    assert distribution.variance == pytest.approx(theoretical_variance, rel=3.0e-3)
    assert distribution.negative_mass_removed < 1.0e-7


def test_density_from_ssvi_is_normalized_and_positive() -> None:
    surface = SSVISurface(
        [0.1, 0.25, 0.5, 1.0],
        [0.018, 0.038, 0.072, 0.13],
        SSVIParams(rho=-0.3, eta=0.3, gamma=0.5),
    )
    strikes = np.linspace(15.0, 350.0, 1_001)
    distribution = density_from_ssvi(surface, 0.5, 100.0, strikes)

    assert distribution.raw_mass > 0.98
    assert np.all(distribution.density >= 0.0)
    assert np.all(np.diff(distribution.cdf) >= -1.0e-14)
    assert 95.0 < distribution.mean < 105.0
    assert distribution.variance > 0.0


def test_model_free_variance_recovers_flat_black_variance() -> None:
    forward = 100.0
    maturity = 0.6
    volatility = 0.24
    discount = 0.985
    standard_deviation = volatility * np.sqrt(maturity)
    strikes = forward * np.exp(
        np.linspace(-7.0 * standard_deviation, 7.0 * standard_deviation, 5_001)
    )
    calls = black76_price(
        forward, strikes, maturity, volatility, discount, "call"
    )
    puts = black76_price(
        forward, strikes, maturity, volatility, discount, "put"
    )
    variance = model_free_variance_from_chain(
        strikes,
        calls,
        puts,
        forward=forward,
        maturity=maturity,
        discount_factor=discount,
    )
    assert variance == pytest.approx(volatility**2, rel=2.0e-4)


def test_model_free_variance_validates_wings_and_direct_otm_api() -> None:
    strikes = np.asarray([80.0, 90.0, 100.0, 110.0, 120.0])
    otm_prices = np.asarray([0.2, 1.0, 4.0, 1.2, 0.3])
    result = model_free_variance(
        strikes, otm_prices, forward=100.0, maturity=0.5
    )
    assert result > 0.0
    with pytest.raises(RiskNeutralError, match="inside"):
        model_free_variance(
            strikes, otm_prices, forward=150.0, maturity=0.5
        )
