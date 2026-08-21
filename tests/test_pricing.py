from __future__ import annotations

import numpy as np
import pytest

from crypto_vol_lab.pricing import (
    ArbitrageBoundsError,
    ImpliedVolatilityError,
    PricingError,
    black76_greeks,
    black76_price,
    black76_price_bounds,
    implied_volatility,
)


def test_black76_atm_value_and_put_call_parity() -> None:
    forward = 100.0
    strike = 100.0
    maturity = 1.0
    volatility = 0.2
    discount = 0.97
    call = black76_price(forward, strike, maturity, volatility, discount, "call")
    put = black76_price(forward, strike, maturity, volatility, discount, "put")

    assert call == pytest.approx(discount * 7.965567455405804, rel=1.0e-12)
    assert call - put == pytest.approx(discount * (forward - strike), abs=1.0e-12)


def test_black76_vectorizes_and_handles_degenerate_cases() -> None:
    prices = black76_price(
        100.0,
        np.asarray([90.0, 100.0, 110.0]),
        np.asarray([0.0, 1.0, 1.0]),
        np.asarray([0.2, 0.0, 0.2]),
        option_type="call",
    )
    assert isinstance(prices, np.ndarray)
    assert prices[0] == pytest.approx(10.0)
    assert prices[1] == pytest.approx(0.0)
    assert prices[2] > 0.0


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("volatility", [0.05, 0.25, 1.2, 3.0])
def test_implied_volatility_round_trip(option_type: str, volatility: float) -> None:
    price = black76_price(
        102.0, 93.0, 0.7, volatility, 0.985, option_type  # type: ignore[arg-type]
    )
    recovered = implied_volatility(
        price,
        102.0,
        93.0,
        0.7,
        0.985,
        option_type,  # type: ignore[arg-type]
    )
    assert recovered == pytest.approx(volatility, rel=2.0e-10, abs=2.0e-12)


def test_implied_volatility_enforces_bounds_and_finite_solution() -> None:
    lower, upper = black76_price_bounds(100.0, 120.0, 0.98, "call")
    assert implied_volatility(lower, 100.0, 120.0, 1.0, 0.98, "call") == 0.0
    with pytest.raises(ArbitrageBoundsError, match="outside"):
        implied_volatility(-0.01, 100.0, 120.0, 1.0, 0.98, "call")
    with pytest.raises(ImpliedVolatilityError, match="infinite"):
        implied_volatility(upper, 100.0, 120.0, 1.0, 0.98, "call")


def test_analytic_greeks_match_central_differences() -> None:
    forward = 103.0
    strike = 97.0
    maturity = 0.8
    volatility = 0.31
    discount = 0.975
    greeks = black76_greeks(
        forward, strike, maturity, volatility, discount, "call"
    )

    forward_step = 1.0e-3
    price_up = black76_price(
        forward + forward_step, strike, maturity, volatility, discount, "call"
    )
    price_down = black76_price(
        forward - forward_step, strike, maturity, volatility, discount, "call"
    )
    price_mid = black76_price(
        forward, strike, maturity, volatility, discount, "call"
    )
    numerical_delta = (price_up - price_down) / (2.0 * forward_step)
    numerical_gamma = (price_up - 2.0 * price_mid + price_down) / forward_step**2

    vol_step = 1.0e-5
    vol_up = black76_price(
        forward, strike, maturity, volatility + vol_step, discount, "call"
    )
    vol_down = black76_price(
        forward, strike, maturity, volatility - vol_step, discount, "call"
    )
    numerical_vega = (vol_up - vol_down) / (2.0 * vol_step)

    assert greeks.price == pytest.approx(price_mid)
    assert greeks.delta == pytest.approx(numerical_delta, rel=1.0e-8)
    assert greeks.gamma == pytest.approx(numerical_gamma, rel=5.0e-5)
    assert greeks.vega == pytest.approx(numerical_vega, rel=1.0e-8)
    assert greeks.theta < 0.0


def test_invalid_inputs_raise_explicit_pricing_error() -> None:
    with pytest.raises(PricingError, match="forward"):
        black76_price(0.0, 100.0, 1.0, 0.2)
    with pytest.raises(PricingError, match="maturity"):
        implied_volatility(10.0, 100.0, 100.0, 0.0)
    with pytest.raises(PricingError, match="option_type"):
        black76_price(100.0, 100.0, 1.0, 0.2, option_type="straddle")  # type: ignore[arg-type]
