"""Black--76 pricing utilities for European options.

The module works with forwards rather than spot prices.  Prices and Greeks are
discounted by ``discount_factor``; volatility is expressed in absolute units
(for example, ``0.20`` for 20%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq  # type: ignore[import-untyped]
from scipy.special import ndtr  # type: ignore[import-untyped]

OptionType: TypeAlias = Literal["call", "put"]
FloatArray: TypeAlias = NDArray[np.float64]


class PricingError(ValueError):
    """Base class for invalid inputs to the pricing routines."""


class ArbitrageBoundsError(PricingError):
    """Raised when an option price violates static no-arbitrage bounds."""


class ImpliedVolatilityError(PricingError):
    """Raised when no finite implied volatility can be found."""


@dataclass(frozen=True, slots=True)
class Black76Greeks:
    """Black--76 price sensitivities.

    ``delta`` and ``gamma`` are derivatives with respect to the forward.
    ``dual_delta`` is the derivative with respect to strike.  ``vega`` is per
    unit of volatility (not per volatility point).  ``theta`` is calendar-time
    decay while keeping the forward and discount factor fixed.
    """

    price: float
    delta: float
    dual_delta: float
    gamma: float
    vega: float
    theta: float
    vanna: float
    vomma: float

    @property
    def delta_forward(self) -> float:
        """Alias making the delta convention explicit."""

        return self.delta


def _option_type(option_type: str) -> OptionType:
    normalized = option_type.lower()
    if normalized not in {"call", "put"}:
        raise PricingError("option_type must be either 'call' or 'put'")
    return normalized  # type: ignore[return-value]


def _normal_pdf(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    return np.asarray(
        np.exp(-0.5 * array * array) / np.sqrt(2.0 * np.pi), dtype=np.float64
    )


def _broadcast_inputs(
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    volatility: ArrayLike,
    discount_factor: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    arrays = tuple(
        np.asarray(item, dtype=np.float64)
        for item in (forward, strike, maturity, volatility, discount_factor)
    )
    try:
        forward_array, strike_array, maturity_array, volatility_array, discount_array = (
            np.broadcast_arrays(*arrays)
        )
    except ValueError as exc:
        raise PricingError("pricing inputs cannot be broadcast to a common shape") from exc

    if not all(np.all(np.isfinite(item)) for item in arrays):
        raise PricingError("pricing inputs must all be finite")
    if np.any(forward_array <= 0.0):
        raise PricingError("forward must be strictly positive")
    if np.any(strike_array <= 0.0):
        raise PricingError("strike must be strictly positive")
    if np.any(maturity_array < 0.0):
        raise PricingError("maturity cannot be negative")
    if np.any(volatility_array < 0.0):
        raise PricingError("volatility cannot be negative")
    if np.any(discount_array <= 0.0):
        raise PricingError("discount_factor must be strictly positive")
    return (
        forward_array,
        strike_array,
        maturity_array,
        volatility_array,
        discount_array,
    )


def black76_price(
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    volatility: ArrayLike,
    discount_factor: ArrayLike = 1.0,
    option_type: OptionType = "call",
) -> float | FloatArray:
    """Return a discounted Black--76 European option price.

    Scalars and NumPy-broadcastable arrays are accepted.  At zero maturity or
    zero volatility the function returns discounted intrinsic value exactly.
    """

    kind = _option_type(option_type)
    forward_array, strike_array, maturity_array, volatility_array, discount_array = (
        _broadcast_inputs(forward, strike, maturity, volatility, discount_factor)
    )
    sign = 1.0 if kind == "call" else -1.0
    intrinsic = discount_array * np.maximum(sign * (forward_array - strike_array), 0.0)
    active = (maturity_array > 0.0) & (volatility_array > 0.0)

    result = np.array(intrinsic, dtype=np.float64, copy=True)
    if np.any(active):
        sqrt_maturity = np.sqrt(maturity_array[active])
        total_volatility = volatility_array[active] * sqrt_maturity
        log_moneyness = np.log(forward_array[active] / strike_array[active])
        d1 = log_moneyness / total_volatility + 0.5 * total_volatility
        d2 = d1 - total_volatility
        result[active] = discount_array[active] * sign * (
            forward_array[active] * ndtr(sign * d1)
            - strike_array[active] * ndtr(sign * d2)
        )

    if result.ndim == 0:
        return float(result)
    return result


def black76_price_bounds(
    forward: float,
    strike: float,
    discount_factor: float = 1.0,
    option_type: OptionType = "call",
) -> tuple[float, float]:
    """Return the finite lower bound and limiting upper price bound."""

    kind = _option_type(option_type)
    values = np.asarray([forward, strike, discount_factor], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise PricingError("forward, strike and discount_factor must be finite")
    if forward <= 0.0 or strike <= 0.0 or discount_factor <= 0.0:
        raise PricingError("forward, strike and discount_factor must be positive")
    if kind == "call":
        return (
            discount_factor * max(forward - strike, 0.0),
            discount_factor * forward,
        )
    return (
        discount_factor * max(strike - forward, 0.0),
        discount_factor * strike,
    )


def implied_volatility(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    discount_factor: float = 1.0,
    option_type: OptionType = "call",
    *,
    initial_upper_volatility: float = 1.0,
    maximum_volatility: float = 20.0,
    price_tolerance: float = 1.0e-12,
    xtol: float = 1.0e-12,
    rtol: float = 1.0e-12,
    max_iterations: int = 200,
) -> float:
    """Invert Black--76 with a safeguarded Brent root search.

    A price at intrinsic value maps to zero volatility.  A price at the open
    upper bound has no finite implied volatility and raises
    :class:`ImpliedVolatilityError`.
    """

    kind = _option_type(option_type)
    values = np.asarray(
        [
            price,
            forward,
            strike,
            maturity,
            discount_factor,
            initial_upper_volatility,
            maximum_volatility,
            price_tolerance,
            xtol,
            rtol,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise PricingError("implied-volatility inputs must all be finite")
    if maturity <= 0.0:
        raise PricingError("maturity must be strictly positive when solving for volatility")
    if initial_upper_volatility <= 0.0 or maximum_volatility <= 0.0:
        raise PricingError("volatility brackets must be strictly positive")
    if initial_upper_volatility > maximum_volatility:
        raise PricingError("initial_upper_volatility cannot exceed maximum_volatility")
    if price_tolerance < 0.0 or xtol <= 0.0 or rtol <= 0.0:
        raise PricingError("solver tolerances must be positive")
    if max_iterations <= 0:
        raise PricingError("max_iterations must be positive")

    lower_price, upper_price = black76_price_bounds(
        forward, strike, discount_factor, kind
    )
    scale = max(1.0, abs(lower_price), abs(upper_price))
    tolerance = price_tolerance * scale
    if price < lower_price - tolerance or price > upper_price + tolerance:
        raise ArbitrageBoundsError(
            f"price {price:.12g} is outside Black--76 bounds "
            f"[{lower_price:.12g}, {upper_price:.12g}]"
        )
    if abs(price - lower_price) <= tolerance:
        return 0.0
    if upper_price - price <= tolerance:
        raise ImpliedVolatilityError(
            "the limiting upper-bound price requires infinite volatility"
        )

    def objective(volatility: float) -> float:
        model_price = black76_price(
            forward,
            strike,
            maturity,
            volatility,
            discount_factor,
            kind,
        )
        return float(model_price) - price

    upper_volatility = initial_upper_volatility
    while objective(upper_volatility) < 0.0 and upper_volatility < maximum_volatility:
        upper_volatility = min(2.0 * upper_volatility, maximum_volatility)
    if objective(upper_volatility) < 0.0:
        raise ImpliedVolatilityError(
            "no implied volatility was bracketed below "
            f"maximum_volatility={maximum_volatility:g}"
        )

    try:
        return float(
            brentq(
                objective,
                0.0,
                upper_volatility,
                xtol=xtol,
                rtol=rtol,
                maxiter=max_iterations,
            )
        )
    except (RuntimeError, ValueError) as exc:
        raise ImpliedVolatilityError("Brent's method failed to converge") from exc


def black76_greeks(
    forward: float,
    strike: float,
    maturity: float,
    volatility: float,
    discount_factor: float = 1.0,
    option_type: OptionType = "call",
) -> Black76Greeks:
    """Return analytic Black--76 Greeks for a non-degenerate option."""

    kind = _option_type(option_type)
    inputs = np.asarray(
        [forward, strike, maturity, volatility, discount_factor], dtype=np.float64
    )
    if not np.all(np.isfinite(inputs)):
        raise PricingError("Greek inputs must all be finite")
    if forward <= 0.0 or strike <= 0.0 or discount_factor <= 0.0:
        raise PricingError("forward, strike and discount_factor must be positive")
    if maturity <= 0.0 or volatility <= 0.0:
        raise PricingError("analytic Greeks require positive maturity and volatility")

    sqrt_maturity = np.sqrt(maturity)
    total_volatility = volatility * sqrt_maturity
    d1 = np.log(forward / strike) / total_volatility + 0.5 * total_volatility
    d2 = d1 - total_volatility
    density = float(_normal_pdf(d1))
    if kind == "call":
        delta = discount_factor * float(ndtr(d1))
        dual_delta = -discount_factor * float(ndtr(d2))
    else:
        delta = discount_factor * (float(ndtr(d1)) - 1.0)
        dual_delta = discount_factor * float(ndtr(-d2))

    gamma = discount_factor * density / (forward * total_volatility)
    vega = discount_factor * forward * density * sqrt_maturity
    theta = -discount_factor * forward * density * volatility / (2.0 * sqrt_maturity)
    vanna = -discount_factor * density * d2 / volatility
    vomma = vega * d1 * d2 / volatility
    price = float(
        black76_price(
            forward,
            strike,
            maturity,
            volatility,
            discount_factor,
            kind,
        )
    )
    return Black76Greeks(
        price=price,
        delta=delta,
        dual_delta=dual_delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        vanna=vanna,
        vomma=vomma,
    )


__all__ = [
    "ArbitrageBoundsError",
    "Black76Greeks",
    "ImpliedVolatilityError",
    "OptionType",
    "PricingError",
    "black76_greeks",
    "black76_price",
    "black76_price_bounds",
    "implied_volatility",
]
