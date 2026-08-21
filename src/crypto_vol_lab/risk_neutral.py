"""Risk-neutral densities and model-free variance from option surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid, trapezoid  # type: ignore[import-untyped]
from scipy.interpolate import UnivariateSpline  # type: ignore[import-untyped]

from .pricing import black76_price
from .svi import SSVISurface

FloatArray: TypeAlias = NDArray[np.float64]


class RiskNeutralError(ValueError):
    """Raised when a risk-neutral calculation receives invalid inputs."""


@dataclass(frozen=True, slots=True)
class RiskNeutralDistribution:
    """A normalized finite-grid risk-neutral distribution and its moments."""

    strikes: FloatArray
    density: FloatArray
    cdf: FloatArray
    raw_mass: float
    negative_mass_removed: float
    mean: float
    variance: float
    standard_deviation: float
    skewness: float
    excess_kurtosis: float


def _strictly_increasing_positive(values: ArrayLike, name: str, minimum_size: int) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum_size:
        raise RiskNeutralError(
            f"{name} must be one-dimensional with at least {minimum_size} values"
        )
    if not np.all(np.isfinite(array)):
        raise RiskNeutralError(f"{name} must contain only finite values")
    if np.any(array <= 0.0) or np.any(np.diff(array) <= 0.0):
        raise RiskNeutralError(f"{name} must be strictly positive and increasing")
    return array


def _distribution_moments(
    strikes: FloatArray, density: FloatArray
) -> tuple[float, float, float, float, float]:
    mean = float(trapezoid(strikes * density, strikes))
    centered = strikes - mean
    variance = max(float(trapezoid(centered**2 * density, strikes)), 0.0)
    standard_deviation = np.sqrt(variance)
    if standard_deviation <= np.finfo(np.float64).eps:
        return mean, variance, standard_deviation, 0.0, -3.0
    skewness = float(
        trapezoid((centered / standard_deviation) ** 3 * density, strikes)
    )
    excess_kurtosis = float(
        trapezoid((centered / standard_deviation) ** 4 * density, strikes) - 3.0
    )
    return mean, variance, standard_deviation, skewness, excess_kurtosis


def breeden_litzenberger_density(
    strikes: ArrayLike,
    call_prices: ArrayLike,
    *,
    discount_factor: float = 1.0,
    evaluation_strikes: ArrayLike | None = None,
    smoothing_factor: float = 0.0,
    clip_negative: bool = True,
) -> RiskNeutralDistribution:
    """Extract a risk-neutral density from a smooth call-price curve.

    A cubic smoothing spline represents ``C(K)`` and the raw density is
    ``d2C/dK2 / discount_factor``.  The finite-grid density is normalized and
    its moments therefore refer to the supplied strike interval.  ``raw_mass``
    exposes omitted-tail or smoothing error before normalization.
    """

    strike_array = _strictly_increasing_positive(strikes, "strikes", 5)
    price_array = np.asarray(call_prices, dtype=np.float64)
    if price_array.ndim != 1 or price_array.shape != strike_array.shape:
        raise RiskNeutralError("call_prices must have the same one-dimensional shape as strikes")
    if not np.all(np.isfinite(price_array)) or np.any(price_array < 0.0):
        raise RiskNeutralError("call_prices must be finite and non-negative")
    if discount_factor <= 0.0 or not np.isfinite(discount_factor):
        raise RiskNeutralError("discount_factor must be finite and strictly positive")
    if smoothing_factor < 0.0 or not np.isfinite(smoothing_factor):
        raise RiskNeutralError("smoothing_factor must be finite and non-negative")

    if evaluation_strikes is None:
        evaluation = strike_array.copy()
    else:
        evaluation = _strictly_increasing_positive(
            evaluation_strikes, "evaluation_strikes", 5
        )
        if evaluation[0] < strike_array[0] or evaluation[-1] > strike_array[-1]:
            raise RiskNeutralError(
                "evaluation_strikes must remain inside the observed strike interval"
            )

    spline = UnivariateSpline(
        strike_array,
        price_array,
        k=3,
        s=smoothing_factor,
        ext=2,
    )
    raw_density = np.asarray(spline.derivative(n=2)(evaluation), dtype=np.float64)
    raw_density /= discount_factor
    if not np.all(np.isfinite(raw_density)):
        raise RiskNeutralError("the smoothed second derivative is not finite")

    negative_component = np.minimum(raw_density, 0.0)
    negative_mass_removed = float(-trapezoid(negative_component, evaluation))
    density = np.maximum(raw_density, 0.0) if clip_negative else raw_density.copy()
    raw_mass = float(trapezoid(density, evaluation))
    if raw_mass <= np.finfo(np.float64).eps:
        raise RiskNeutralError("the extracted density has non-positive mass")
    density /= raw_mass
    cdf = np.asarray(cumulative_trapezoid(density, evaluation, initial=0.0))
    if cdf[-1] <= 0.0:
        raise RiskNeutralError("the extracted cumulative density has non-positive mass")
    cdf /= cdf[-1]
    cdf[-1] = 1.0
    mean, variance, standard_deviation, skewness, excess_kurtosis = (
        _distribution_moments(evaluation, density)
    )
    return RiskNeutralDistribution(
        strikes=evaluation.copy(),
        density=density,
        cdf=cdf,
        raw_mass=raw_mass,
        negative_mass_removed=negative_mass_removed,
        mean=mean,
        variance=variance,
        standard_deviation=standard_deviation,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )


def density_from_ssvi(
    surface: SSVISurface,
    maturity: float,
    forward: float,
    strikes: ArrayLike,
    *,
    discount_factor: float = 1.0,
    evaluation_strikes: ArrayLike | None = None,
    smoothing_factor: float = 0.0,
    clip_negative: bool = True,
) -> RiskNeutralDistribution:
    """Price a smooth SSVI call curve and apply Breeden--Litzenberger."""

    if maturity <= 0.0 or not np.isfinite(maturity):
        raise RiskNeutralError("maturity must be finite and strictly positive")
    if forward <= 0.0 or not np.isfinite(forward):
        raise RiskNeutralError("forward must be finite and strictly positive")
    strike_array = _strictly_increasing_positive(strikes, "strikes", 5)
    log_moneyness = np.log(strike_array / forward)
    variance = np.asarray(
        surface.total_variance(log_moneyness, maturity), dtype=np.float64
    )
    volatility = np.sqrt(variance / maturity)
    call_prices = np.asarray(
        black76_price(
            forward,
            strike_array,
            maturity,
            volatility,
            discount_factor,
            "call",
        ),
        dtype=np.float64,
    )
    return breeden_litzenberger_density(
        strike_array,
        call_prices,
        discount_factor=discount_factor,
        evaluation_strikes=evaluation_strikes,
        smoothing_factor=smoothing_factor,
        clip_negative=clip_negative,
    )


def model_free_variance(
    strikes: ArrayLike,
    otm_option_prices: ArrayLike,
    *,
    forward: float,
    maturity: float,
    discount_factor: float = 1.0,
) -> float:
    """Integrate the continuous model-free variance-swap formula.

    ``otm_option_prices`` must contain discounted puts below the forward and
    discounted calls above it.  At exactly the forward, the average of put and
    call is conventional.  Integration is trapezoidal on the supplied grid;
    callers should include sufficiently deep wings and report truncation tests.
    """

    strike_array = _strictly_increasing_positive(strikes, "strikes", 3)
    price_array = np.asarray(otm_option_prices, dtype=np.float64)
    if price_array.ndim != 1 or price_array.shape != strike_array.shape:
        raise RiskNeutralError(
            "otm_option_prices must have the same one-dimensional shape as strikes"
        )
    if not np.all(np.isfinite(price_array)) or np.any(price_array < 0.0):
        raise RiskNeutralError("otm_option_prices must be finite and non-negative")
    if forward <= strike_array[0] or forward >= strike_array[-1] or not np.isfinite(forward):
        raise RiskNeutralError("forward must lie strictly inside the strike interval")
    if maturity <= 0.0 or not np.isfinite(maturity):
        raise RiskNeutralError("maturity must be finite and strictly positive")
    if discount_factor <= 0.0 or not np.isfinite(discount_factor):
        raise RiskNeutralError("discount_factor must be finite and strictly positive")
    integral = float(
        trapezoid(price_array / (discount_factor * strike_array**2), strike_array)
    )
    variance = 2.0 * integral / maturity
    if variance < -1.0e-14:
        raise RiskNeutralError("integrated model-free variance is negative")
    return max(variance, 0.0)


def model_free_variance_from_chain(
    strikes: ArrayLike,
    call_prices: ArrayLike,
    put_prices: ArrayLike,
    *,
    forward: float,
    maturity: float,
    discount_factor: float = 1.0,
) -> float:
    """Select OTM quotes from call/put arrays and integrate model-free variance."""

    strike_array = _strictly_increasing_positive(strikes, "strikes", 3)
    calls = np.asarray(call_prices, dtype=np.float64)
    puts = np.asarray(put_prices, dtype=np.float64)
    if calls.shape != strike_array.shape or puts.shape != strike_array.shape:
        raise RiskNeutralError("call_prices and put_prices must match the strike grid")
    if (
        not np.all(np.isfinite(calls))
        or not np.all(np.isfinite(puts))
        or np.any(calls < 0.0)
        or np.any(puts < 0.0)
    ):
        raise RiskNeutralError("option prices must be finite and non-negative")
    is_forward = np.isclose(
        strike_array,
        forward,
        rtol=float(8.0 * np.finfo(np.float64).eps),
        atol=0.0,
    )
    otm_prices = np.where(
        strike_array < forward,
        puts,
        np.where(is_forward, 0.5 * (calls + puts), calls),
    )
    return model_free_variance(
        strike_array,
        otm_prices,
        forward=forward,
        maturity=maturity,
        discount_factor=discount_factor,
    )


__all__ = [
    "RiskNeutralDistribution",
    "RiskNeutralError",
    "breeden_litzenberger_density",
    "density_from_ssvi",
    "model_free_variance",
    "model_free_variance_from_chain",
]
