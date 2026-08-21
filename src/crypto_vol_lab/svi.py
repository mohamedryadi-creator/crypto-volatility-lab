"""SVI/SSVI volatility surfaces and static-arbitrage diagnostics.

Log-moneyness is defined as ``k = log(K / F)`` and every surface function
returns *total variance* ``w = sigma_implied**2 * maturity``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]
from scipy.optimize import least_squares  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

FloatArray: TypeAlias = NDArray[np.float64]
RobustLoss: TypeAlias = Literal["linear", "soft_l1", "huber", "cauchy", "arctan"]
_PARAMETER_EPSILON = 1.0e-10


class SVIError(ValueError):
    """Base class for invalid SVI inputs."""


class NoArbitrageError(SVIError):
    """Raised when a requested surface violates an enforced condition."""


@dataclass(frozen=True, slots=True)
class RawSVIParams:
    """Raw-SVI parameters for ``a + b[rho(k-m) + sqrt((k-m)^2+sigma^2)]``."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.a, self.b, self.rho, self.m, self.sigma], dtype=np.float64
        )
        if not np.all(np.isfinite(values)):
            raise SVIError("raw-SVI parameters must all be finite")
        if self.b < 0.0:
            raise SVIError("raw-SVI b must be non-negative")
        if self.sigma <= 0.0:
            raise SVIError("raw-SVI sigma must be strictly positive")
        if abs(self.rho) >= 1.0:
            raise SVIError("raw-SVI rho must satisfy |rho| < 1")
        if self.minimum_total_variance < -1.0e-12:
            raise SVIError("raw-SVI total variance must be non-negative for every strike")

    @property
    def minimum_total_variance(self) -> float:
        """Analytic minimum of the raw-SVI smile over log-moneyness."""

        return float(
            self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho * self.rho)
        )

    def as_array(self) -> FloatArray:
        """Return parameters ordered as ``(a, b, rho, m, sigma)``."""

        return np.asarray(
            [self.a, self.b, self.rho, self.m, self.sigma], dtype=np.float64
        )


@dataclass(frozen=True, slots=True)
class SVIFitResult:
    """Result and diagnostics from a raw-SVI least-squares calibration."""

    params: RawSVIParams
    success: bool
    cost: float
    rmse: float
    weighted_rmse: float
    inside_spread_fraction: float | None
    optimality: float
    nfev: int
    message: str


@dataclass(frozen=True, slots=True)
class SSVIParams:
    """Power-law SSVI parameters.

    ``phi(theta) = eta / (theta**gamma * (1+theta)**(1-gamma))``.
    Restricting ``gamma`` to ``[0, 1]`` gives the required monotonicity of
    ``theta * phi(theta)`` for calendar-spread consistency.
    """

    rho: float
    eta: float
    gamma: float

    def __post_init__(self) -> None:
        values = np.asarray([self.rho, self.eta, self.gamma], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise SVIError("SSVI parameters must all be finite")
        if abs(self.rho) >= 1.0:
            raise SVIError("SSVI rho must satisfy |rho| < 1")
        if self.eta <= 0.0:
            raise SVIError("SSVI eta must be strictly positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise SVIError("SSVI gamma must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class SSVINoArbitrageReport:
    """Pointwise sufficient-condition report over a theta grid."""

    theta: FloatArray
    linear_condition: FloatArray
    quadratic_condition: FloatArray
    minimum_linear_margin: float
    minimum_quadratic_margin: float
    passed: bool


@dataclass(frozen=True, slots=True)
class SSVIFitResult:
    """Result of a global power-law SSVI calibration."""

    surface: SSVISurface
    params: SSVIParams
    success: bool
    cost: float
    rmse: float
    weighted_rmse: float
    optimality: float
    nfev: int
    message: str
    no_arbitrage_report: SSVINoArbitrageReport


@dataclass(frozen=True, slots=True)
class ButterflyDiagnostic:
    """Durrleman ``g(k)`` butterfly-arbitrage diagnostic."""

    log_moneyness: FloatArray
    g: FloatArray
    minimum_g: float
    violation_count: int
    passed: bool


@dataclass(frozen=True, slots=True)
class CalendarArbitrageDiagnostic:
    """Total-variance increments along a maturity grid."""

    maturities: FloatArray
    log_moneyness: FloatArray
    total_variance: FloatArray
    increments: FloatArray
    minimum_increment: float
    violation_count: int
    passed: bool


@dataclass(frozen=True, slots=True)
class PriceArbitrageDiagnostic:
    """Monotonicity, vertical-spread and convexity checks across strikes."""

    strikes: FloatArray
    prices: FloatArray
    slopes: FloatArray
    slope_increments: FloatArray
    monotonicity_violations: int
    vertical_spread_violations: int
    convexity_violations: int
    negative_price_violations: int
    minimum_convexity_margin: float
    passed: bool


def _one_dimensional(values: ArrayLike, name: str, *, minimum_size: int = 1) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum_size:
        raise SVIError(f"{name} must be one-dimensional with at least {minimum_size} values")
    if not np.all(np.isfinite(array)):
        raise SVIError(f"{name} must contain only finite values")
    return array


def raw_svi_total_variance(log_moneyness: ArrayLike, params: RawSVIParams) -> float | FloatArray:
    """Evaluate a raw-SVI total-variance smile."""

    k = np.asarray(log_moneyness, dtype=np.float64)
    if not np.all(np.isfinite(k)):
        raise SVIError("log_moneyness must contain only finite values")
    centered = k - params.m
    result = params.a + params.b * (
        params.rho * centered + np.sqrt(centered * centered + params.sigma**2)
    )
    if result.ndim == 0:
        return float(result)
    return np.asarray(result, dtype=np.float64)


def raw_svi_derivatives(
    log_moneyness: ArrayLike, params: RawSVIParams
) -> tuple[float | FloatArray, float | FloatArray, float | FloatArray]:
    """Return ``(w, dw/dk, d2w/dk2)`` analytically."""

    k = np.asarray(log_moneyness, dtype=np.float64)
    if not np.all(np.isfinite(k)):
        raise SVIError("log_moneyness must contain only finite values")
    centered = k - params.m
    radius = np.sqrt(centered * centered + params.sigma**2)
    variance = params.a + params.b * (params.rho * centered + radius)
    first = params.b * (params.rho + centered / radius)
    second = params.b * params.sigma**2 / radius**3
    if variance.ndim == 0:
        return float(variance), float(first), float(second)
    return variance, first, second


def raw_svi_jacobian(log_moneyness: ArrayLike, params: RawSVIParams) -> FloatArray:
    """Analytic Jacobian with columns ``(a, b, rho, m, sigma)``."""

    k = np.atleast_1d(np.asarray(log_moneyness, dtype=np.float64))
    if k.ndim != 1 or not np.all(np.isfinite(k)):
        raise SVIError("log_moneyness must be a finite scalar or one-dimensional array")
    centered = k - params.m
    radius = np.sqrt(centered * centered + params.sigma**2)
    return np.column_stack(
        (
            np.ones_like(k),
            params.rho * centered + radius,
            params.b * centered,
            params.b * (-params.rho - centered / radius),
            params.b * params.sigma / radius,
        )
    )


def _softplus(value: ArrayLike) -> FloatArray:
    return np.logaddexp(0.0, np.asarray(value, dtype=np.float64))


def _inverse_softplus(value: float) -> float:
    if value <= 0.0 or not np.isfinite(value):
        raise SVIError("inverse-softplus input must be finite and positive")
    if value > 30.0:
        return value
    return float(value + np.log(-np.expm1(-value)))


def raw_svi_from_unconstrained(values: ArrayLike) -> RawSVIParams:
    """Map five unconstrained reals to an everywhere-positive raw-SVI smile."""

    unconstrained = np.asarray(values, dtype=np.float64)
    if unconstrained.shape != (5,) or not np.all(np.isfinite(unconstrained)):
        raise SVIError("unconstrained raw-SVI parameters must be five finite values")
    minimum_variance = float(_softplus(unconstrained[0])) + _PARAMETER_EPSILON
    b = float(_softplus(unconstrained[1]))
    rho = (1.0 - _PARAMETER_EPSILON) * float(np.tanh(unconstrained[2]))
    m = float(unconstrained[3])
    sigma = float(_softplus(unconstrained[4])) + _PARAMETER_EPSILON
    a = minimum_variance - b * sigma * np.sqrt(1.0 - rho * rho)
    return RawSVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def raw_svi_to_unconstrained(params: RawSVIParams) -> FloatArray:
    """Inverse of :func:`raw_svi_from_unconstrained`, up to numerical floors."""

    minimum_variance = max(params.minimum_total_variance, _PARAMETER_EPSILON)
    b = max(params.b, _PARAMETER_EPSILON)
    sigma = max(params.sigma - _PARAMETER_EPSILON, _PARAMETER_EPSILON)
    scaled_rho = np.clip(
        params.rho / (1.0 - _PARAMETER_EPSILON),
        -1.0 + _PARAMETER_EPSILON,
        1.0 - _PARAMETER_EPSILON,
    )
    return np.asarray(
        [
            _inverse_softplus(minimum_variance - _PARAMETER_EPSILON + _PARAMETER_EPSILON**2),
            _inverse_softplus(b),
            np.arctanh(scaled_rho),
            params.m,
            _inverse_softplus(sigma),
        ],
        dtype=np.float64,
    )


def _parameter_transform_jacobian(unconstrained: FloatArray) -> FloatArray:
    params = raw_svi_from_unconstrained(unconstrained)
    derivative_minimum = float(expit(unconstrained[0]))
    derivative_b = float(expit(unconstrained[1]))
    derivative_rho = (1.0 - _PARAMETER_EPSILON) * (
        1.0 - float(np.tanh(unconstrained[2])) ** 2
    )
    derivative_sigma = float(expit(unconstrained[4]))
    root = np.sqrt(1.0 - params.rho * params.rho)

    jacobian = np.zeros((5, 5), dtype=np.float64)
    jacobian[0, 0] = derivative_minimum
    jacobian[0, 1] = -derivative_b * params.sigma * root
    jacobian[0, 2] = (
        params.b * params.sigma * params.rho / root * derivative_rho
    )
    jacobian[0, 4] = -params.b * root * derivative_sigma
    jacobian[1, 1] = derivative_b
    jacobian[2, 2] = derivative_rho
    jacobian[3, 3] = 1.0
    jacobian[4, 4] = derivative_sigma
    return jacobian


def svi_calibration_weights(
    total_variance_mid: ArrayLike,
    *,
    total_variance_bid: ArrayLike | None = None,
    total_variance_ask: ArrayLike | None = None,
    vega: ArrayLike | None = None,
    spread_floor: float = 1.0e-5,
) -> FloatArray:
    """Build stable inverse-spread, vega-aware least-squares multipliers.

    The returned multiplier is ``sqrt(normalized_vega) / half_spread`` and is
    itself median-normalized.  This makes the objective unitless while giving
    more influence to liquid, economically material quotes.
    """

    mid = _one_dimensional(total_variance_mid, "total_variance_mid")
    if np.any(mid <= 0.0):
        raise SVIError("total_variance_mid must be strictly positive")
    if spread_floor <= 0.0 or not np.isfinite(spread_floor):
        raise SVIError("spread_floor must be finite and strictly positive")

    if (total_variance_bid is None) != (total_variance_ask is None):
        raise SVIError("total_variance_bid and total_variance_ask must be supplied together")
    if total_variance_bid is None:
        half_spread = np.ones_like(mid)
    else:
        assert total_variance_ask is not None
        bid = _one_dimensional(total_variance_bid, "total_variance_bid")
        ask = _one_dimensional(total_variance_ask, "total_variance_ask")
        if bid.shape != mid.shape or ask.shape != mid.shape:
            raise SVIError("bid, mid and ask arrays must have identical shapes")
        if np.any(bid > mid) or np.any(mid > ask):
            raise SVIError("quotes must satisfy bid <= mid <= ask")
        half_spread = np.maximum(0.5 * (ask - bid), spread_floor)

    if vega is None:
        vega_multiplier = np.ones_like(mid)
    else:
        vega_array = _one_dimensional(vega, "vega")
        if vega_array.shape != mid.shape:
            raise SVIError("vega and total_variance_mid must have identical shapes")
        if np.any(vega_array < 0.0):
            raise SVIError("vega cannot be negative")
        positive = vega_array[vega_array > 0.0]
        if positive.size == 0:
            raise SVIError("at least one vega must be strictly positive")
        vega_multiplier = np.sqrt(vega_array / np.median(positive))

    weights = vega_multiplier / half_spread
    positive_weights = weights[weights > 0.0]
    if positive_weights.size == 0:
        raise SVIError("calibration weights are all zero")
    return np.asarray(weights / np.median(positive_weights), dtype=np.float64)


def _default_initial_params(k: FloatArray, variance: FloatArray) -> RawSVIParams:
    span = max(float(np.ptp(k)), 0.1)
    slope = float(np.polyfit(k, variance, 1)[0]) if k.size > 1 else 0.0
    spread = max(float(np.percentile(variance, 90) - np.percentile(variance, 10)), 1.0e-4)
    b = max(spread / span, 1.0e-3)
    rho = float(np.clip(slope / b, -0.8, 0.8))
    m = float(k[np.argmin(variance)])
    sigma = max(0.2 * span, 0.05)
    minimum_variance = max(0.8 * float(np.min(variance)), 1.0e-5)
    a = minimum_variance - b * sigma * np.sqrt(1.0 - rho * rho)
    return RawSVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def fit_raw_svi(
    log_moneyness: ArrayLike,
    total_variance_mid: ArrayLike,
    *,
    total_variance_bid: ArrayLike | None = None,
    total_variance_ask: ArrayLike | None = None,
    vega: ArrayLike | None = None,
    initial_params: RawSVIParams | None = None,
    spread_floor: float = 1.0e-5,
    loss: RobustLoss = "soft_l1",
    f_scale: float = 1.0,
    midpoint_penalty: float = 1.0e-4,
    max_nfev: int = 5_000,
) -> SVIFitResult:
    """Calibrate raw SVI with transformed parameters and analytic Jacobian.

    When bid and ask are supplied, the primary residual is the signed distance
    to that interval (zero inside it).  A small midpoint residual identifies a
    unique solution when several curves lie fully within the quoted spreads.
    """

    k = _one_dimensional(log_moneyness, "log_moneyness", minimum_size=5)
    variance = _one_dimensional(total_variance_mid, "total_variance_mid", minimum_size=5)
    if k.shape != variance.shape:
        raise SVIError("log_moneyness and total_variance_mid must have identical shapes")
    if np.unique(k).size < 5:
        raise SVIError("at least five distinct log-moneyness values are required")
    if np.any(variance <= 0.0):
        raise SVIError("total_variance_mid must be strictly positive")
    if loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
        raise SVIError("unsupported robust loss")
    if f_scale <= 0.0 or not np.isfinite(f_scale):
        raise SVIError("f_scale must be finite and strictly positive")
    if midpoint_penalty < 0.0 or not np.isfinite(midpoint_penalty):
        raise SVIError("midpoint_penalty must be finite and non-negative")
    if max_nfev <= 0:
        raise SVIError("max_nfev must be positive")

    weights = svi_calibration_weights(
        variance,
        total_variance_bid=total_variance_bid,
        total_variance_ask=total_variance_ask,
        vega=vega,
        spread_floor=spread_floor,
    )
    starting_params = initial_params or _default_initial_params(k, variance)
    starting_point = raw_svi_to_unconstrained(starting_params)
    has_quotes = total_variance_bid is not None
    bid_array = (
        np.asarray(total_variance_bid, dtype=np.float64) if has_quotes else None
    )
    ask_array = (
        np.asarray(total_variance_ask, dtype=np.float64) if has_quotes else None
    )

    def quote_distance(model: FloatArray) -> FloatArray:
        if bid_array is None or ask_array is None:
            return model - variance
        return np.where(
            model < bid_array,
            model - bid_array,
            np.where(model > ask_array, model - ask_array, 0.0),
        )

    def quote_derivative(model: FloatArray) -> FloatArray:
        if bid_array is None or ask_array is None:
            return np.ones_like(model)
        return ((model < bid_array) | (model > ask_array)).astype(np.float64)

    def residuals(unconstrained: FloatArray) -> FloatArray:
        params = raw_svi_from_unconstrained(unconstrained)
        model = np.asarray(raw_svi_total_variance(k, params), dtype=np.float64)
        primary = weights * quote_distance(model)
        if not has_quotes or midpoint_penalty == 0.0:
            return primary
        midpoint = np.sqrt(midpoint_penalty) * weights * (model - variance)
        return np.concatenate((primary, midpoint))

    def jacobian(unconstrained: FloatArray) -> FloatArray:
        params = raw_svi_from_unconstrained(unconstrained)
        model = np.asarray(raw_svi_total_variance(k, params), dtype=np.float64)
        physical = raw_svi_jacobian(k, params)
        transformed = physical @ _parameter_transform_jacobian(unconstrained)
        primary = (
            weights * quote_derivative(model)
        )[:, np.newaxis] * transformed
        if not has_quotes or midpoint_penalty == 0.0:
            return primary
        midpoint = (
            np.sqrt(midpoint_penalty) * weights[:, np.newaxis] * transformed
        )
        return np.vstack((primary, midpoint))

    optimization = least_squares(
        residuals,
        starting_point,
        jac=jacobian,
        method="trf",
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        x_scale="jac",
    )
    params = raw_svi_from_unconstrained(optimization.x)
    model = np.asarray(raw_svi_total_variance(k, params), dtype=np.float64)
    errors = model - variance
    if bid_array is None or ask_array is None:
        inside_spread_fraction = None
    else:
        inside_spread_fraction = float(
            np.mean((model >= bid_array) & (model <= ask_array))
        )
    return SVIFitResult(
        params=params,
        success=bool(optimization.success),
        cost=float(optimization.cost),
        rmse=float(np.sqrt(np.mean(errors * errors))),
        weighted_rmse=float(np.sqrt(np.mean((weights * errors) ** 2))),
        inside_spread_fraction=inside_spread_fraction,
        optimality=float(optimization.optimality),
        nfev=int(optimization.nfev),
        message=str(optimization.message),
    )


def ssvi_phi(atm_total_variance: ArrayLike, params: SSVIParams) -> float | FloatArray:
    """Evaluate the modified power-law SSVI ``phi(theta)``."""

    theta = np.asarray(atm_total_variance, dtype=np.float64)
    if not np.all(np.isfinite(theta)) or np.any(theta <= 0.0):
        raise SVIError("atm_total_variance must be finite and strictly positive")
    result = params.eta / (
        theta**params.gamma * (1.0 + theta) ** (1.0 - params.gamma)
    )
    if result.ndim == 0:
        return float(result)
    return result


def ssvi_total_variance(
    log_moneyness: ArrayLike,
    atm_total_variance: ArrayLike,
    params: SSVIParams,
) -> float | FloatArray:
    """Evaluate the SSVI total-variance formula."""

    k = np.asarray(log_moneyness, dtype=np.float64)
    theta = np.asarray(atm_total_variance, dtype=np.float64)
    try:
        k, theta = np.broadcast_arrays(k, theta)
    except ValueError as exc:
        raise SVIError("log_moneyness and atm_total_variance cannot be broadcast") from exc
    if not np.all(np.isfinite(k)):
        raise SVIError("log_moneyness must contain only finite values")
    if not np.all(np.isfinite(theta)) or np.any(theta < 0.0):
        raise SVIError("atm_total_variance must be finite and non-negative")

    result = np.zeros_like(theta, dtype=np.float64)
    positive = theta > 0.0
    if np.any(positive):
        phi = np.asarray(ssvi_phi(theta[positive], params), dtype=np.float64)
        scaled = phi * k[positive]
        root = np.sqrt(
            (scaled + params.rho) ** 2 + 1.0 - params.rho * params.rho
        )
        result[positive] = 0.5 * theta[positive] * (
            1.0 + params.rho * scaled + root
        )
    if result.ndim == 0:
        return float(result)
    return result


def ssvi_sufficient_no_arbitrage(
    atm_total_variance: ArrayLike,
    params: SSVIParams,
    *,
    tolerance: float = 1.0e-12,
) -> SSVINoArbitrageReport:
    """Check standard sufficient SSVI butterfly-arbitrage inequalities.

    The checked quantities are ``theta*phi*(1+|rho|) < 4`` and
    ``theta*phi**2*(1+|rho|) <= 4``.  Together with monotone ``theta`` and the
    power-law parameter restrictions, these are practical sufficient checks.
    """

    theta = _one_dimensional(atm_total_variance, "atm_total_variance")
    if np.any(theta <= 0.0):
        raise SVIError("atm_total_variance must be strictly positive")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise SVIError("tolerance must be finite and non-negative")
    phi = np.asarray(ssvi_phi(theta, params), dtype=np.float64)
    rho_factor = 1.0 + abs(params.rho)
    linear = theta * phi * rho_factor
    quadratic = theta * phi * phi * rho_factor
    linear_margin = float(np.min(4.0 - linear))
    quadratic_margin = float(np.min(4.0 - quadratic))
    passed = bool(linear_margin > -tolerance and quadratic_margin >= -tolerance)
    return SSVINoArbitrageReport(
        theta=theta.copy(),
        linear_condition=linear,
        quadratic_condition=quadratic,
        minimum_linear_margin=linear_margin,
        minimum_quadratic_margin=quadratic_margin,
        passed=passed,
    )


class MonotoneThetaCurve:
    """Shape-preserving interpolation of ATM total variance through maturity."""

    def __init__(self, maturities: ArrayLike, atm_total_variances: ArrayLike) -> None:
        maturity_array = _one_dimensional(maturities, "maturities", minimum_size=2)
        theta_array = _one_dimensional(
            atm_total_variances, "atm_total_variances", minimum_size=2
        )
        if maturity_array.shape != theta_array.shape:
            raise SVIError("maturities and atm_total_variances must have identical shapes")
        if np.any(maturity_array <= 0.0) or np.any(np.diff(maturity_array) <= 0.0):
            raise SVIError("maturities must be strictly positive and increasing")
        if np.any(theta_array <= 0.0) or np.any(np.diff(theta_array) < 0.0):
            raise SVIError("atm_total_variances must be positive and non-decreasing")
        self._maturities = maturity_array.copy()
        self._theta = theta_array.copy()
        self._interpolator = PchipInterpolator(
            self._maturities, self._theta, extrapolate=False
        )

    @property
    def maturities(self) -> FloatArray:
        return self._maturities.copy()

    @property
    def atm_total_variances(self) -> FloatArray:
        return self._theta.copy()

    def __call__(self, maturity: ArrayLike) -> float | FloatArray:
        values = np.asarray(maturity, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise SVIError("maturity must be finite and non-negative")
        flat = values.reshape(-1)
        output = np.empty_like(flat)
        first_time = self._maturities[0]
        last_time = self._maturities[-1]
        below = flat < first_time
        above = flat > last_time
        within = ~(below | above)
        output[below] = self._theta[0] * flat[below] / first_time
        if np.any(within):
            output[within] = self._interpolator(flat[within])
        final_slope = max(
            (self._theta[-1] - self._theta[-2])
            / (self._maturities[-1] - self._maturities[-2]),
            0.0,
        )
        output[above] = self._theta[-1] + final_slope * (flat[above] - last_time)
        reshaped = output.reshape(values.shape)
        if reshaped.ndim == 0:
            return float(reshaped)
        return reshaped


class SSVISurface:
    """A power-law SSVI surface driven by a monotone ATM variance curve."""

    def __init__(
        self,
        maturities: ArrayLike,
        atm_total_variances: ArrayLike,
        params: SSVIParams,
        *,
        enforce_sufficient_no_arbitrage: bool = True,
    ) -> None:
        self.theta_curve = MonotoneThetaCurve(maturities, atm_total_variances)
        self.params = params
        dense_theta = np.linspace(
            self.theta_curve.atm_total_variances[0],
            self.theta_curve.atm_total_variances[-1],
            512,
            dtype=np.float64,
        )
        report = ssvi_sufficient_no_arbitrage(dense_theta, params)
        if enforce_sufficient_no_arbitrage and not report.passed:
            raise NoArbitrageError(
                "SSVI parameters fail sufficient no-butterfly-arbitrage conditions"
            )
        self.no_arbitrage_report = report

    @property
    def maturities(self) -> FloatArray:
        return self.theta_curve.maturities

    @property
    def atm_total_variances(self) -> FloatArray:
        return self.theta_curve.atm_total_variances

    def theta(self, maturity: ArrayLike) -> float | FloatArray:
        return self.theta_curve(maturity)

    def total_variance(
        self, log_moneyness: ArrayLike, maturity: ArrayLike
    ) -> float | FloatArray:
        theta = self.theta(maturity)
        return ssvi_total_variance(log_moneyness, theta, self.params)

    def implied_volatility(
        self, log_moneyness: ArrayLike, maturity: ArrayLike
    ) -> float | FloatArray:
        maturity_array = np.asarray(maturity, dtype=np.float64)
        variance = np.asarray(
            self.total_variance(log_moneyness, maturity_array), dtype=np.float64
        )
        try:
            maturity_broadcast = np.broadcast_to(maturity_array, variance.shape)
        except ValueError as exc:
            raise SVIError("maturity cannot be broadcast to the surface output") from exc
        if np.any(maturity_broadcast <= 0.0):
            raise SVIError("positive maturity is required for implied volatility")
        result = np.sqrt(variance / maturity_broadcast)
        if result.ndim == 0:
            return float(result)
        return np.asarray(result, dtype=np.float64)


def _ssvi_from_unconstrained(values: ArrayLike) -> SSVIParams:
    unconstrained = np.asarray(values, dtype=np.float64)
    if unconstrained.shape != (3,) or not np.all(np.isfinite(unconstrained)):
        raise SVIError("unconstrained SSVI parameters must be three finite values")
    rho = (1.0 - _PARAMETER_EPSILON) * float(np.tanh(unconstrained[0]))
    eta = float(_softplus(unconstrained[1])) + _PARAMETER_EPSILON
    gamma = float(expit(unconstrained[2]))
    return SSVIParams(rho=rho, eta=eta, gamma=gamma)


def _ssvi_to_unconstrained(params: SSVIParams) -> FloatArray:
    scaled_rho = np.clip(
        params.rho / (1.0 - _PARAMETER_EPSILON),
        -1.0 + _PARAMETER_EPSILON,
        1.0 - _PARAMETER_EPSILON,
    )
    eta_target = max(params.eta - _PARAMETER_EPSILON, _PARAMETER_EPSILON)
    clipped_gamma = float(
        np.clip(params.gamma, _PARAMETER_EPSILON, 1.0 - _PARAMETER_EPSILON)
    )
    return np.asarray(
        [
            np.arctanh(scaled_rho),
            _inverse_softplus(eta_target),
            np.log(clipped_gamma / (1.0 - clipped_gamma)),
        ],
        dtype=np.float64,
    )


def fit_ssvi_surface(
    maturities: ArrayLike,
    log_moneyness: ArrayLike,
    observed_total_variance: ArrayLike,
    atm_maturities: ArrayLike,
    atm_total_variances: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    initial_params: SSVIParams | None = None,
    loss: RobustLoss = "soft_l1",
    f_scale: float = 1.0,
    no_arbitrage_penalty: float = 10_000.0,
    enforce_sufficient_no_arbitrage: bool = True,
    max_nfev: int = 5_000,
) -> SSVIFitResult:
    """Fit one global power-law SSVI surface across strikes and maturities.

    Observation weights enter as conventional non-negative least-squares
    weights.  The optimizer uses unconstrained ``rho``, ``eta`` and ``gamma``
    coordinates, augments the residual vector with violations of the two
    sufficient SSVI inequalities, and hard-rejects a violating final surface
    when ``enforce_sufficient_no_arbitrage`` is true.
    """

    observation_maturities = _one_dimensional(
        maturities, "maturities", minimum_size=6
    )
    k = _one_dimensional(log_moneyness, "log_moneyness", minimum_size=6)
    observed = _one_dimensional(
        observed_total_variance, "observed_total_variance", minimum_size=6
    )
    if observation_maturities.shape != k.shape or k.shape != observed.shape:
        raise SVIError(
            "maturities, log_moneyness and observed_total_variance must match"
        )
    if np.any(observation_maturities <= 0.0):
        raise SVIError("observation maturities must be strictly positive")
    if np.any(observed <= 0.0):
        raise SVIError("observed_total_variance must be strictly positive")
    if loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
        raise SVIError("unsupported robust loss")
    if f_scale <= 0.0 or not np.isfinite(f_scale):
        raise SVIError("f_scale must be finite and strictly positive")
    if no_arbitrage_penalty <= 0.0 or not np.isfinite(no_arbitrage_penalty):
        raise SVIError("no_arbitrage_penalty must be finite and strictly positive")
    if max_nfev <= 0:
        raise SVIError("max_nfev must be positive")

    theta_curve = MonotoneThetaCurve(atm_maturities, atm_total_variances)
    observation_theta = np.asarray(
        theta_curve(observation_maturities), dtype=np.float64
    )
    theta_min = min(
        float(np.min(observation_theta)),
        float(np.min(theta_curve.atm_total_variances)),
    )
    theta_max = max(
        float(np.max(observation_theta)),
        float(np.max(theta_curve.atm_total_variances)),
    )
    theta_audit = np.linspace(theta_min, theta_max, 256, dtype=np.float64)

    if weights is None:
        residual_multiplier = np.ones_like(observed)
    else:
        weight_array = _one_dimensional(weights, "weights", minimum_size=6)
        if weight_array.shape != observed.shape:
            raise SVIError("weights must match observed_total_variance")
        if np.any(weight_array < 0.0):
            raise SVIError("weights cannot be negative")
        positive_weights = weight_array[weight_array > 0.0]
        if positive_weights.size == 0:
            raise SVIError("at least one weight must be strictly positive")
        residual_multiplier = np.sqrt(weight_array / np.median(positive_weights))

    starting_params = initial_params or SSVIParams(rho=-0.2, eta=0.3, gamma=0.5)
    starting_point = _ssvi_to_unconstrained(starting_params)
    penalty_multiplier = np.sqrt(no_arbitrage_penalty)
    condition_limit = 4.0 - 1.0e-10

    def residuals(unconstrained: FloatArray) -> FloatArray:
        params = _ssvi_from_unconstrained(unconstrained)
        model = np.asarray(
            ssvi_total_variance(k, observation_theta, params), dtype=np.float64
        )
        data_residuals = residual_multiplier * (model - observed)
        phi = np.asarray(ssvi_phi(theta_audit, params), dtype=np.float64)
        rho_factor = 1.0 + abs(params.rho)
        linear_excess = np.maximum(
            theta_audit * phi * rho_factor - condition_limit, 0.0
        )
        quadratic_excess = np.maximum(
            theta_audit * phi * phi * rho_factor - condition_limit, 0.0
        )
        return np.concatenate(
            (
                data_residuals,
                penalty_multiplier * linear_excess,
                penalty_multiplier * quadratic_excess,
            )
        )

    optimization = least_squares(
        residuals,
        starting_point,
        jac="3-point",
        method="trf",
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        x_scale="jac",
    )
    params = _ssvi_from_unconstrained(optimization.x)
    report = ssvi_sufficient_no_arbitrage(theta_audit, params)
    if enforce_sufficient_no_arbitrage and not report.passed:
        raise NoArbitrageError(
            "calibrated SSVI parameters fail sufficient no-arbitrage conditions"
        )
    surface = SSVISurface(
        theta_curve.maturities,
        theta_curve.atm_total_variances,
        params,
        enforce_sufficient_no_arbitrage=enforce_sufficient_no_arbitrage,
    )
    model = np.asarray(
        ssvi_total_variance(k, observation_theta, params), dtype=np.float64
    )
    errors = model - observed
    return SSVIFitResult(
        surface=surface,
        params=params,
        success=bool(optimization.success),
        cost=float(optimization.cost),
        rmse=float(np.sqrt(np.mean(errors * errors))),
        weighted_rmse=float(
            np.sqrt(np.mean((residual_multiplier * errors) ** 2))
        ),
        optimality=float(optimization.optimality),
        nfev=int(optimization.nfev),
        message=str(optimization.message),
        no_arbitrage_report=report,
    )


def butterfly_g(
    log_moneyness: ArrayLike,
    total_variance: ArrayLike,
    first_derivative: ArrayLike,
    second_derivative: ArrayLike,
) -> float | FloatArray:
    """Evaluate Durrleman's density factor ``g(k)``."""

    try:
        k, variance, first, second = np.broadcast_arrays(
            np.asarray(log_moneyness, dtype=np.float64),
            np.asarray(total_variance, dtype=np.float64),
            np.asarray(first_derivative, dtype=np.float64),
            np.asarray(second_derivative, dtype=np.float64),
        )
    except ValueError as exc:
        raise SVIError("butterfly diagnostic inputs cannot be broadcast") from exc
    if not all(np.all(np.isfinite(item)) for item in (k, variance, first, second)):
        raise SVIError("butterfly diagnostic inputs must be finite")
    if np.any(variance <= 0.0):
        raise SVIError("total variance must be strictly positive for butterfly diagnostics")
    result = (
        (1.0 - k * first / (2.0 * variance)) ** 2
        - 0.25 * first * first * (1.0 / variance + 0.25)
        + 0.5 * second
    )
    if result.ndim == 0:
        return float(result)
    return np.asarray(result, dtype=np.float64)


def raw_svi_butterfly_g(
    log_moneyness: ArrayLike, params: RawSVIParams
) -> float | FloatArray:
    variance, first, second = raw_svi_derivatives(log_moneyness, params)
    return butterfly_g(log_moneyness, variance, first, second)


def ssvi_butterfly_g(
    log_moneyness: ArrayLike,
    atm_total_variance: ArrayLike,
    params: SSVIParams,
) -> float | FloatArray:
    """Evaluate ``g(k)`` for SSVI using analytic strike derivatives."""

    k = np.asarray(log_moneyness, dtype=np.float64)
    theta = np.asarray(atm_total_variance, dtype=np.float64)
    try:
        k, theta = np.broadcast_arrays(k, theta)
    except ValueError as exc:
        raise SVIError("log_moneyness and atm_total_variance cannot be broadcast") from exc
    if np.any(theta <= 0.0):
        raise SVIError("atm_total_variance must be strictly positive")
    phi = np.asarray(ssvi_phi(theta, params), dtype=np.float64)
    scaled = phi * k
    root = np.sqrt((scaled + params.rho) ** 2 + 1.0 - params.rho**2)
    variance = np.asarray(ssvi_total_variance(k, theta, params), dtype=np.float64)
    first = 0.5 * theta * phi * (params.rho + (scaled + params.rho) / root)
    second = (
        0.5
        * theta
        * phi**2
        * (1.0 - params.rho**2)
        / root**3
    )
    return butterfly_g(k, variance, first, second)


def butterfly_diagnostic(
    log_moneyness: ArrayLike,
    g_values: ArrayLike,
    *,
    tolerance: float = 1.0e-10,
) -> ButterflyDiagnostic:
    k = _one_dimensional(log_moneyness, "log_moneyness")
    g = _one_dimensional(g_values, "g_values")
    if k.shape != g.shape:
        raise SVIError("log_moneyness and g_values must have identical shapes")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise SVIError("tolerance must be finite and non-negative")
    violations = int(np.count_nonzero(g < -tolerance))
    return ButterflyDiagnostic(
        log_moneyness=k.copy(),
        g=g.copy(),
        minimum_g=float(np.min(g)),
        violation_count=violations,
        passed=violations == 0,
    )


def calendar_arbitrage_diagnostic(
    surface: SSVISurface,
    maturities: ArrayLike,
    log_moneyness: ArrayLike,
    *,
    tolerance: float = 1.0e-10,
) -> CalendarArbitrageDiagnostic:
    """Check that total variance is non-decreasing at each fixed strike coordinate."""

    maturity_grid = _one_dimensional(maturities, "maturities", minimum_size=2)
    k_grid = _one_dimensional(log_moneyness, "log_moneyness")
    if np.any(maturity_grid < 0.0) or np.any(np.diff(maturity_grid) <= 0.0):
        raise SVIError("maturities must be non-negative and strictly increasing")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise SVIError("tolerance must be finite and non-negative")
    variance = np.vstack(
        [
            np.asarray(surface.total_variance(k_grid, maturity), dtype=np.float64)
            for maturity in maturity_grid
        ]
    )
    increments = np.diff(variance, axis=0)
    violations = int(np.count_nonzero(increments < -tolerance))
    return CalendarArbitrageDiagnostic(
        maturities=maturity_grid.copy(),
        log_moneyness=k_grid.copy(),
        total_variance=variance,
        increments=increments,
        minimum_increment=float(np.min(increments)),
        violation_count=violations,
        passed=violations == 0,
    )


def price_arbitrage_diagnostic(
    strikes: ArrayLike,
    prices: ArrayLike,
    *,
    option_type: Literal["call", "put"] = "call",
    discount_factor: float = 1.0,
    tolerance: float = 1.0e-10,
) -> PriceArbitrageDiagnostic:
    """Check option-price monotonicity, vertical bounds and convexity in strike."""

    strike_array = _one_dimensional(strikes, "strikes", minimum_size=3)
    price_array = _one_dimensional(prices, "prices", minimum_size=3)
    if strike_array.shape != price_array.shape:
        raise SVIError("strikes and prices must have identical shapes")
    if np.any(strike_array <= 0.0) or np.any(np.diff(strike_array) <= 0.0):
        raise SVIError("strikes must be strictly positive and increasing")
    if option_type not in {"call", "put"}:
        raise SVIError("option_type must be either 'call' or 'put'")
    if discount_factor <= 0.0 or not np.isfinite(discount_factor):
        raise SVIError("discount_factor must be finite and strictly positive")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise SVIError("tolerance must be finite and non-negative")

    slopes = np.diff(price_array) / np.diff(strike_array)
    slope_increments = np.diff(slopes)
    if option_type == "call":
        monotonicity_violations = int(np.count_nonzero(np.diff(price_array) > tolerance))
        vertical_violations = int(
            np.count_nonzero((slopes > tolerance) | (slopes < -discount_factor - tolerance))
        )
    else:
        monotonicity_violations = int(np.count_nonzero(np.diff(price_array) < -tolerance))
        vertical_violations = int(
            np.count_nonzero((slopes < -tolerance) | (slopes > discount_factor + tolerance))
        )
    convexity_violations = int(np.count_nonzero(slope_increments < -tolerance))
    negative_violations = int(np.count_nonzero(price_array < -tolerance))
    return PriceArbitrageDiagnostic(
        strikes=strike_array.copy(),
        prices=price_array.copy(),
        slopes=slopes,
        slope_increments=slope_increments,
        monotonicity_violations=monotonicity_violations,
        vertical_spread_violations=vertical_violations,
        convexity_violations=convexity_violations,
        negative_price_violations=negative_violations,
        minimum_convexity_margin=float(np.min(slope_increments)),
        passed=(
            monotonicity_violations == 0
            and vertical_violations == 0
            and convexity_violations == 0
            and negative_violations == 0
        ),
    )


__all__ = [
    "ButterflyDiagnostic",
    "CalendarArbitrageDiagnostic",
    "MonotoneThetaCurve",
    "NoArbitrageError",
    "PriceArbitrageDiagnostic",
    "RawSVIParams",
    "SSVIFitResult",
    "SSVINoArbitrageReport",
    "SSVIParams",
    "SSVISurface",
    "SVIError",
    "SVIFitResult",
    "butterfly_diagnostic",
    "butterfly_g",
    "calendar_arbitrage_diagnostic",
    "fit_raw_svi",
    "fit_ssvi_surface",
    "price_arbitrage_diagnostic",
    "raw_svi_butterfly_g",
    "raw_svi_derivatives",
    "raw_svi_from_unconstrained",
    "raw_svi_jacobian",
    "raw_svi_to_unconstrained",
    "raw_svi_total_variance",
    "ssvi_butterfly_g",
    "ssvi_phi",
    "ssvi_sufficient_no_arbitrage",
    "ssvi_total_variance",
    "svi_calibration_weights",
]
