"""Dependency-light inference and performance statistics for backtest results.

The resampling routines preserve the dependence structure that ordinary iid
bootstraps destroy: either complete UTC calendar-day clusters, fixed circular
blocks, or Politis--Romano stationary blocks are sampled.  Sharpe inference
uses finite-sample skewness and kurtosis, and the deflated Sharpe benchmark
penalizes the number of strategies tried.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from statistics import NormalDist
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

Statistic = Callable[[NDArray[np.float64]], float]
DAILY_PERIODS_PER_YEAR = 365.0


class TradeLike(Protocol):
    net_pnl: float
    turnover: float
    exit_time: datetime
    requested_horizon_hours: float


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    method: str
    estimate: float
    standard_error: float
    confidence_level: float
    ci_low: float
    ci_high: float
    n_resamples: int
    replicates: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    observations: int
    total_pnl: float
    mean_pnl: float
    volatility: float
    annualized_sharpe: float
    probabilistic_sharpe: float
    max_drawdown: float
    max_drawdown_duration: int
    hit_rate: float
    profit_factor: float
    total_turnover: float
    average_turnover: float


@dataclass(frozen=True, slots=True)
class DailyTradeAggregate:
    """Net trading activity assigned to one UTC calendar exit day."""

    day: date
    net_pnl: float
    turnover: float
    trade_count: int


def _as_clean_array(values: ArrayLike, *, minimum_size: int = 1) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if array.size < minimum_size:
        raise ValueError(f"at least {minimum_size} observations are required")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contain NaN or infinite observations")
    return array.astype(np.float64, copy=False)


def _validate_bootstrap_inputs(
    values: ArrayLike,
    n_resamples: int,
    confidence_level: float,
) -> NDArray[np.float64]:
    array = _as_clean_array(values)
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least one")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    return array


def _finish_bootstrap(
    method: str,
    values: NDArray[np.float64],
    replicates: NDArray[np.float64],
    statistic: Statistic,
    confidence_level: float,
) -> BootstrapResult:
    if not np.all(np.isfinite(replicates)):
        raise ValueError("bootstrap statistic produced a non-finite replicate")
    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0])
    standard_error = float(np.std(replicates, ddof=1)) if replicates.size > 1 else 0.0
    return BootstrapResult(
        method=method,
        estimate=float(statistic(values)),
        standard_error=standard_error,
        confidence_level=confidence_level,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_resamples=int(replicates.size),
        replicates=replicates,
    )


def _calendar_day(value: object) -> date | str:
    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            return value.astimezone(UTC).date()
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[D]"))
    if hasattr(value, "date"):
        candidate = value.date()
        if isinstance(candidate, date):
            return candidate
    # ISO timestamps sort and cluster correctly by their YYYY-MM-DD prefix.
    text = str(value)
    if len(text) >= 10:
        return text[:10]
    raise ValueError(f"cannot extract a calendar day from {value!r}")


def day_clustered_bootstrap(
    values: ArrayLike,
    timestamps: Sequence[object],
    statistic: Statistic = np.mean,
    *,
    n_resamples: int = 2_000,
    confidence_level: float = 0.95,
    seed: int | None = 0,
) -> BootstrapResult:
    """Resample complete calendar days with replacement.

    All observations from a sampled day are kept together, preserving
    cross-sectional and intraday dependence.  Days are drawn independently;
    irregularly populated days retain their original number of observations.
    """

    array = _validate_bootstrap_inputs(values, n_resamples, confidence_level)
    if len(timestamps) != array.size:
        raise ValueError("timestamps and values must have identical lengths")
    clustered: defaultdict[date | str, list[int]] = defaultdict(list)
    for index, timestamp in enumerate(timestamps):
        clustered[_calendar_day(timestamp)].append(index)
    clusters = [np.asarray(indices, dtype=int) for _, indices in sorted(clustered.items())]
    if not clusters:
        raise ValueError("at least one day cluster is required")

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_resamples, dtype=float)
    for sample_index in range(n_resamples):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[int(index)] for index in chosen])
        replicates[sample_index] = float(statistic(array[indices]))
    return _finish_bootstrap(
        "day_clustered", array, replicates, statistic, confidence_level
    )


def stationary_bootstrap(
    values: ArrayLike,
    statistic: Statistic = np.mean,
    *,
    mean_block_length: float = 24.0,
    n_resamples: int = 2_000,
    confidence_level: float = 0.95,
    seed: int | None = 0,
) -> BootstrapResult:
    """Politis--Romano stationary bootstrap with circular continuation."""

    array = _validate_bootstrap_inputs(values, n_resamples, confidence_level)
    if mean_block_length < 1.0:
        raise ValueError("mean_block_length must be at least one")
    restart_probability = 1.0 / mean_block_length
    rng = np.random.default_rng(seed)
    n = array.size
    replicates = np.empty(n_resamples, dtype=float)
    indices = np.empty(n, dtype=int)
    for sample_index in range(n_resamples):
        indices[0] = int(rng.integers(0, n))
        for position in range(1, n):
            if rng.random() < restart_probability:
                indices[position] = int(rng.integers(0, n))
            else:
                indices[position] = (indices[position - 1] + 1) % n
        replicates[sample_index] = float(statistic(array[indices]))
    return _finish_bootstrap(
        "stationary", array, replicates, statistic, confidence_level
    )


def block_bootstrap(
    values: ArrayLike,
    statistic: Statistic = np.mean,
    *,
    block_length: int = 24,
    n_resamples: int = 2_000,
    confidence_level: float = 0.95,
    seed: int | None = 0,
) -> BootstrapResult:
    """Fixed-length circular block bootstrap."""

    array = _validate_bootstrap_inputs(values, n_resamples, confidence_level)
    if block_length < 1:
        raise ValueError("block_length must be at least one")
    rng = np.random.default_rng(seed)
    n = array.size
    blocks_needed = math.ceil(n / block_length)
    offsets = np.arange(block_length, dtype=int)
    replicates = np.empty(n_resamples, dtype=float)
    for sample_index in range(n_resamples):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = np.concatenate([(start + offsets) % n for start in starts])[:n]
        replicates[sample_index] = float(statistic(array[indices]))
    return _finish_bootstrap(
        "fixed_block", array, replicates, statistic, confidence_level
    )


def sharpe_ratio(values: ArrayLike, *, periods_per_year: float = 1.0) -> float:
    """Annualized arithmetic Sharpe ratio with zero risk-free rate."""

    array = _as_clean_array(values, minimum_size=2)
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be strictly positive")
    volatility = float(np.std(array, ddof=1))
    if volatility <= 0.0:
        return float("nan")
    return float(np.mean(array) / volatility * math.sqrt(periods_per_year))


def _standardized_moments(values: NDArray[np.float64]) -> tuple[float, float]:
    centered = values - np.mean(values)
    variance = float(np.mean(centered * centered))
    if variance <= 0.0:
        return float("nan"), float("nan")
    scale = math.sqrt(variance)
    skewness = float(np.mean((centered / scale) ** 3))
    pearson_kurtosis = float(np.mean((centered / scale) ** 4))
    return skewness, pearson_kurtosis


def probabilistic_sharpe_ratio(
    values: ArrayLike,
    *,
    benchmark_sharpe: float = 0.0,
    periods_per_year: float = 1.0,
) -> float:
    """Probability that the population Sharpe exceeds ``benchmark_sharpe``.

    The Bailey--López de Prado approximation corrects the Sharpe standard error
    for observed skewness and Pearson kurtosis.  Both reported and benchmark
    Sharpe ratios use the requested annualization convention.
    """

    array = _as_clean_array(values, minimum_size=3)
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be strictly positive")
    annualized = sharpe_ratio(array, periods_per_year=periods_per_year)
    if not math.isfinite(annualized):
        return float("nan")
    observed = annualized / math.sqrt(periods_per_year)
    benchmark = benchmark_sharpe / math.sqrt(periods_per_year)
    skewness, kurtosis = _standardized_moments(array)
    denominator_squared = (
        1.0
        - skewness * observed
        + 0.25 * (kurtosis - 1.0) * observed * observed
    )
    if not math.isfinite(denominator_squared) or denominator_squared <= 0.0:
        return float("nan")
    z_score = (
        (observed - benchmark)
        * math.sqrt(array.size - 1.0)
        / math.sqrt(denominator_squared)
    )
    return float(NormalDist().cdf(z_score))


def expected_maximum_sharpe(
    *,
    n_trials: int,
    sharpe_std: float,
    mean_sharpe: float = 0.0,
) -> float:
    """Expected maximum Sharpe among independent approximately normal trials."""

    if n_trials < 1:
        raise ValueError("n_trials must be at least one")
    if sharpe_std < 0.0:
        raise ValueError("sharpe_std cannot be negative")
    if n_trials == 1 or sharpe_std == 0.0:
        return float(mean_sharpe)
    euler_mascheroni = 0.5772156649015329
    normal = NormalDist()
    first = normal.inv_cdf(1.0 - 1.0 / n_trials)
    second = normal.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return float(
        mean_sharpe
        + sharpe_std
        * ((1.0 - euler_mascheroni) * first + euler_mascheroni * second)
    )


def deflated_sharpe_ratio(
    values: ArrayLike,
    *,
    n_trials: int,
    periods_per_year: float = 1.0,
    mean_trial_sharpe: float = 0.0,
    trial_sharpe_std: float | None = None,
) -> float:
    """Probabilistic Sharpe penalized for strategy-selection multiplicity.

    ``mean_trial_sharpe`` and ``trial_sharpe_std`` are annualized.  When the
    latter is unavailable, a conservative iid standard error is estimated for
    an annualized Sharpe under the null.  Supplying the dispersion of all tried
    strategy Sharpes is preferable in a research report.
    """

    array = _as_clean_array(values, minimum_size=3)
    if n_trials < 1:
        raise ValueError("n_trials must be at least one")
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be strictly positive")
    if trial_sharpe_std is None:
        trial_sharpe_std = math.sqrt(periods_per_year / max(1.0, array.size - 1.0))
    benchmark = expected_maximum_sharpe(
        n_trials=n_trials,
        sharpe_std=trial_sharpe_std,
        mean_sharpe=mean_trial_sharpe,
    )
    return probabilistic_sharpe_ratio(
        array,
        benchmark_sharpe=benchmark,
        periods_per_year=periods_per_year,
    )


def adjust_pvalues(
    pvalues: ArrayLike,
    *,
    method: Literal["holm", "benjamini-hochberg", "bonferroni"] = "holm",
) -> NDArray[np.float64]:
    """Return family-wise or false-discovery-rate adjusted p-values."""

    array = _as_clean_array(pvalues)
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    m = array.size
    if method == "bonferroni":
        return np.minimum(1.0, array * m)

    order = np.argsort(array)
    ordered = array[order]
    adjusted_ordered = np.empty(m, dtype=float)
    if method == "holm":
        running_max = 0.0
        for index, value in enumerate(ordered):
            candidate = (m - index) * value
            running_max = max(running_max, candidate)
            adjusted_ordered[index] = min(1.0, running_max)
    elif method == "benjamini-hochberg":
        running_min = 1.0
        for index in range(m - 1, -1, -1):
            rank = index + 1
            candidate = ordered[index] * m / rank
            running_min = min(running_min, candidate)
            adjusted_ordered[index] = min(1.0, running_min)
    else:
        raise ValueError(f"unsupported multiple-testing method: {method}")

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted


def pnl_drawdown(values: ArrayLike) -> tuple[float, int, NDArray[np.float64]]:
    """Maximum peak-to-trough PnL loss, duration, and drawdown path."""

    array = _as_clean_array(values)
    cumulative = np.cumsum(array)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
    drawdowns = cumulative - running_peak[1:]
    max_drawdown = float(max(0.0, -np.min(drawdowns)))

    longest = 0
    current = 0
    for drawdown in drawdowns:
        if drawdown < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return max_drawdown, longest, drawdowns


def performance_summary(
    pnl: ArrayLike,
    *,
    turnover: ArrayLike | None = None,
    periods_per_year: float = 1.0,
    benchmark_sharpe: float = 0.0,
) -> PerformanceSummary:
    """Summarize PnL, risk, hit rate and trading intensity."""

    array = _as_clean_array(pnl)
    if turnover is None:
        turnover_array = np.zeros_like(array)
    else:
        turnover_array = _as_clean_array(turnover)
        if turnover_array.size != array.size:
            raise ValueError("turnover and pnl must have identical lengths")
        if np.any(turnover_array < 0.0):
            raise ValueError("turnover cannot be negative")

    volatility = float(np.std(array, ddof=1)) if array.size > 1 else float("nan")
    annualized_sharpe = (
        sharpe_ratio(array, periods_per_year=periods_per_year)
        if array.size > 1
        else float("nan")
    )
    psr = (
        probabilistic_sharpe_ratio(
            array,
            benchmark_sharpe=benchmark_sharpe,
            periods_per_year=periods_per_year,
        )
        if array.size > 2
        else float("nan")
    )
    max_drawdown, duration, _ = pnl_drawdown(array)
    gains = float(np.sum(array[array > 0.0]))
    losses = float(-np.sum(array[array < 0.0]))
    if losses > 0.0:
        profit_factor = gains / losses
    elif gains > 0.0:
        profit_factor = math.inf
    else:
        profit_factor = float("nan")
    return PerformanceSummary(
        observations=int(array.size),
        total_pnl=float(np.sum(array)),
        mean_pnl=float(np.mean(array)),
        volatility=volatility,
        annualized_sharpe=annualized_sharpe,
        probabilistic_sharpe=psr,
        max_drawdown=max_drawdown,
        max_drawdown_duration=duration,
        hit_rate=float(np.mean(array > 0.0)),
        profit_factor=float(profit_factor),
        total_turnover=float(np.sum(turnover_array)),
        average_turnover=float(np.mean(turnover_array)),
    )


def aggregate_daily_trades(
    trades: Iterable[TradeLike],
) -> dict[float, tuple[DailyTradeAggregate, ...]]:
    """Sum trade P&L and turnover by horizon and UTC exit day.

    Daily aggregation makes the independent observation used by performance
    statistics explicit.  A trade must have a timezone-aware ``exit_time`` so
    that the calendar-day assignment cannot depend on the host timezone.
    """

    pnl_by_key: defaultdict[tuple[float, date], float] = defaultdict(float)
    turnover_by_key: defaultdict[tuple[float, date], float] = defaultdict(float)
    count_by_key: defaultdict[tuple[float, date], int] = defaultdict(int)

    for trade in trades:
        exit_time = trade.exit_time
        if exit_time.tzinfo is None or exit_time.utcoffset() is None:
            raise ValueError("trade exit_time must be timezone-aware")
        horizon = float(trade.requested_horizon_hours)
        net_pnl = float(trade.net_pnl)
        turnover = float(trade.turnover)
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise ValueError("trade holding horizon must be finite and positive")
        if not math.isfinite(net_pnl):
            raise ValueError("trade net_pnl must be finite")
        if not math.isfinite(turnover) or turnover < 0.0:
            raise ValueError("trade turnover must be finite and non-negative")

        key = (horizon, exit_time.astimezone(UTC).date())
        pnl_by_key[key] += net_pnl
        turnover_by_key[key] += turnover
        count_by_key[key] += 1

    horizons = sorted({horizon for horizon, _ in pnl_by_key})
    return {
        horizon: tuple(
            DailyTradeAggregate(
                day=day,
                net_pnl=pnl_by_key[(horizon, day)],
                turnover=turnover_by_key[(horizon, day)],
                trade_count=count_by_key[(horizon, day)],
            )
            for day in sorted(
                day for candidate_horizon, day in pnl_by_key if candidate_horizon == horizon
            )
        )
        for horizon in horizons
    }


def summarize_trades(
    trades: Iterable[TradeLike],
    *,
    periods_per_year: float = DAILY_PERIODS_PER_YEAR,
) -> dict[float, PerformanceSummary]:
    """Summarize daily P&L separately for each holding horizon.

    Trades are first summed by UTC exit day.  Consequently ``observations``,
    mean P&L, volatility, hit rate, drawdown and Sharpe all use daily totals;
    the default crypto-market annualization is ``sqrt(365)``.
    """

    grouped = aggregate_daily_trades(trades)
    summaries: dict[float, PerformanceSummary] = {}
    for horizon, cohort in sorted(grouped.items()):
        summaries[horizon] = performance_summary(
            [aggregate.net_pnl for aggregate in cohort],
            turnover=[aggregate.turnover for aggregate in cohort],
            periods_per_year=periods_per_year,
        )
    return summaries


__all__ = [
    "DAILY_PERIODS_PER_YEAR",
    "BootstrapResult",
    "DailyTradeAggregate",
    "PerformanceSummary",
    "adjust_pvalues",
    "aggregate_daily_trades",
    "block_bootstrap",
    "day_clustered_bootstrap",
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe",
    "performance_summary",
    "pnl_drawdown",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
    "stationary_bootstrap",
    "summarize_trades",
]
