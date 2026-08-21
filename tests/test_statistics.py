from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from crypto_vol_lab.statistics import (
    DAILY_PERIODS_PER_YEAR,
    adjust_pvalues,
    aggregate_daily_trades,
    block_bootstrap,
    day_clustered_bootstrap,
    deflated_sharpe_ratio,
    performance_summary,
    pnl_drawdown,
    probabilistic_sharpe_ratio,
    stationary_bootstrap,
    summarize_trades,
)


def test_day_clustered_bootstrap_keeps_intraday_observations_together() -> None:
    timestamps = [
        datetime(2026, 1, 1, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 1, tzinfo=UTC),
        datetime(2026, 2, 1, 0, tzinfo=UTC),
        datetime(2026, 2, 1, 1, tzinfo=UTC),
    ]
    result = day_clustered_bootstrap(
        [1.0, 1.0, -1.0, -1.0],
        timestamps,
        statistic=np.sum,
        n_resamples=200,
        seed=7,
    )

    assert result.estimate == 0.0
    assert set(np.unique(result.replicates)).issubset({-4.0, 0.0, 4.0})
    assert result.method == "day_clustered"


def test_stationary_and_fixed_block_bootstraps_are_seeded_and_finite() -> None:
    values = np.sin(np.arange(64, dtype=float) / 5.0) + 0.01
    first = stationary_bootstrap(
        values,
        mean_block_length=8.0,
        n_resamples=100,
        seed=11,
    )
    second = stationary_bootstrap(
        values,
        mean_block_length=8.0,
        n_resamples=100,
        seed=11,
    )
    fixed = block_bootstrap(values, block_length=8, n_resamples=100, seed=11)

    np.testing.assert_array_equal(first.replicates, second.replicates)
    assert first.estimate == pytest.approx(float(np.mean(values)))
    assert np.isfinite(first.standard_error)
    assert np.isfinite(fixed.ci_low)
    assert fixed.ci_low <= fixed.ci_high


def test_probabilistic_and_deflated_sharpe_penalize_multiple_trials() -> None:
    values = np.asarray(
        [0.03, -0.01, 0.02, 0.01, -0.005, 0.025, 0.015, -0.002] * 8
    )
    psr = probabilistic_sharpe_ratio(values, periods_per_year=12.0)
    dsr = deflated_sharpe_ratio(
        values,
        n_trials=20,
        periods_per_year=12.0,
    )

    assert 0.5 < psr <= 1.0
    assert 0.0 <= dsr <= psr


def test_multiple_testing_adjustments_preserve_input_order() -> None:
    pvalues = np.asarray([0.04, 0.001, 0.02])
    holm = adjust_pvalues(pvalues, method="holm")
    bh = adjust_pvalues(pvalues, method="benjamini-hochberg")
    bonferroni = adjust_pvalues(pvalues, method="bonferroni")

    np.testing.assert_allclose(holm, [0.04, 0.003, 0.04])
    np.testing.assert_allclose(bh, [0.04, 0.003, 0.03])
    np.testing.assert_allclose(bonferroni, [0.12, 0.003, 0.06])


def test_drawdown_turnover_and_hit_rate_summary() -> None:
    pnl = np.asarray([2.0, -1.0, -3.0, 4.0, -1.0])
    drawdown, duration, path = pnl_drawdown(pnl)
    summary = performance_summary(pnl, turnover=[1, 2, 3, 4, 5])

    assert drawdown == pytest.approx(4.0)
    assert duration == 2
    np.testing.assert_allclose(path, [0.0, -1.0, -4.0, 0.0, -1.0])
    assert summary.max_drawdown == pytest.approx(4.0)
    assert summary.hit_rate == pytest.approx(0.4)
    assert summary.total_turnover == pytest.approx(15.0)
    assert summary.profit_factor == pytest.approx(1.2)


@dataclass
class _Trade:
    net_pnl: float
    turnover: float
    exit_time: datetime
    requested_horizon_hours: float


def test_trade_summaries_are_separate_by_horizon() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trades = [
        _Trade(1.0, 10.0, start, 1.0),
        _Trade(-0.5, 8.0, start + timedelta(hours=1), 1.0),
        _Trade(2.0, 20.0, start + timedelta(hours=4), 4.0),
    ]
    summaries = summarize_trades(trades)

    assert set(summaries) == {1.0, 4.0}
    assert summaries[1.0].observations == 1
    assert summaries[1.0].total_pnl == pytest.approx(0.5)
    assert np.isnan(summaries[1.0].annualized_sharpe)
    assert summaries[4.0].total_turnover == pytest.approx(20.0)


def test_trade_sharpe_uses_utc_daily_totals_before_annualization() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trades = [
        _Trade(1.0, 10.0, start, 1.0),
        # 00:30 at UTC+02 is still the previous UTC calendar day.
        _Trade(1.0, 20.0, datetime.fromisoformat("2026-01-02T00:30:00+02:00"), 1.0),
        _Trade(-1.0, 30.0, start + timedelta(days=1), 1.0),
        _Trade(3.0, 40.0, start + timedelta(days=2), 1.0),
    ]

    daily = aggregate_daily_trades(trades)[1.0]
    summary = summarize_trades(trades)[1.0]
    expected_daily_pnl = np.asarray([2.0, -1.0, 3.0])
    expected_sharpe = (
        float(np.mean(expected_daily_pnl))
        / float(np.std(expected_daily_pnl, ddof=1))
        * np.sqrt(DAILY_PERIODS_PER_YEAR)
    )

    assert [aggregate.net_pnl for aggregate in daily] == [2.0, -1.0, 3.0]
    assert [aggregate.turnover for aggregate in daily] == [30.0, 30.0, 40.0]
    assert [aggregate.trade_count for aggregate in daily] == [2, 1, 1]
    assert summary.observations == 3
    assert summary.mean_pnl == pytest.approx(float(np.mean(expected_daily_pnl)))
    assert summary.annualized_sharpe == pytest.approx(expected_sharpe)


def test_daily_trade_aggregation_requires_timezone_aware_exit_time() -> None:
    trade = _Trade(1.0, 1.0, datetime(2026, 1, 1), 1.0)

    with pytest.raises(ValueError, match="timezone-aware"):
        summarize_trades([trade])


def test_invalid_bootstrap_arguments_are_explicit() -> None:
    with pytest.raises(ValueError, match="mean_block_length"):
        stationary_bootstrap([1.0, 2.0], mean_block_length=0.0)
    with pytest.raises(ValueError, match="p-values"):
        adjust_pvalues([1.1])
