from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from crypto_vol_lab.backtest import BacktestConfig, RiskLimits, run_backtest
from crypto_vol_lab.data_models import QuoteSnapshot
from crypto_vol_lab.signals import (
    ResidualSignalConfig,
    SelectionConfig,
    compute_residual_signals,
    select_vega_neutral_portfolios,
)


def _option(
    symbol: str,
    snapshot_time: datetime,
    bid: float,
    ask: float,
    *,
    delta: float,
    vega: float = 10.0,
    gamma: float = 0.001,
    underlying: float = 100.0,
    expiration: datetime | None = None,
    asset: str = "BTC",
) -> QuoteSnapshot:
    return QuoteSnapshot(
        exchange="deribit",
        symbol=symbol,
        asset=asset,
        instrument_type="option",
        timestamp=snapshot_time,
        local_timestamp=snapshot_time,
        snapshot_time=snapshot_time,
        bid_price=bid,
        ask_price=ask,
        option_type="call",
        strike_price=100.0 if symbol == "CHEAP" else 110.0,
        expiration=expiration or snapshot_time + timedelta(days=7),
        underlying_price=underlying,
        delta=delta,
        gamma=gamma,
        vega=vega,
    )


def _perpetual(
    snapshot_time: datetime,
    price: float,
    *,
    asset: str = "BTC",
    available_at: datetime | None = None,
) -> QuoteSnapshot:
    arrival = available_at or snapshot_time
    return QuoteSnapshot(
        exchange="deribit",
        symbol=f"{asset}-PERPETUAL",
        asset=asset,
        instrument_type="perpetual",
        timestamp=arrival,
        local_timestamp=arrival,
        snapshot_time=snapshot_time,
        bid_price=price,
        ask_price=price,
    )


def _market(*, asset: str = "BTC") -> list[QuoteSnapshot]:
    t0 = datetime(2026, 1, 1, 0, tzinfo=UTC)
    expiry = t0 + timedelta(days=7)
    return [
        _option(
            "CHEAP", t0, 0.09, 0.10, delta=0.60, expiration=expiry, asset=asset
        ),
        _option(
            "RICH", t0, 0.14, 0.15, delta=0.40, expiration=expiry, asset=asset
        ),
        _perpetual(t0, 100.0, asset=asset),
        _option(
            "CHEAP",
            t0 + timedelta(hours=1),
            0.10,
            0.11,
            delta=0.60,
            underlying=100.0,
            expiration=expiry,
            asset=asset,
        ),
        _option(
            "RICH",
            t0 + timedelta(hours=1),
            0.14,
            0.15,
            delta=0.40,
            underlying=100.0,
            expiration=expiry,
            asset=asset,
        ),
        _perpetual(t0 + timedelta(hours=1), 100.0, asset=asset),
        _option(
            "CHEAP",
            t0 + timedelta(hours=2),
            0.12,
            0.13,
            delta=0.61,
            underlying=103.0,
            expiration=expiry,
            asset=asset,
        ),
        _option(
            "RICH",
            t0 + timedelta(hours=2),
            0.12,
            0.13,
            delta=0.42,
            underlying=103.0,
            expiration=expiry,
            asset=asset,
        ),
        _perpetual(t0 + timedelta(hours=2), 103.0, asset=asset),
    ]


def _model_price(quote: QuoteSnapshot) -> float:
    del quote
    return 0.12


def _config(*, max_position: float = 100.0) -> BacktestConfig:
    return BacktestConfig(
        horizons_hours=(1.0,),
        signal=ResidualSignalConfig(max_abs_score=None),
        selection=SelectionConfig(
            minimum_abs_score=1.0,
            legs_per_side=1,
            target_gross_vega=20.0,
        ),
        risk_limits=RiskLimits(
            max_gross_vega=1e9,
            max_gross_notional=1e9,
            max_abs_position_per_leg=max_position,
            max_scenario_loss=1e9,
        ),
        option_fee_rate=0.0,
        perpetual_fee_rate=0.0,
        constant_funding_rate_per_hour=0.001,
    )


def test_signal_is_zero_inside_spread_and_selection_is_same_expiry() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    common_expiry = t0 + timedelta(days=7)
    quotes = [
        _option("CHEAP", t0, 9.0, 11.0, delta=0.5, expiration=common_expiry),
        _option("RICH", t0, 13.0, 15.0, delta=0.4, expiration=common_expiry),
        _option("INSIDE", t0, 9.0, 11.0, delta=0.3, expiration=common_expiry),
    ]
    model = {"CHEAP": 13.0, "RICH": 11.0, "INSIDE": 10.0}
    batch = compute_residual_signals(
        quotes,
        model,
        ResidualSignalConfig(max_abs_score=None),
        strict=True,
    )
    by_symbol = {signal.instrument: signal for signal in batch.signals}
    assert by_symbol["CHEAP"].score < 0.0
    assert by_symbol["RICH"].score > 0.0
    assert by_symbol["INSIDE"].score == 0.0

    portfolios = select_vega_neutral_portfolios(
        batch.signals,
        SelectionConfig(
            minimum_abs_score=0.5,
            legs_per_side=1,
            target_gross_vega=20.0,
        ),
    )
    assert len(portfolios) == 1
    portfolio = portfolios[0]
    assert {leg.signal.expiry for leg in portfolio.legs} == {common_expiry}
    assert portfolio.net_vega == pytest.approx(0.0, abs=1e-12)
    assert {leg.side for leg in portfolio.legs} == {"long", "short"}


def test_backtest_uses_next_snapshot_and_attributes_inverse_pnl() -> None:
    result = run_backtest(_market(), _model_price, _config())

    assert result.diagnostics.completed_trades == 1
    trade = result.trades[0]
    assert trade.entry_market_time == trade.signal_time + timedelta(hours=1)
    assert trade.entry_time > trade.decision_time
    assert trade.exit_market_time == trade.entry_market_time + timedelta(hours=1)

    # Long: (0.12 - 0.11) * 103.  Short: -(0.13 - 0.14) * 103.
    assert trade.option_gross_pnl == pytest.approx(2.06)
    # Inverse net-transaction deltas: (0.60 - 0.105) - (0.40 - 0.145) = 0.24.
    assert trade.entry_net_delta_before_hedge == pytest.approx(0.24)
    # The option hedge target is -0.24 BTC.  Deribit inverse futures are ordered
    # in USD, hence N=-0.24*100=-24 USD, or -2.4 BTC-PERPETUAL contracts ($10).
    assert trade.hedge_quantity == pytest.approx(-0.24)
    assert trade.hedge_notional_usd == pytest.approx(-24.0)
    assert trade.hedge_contracts == pytest.approx(-2.4)
    expected_coin_pnl = -24.0 * (1.0 / 100.0 - 1.0 / 103.0)
    assert trade.hedge_gross_pnl == pytest.approx(expected_coin_pnl * 103.0)
    # Positive funding is received by this short USD-notional hedge.
    assert trade.funding_pnl == pytest.approx(24.0 * 0.001)
    assert trade.net_pnl == pytest.approx(1.364)
    assert trade.pnl_attribution["net"] == pytest.approx(trade.net_pnl)


def test_pretrade_position_limit_scales_both_sides_without_breaking_vega_neutrality() -> None:
    result = run_backtest(_market(), _model_price, _config(max_position=0.5))

    trade = result.trades[0]
    assert trade.risk_scale == pytest.approx(0.5)
    assert max(abs(leg.quantity) for leg in trade.legs) <= 0.5
    assert trade.net_vega == pytest.approx(0.0, abs=1e-12)
    assert result.diagnostics.risk_scaled_trades == 1


def test_missing_next_snapshot_leg_is_skipped_instead_of_stale_filled() -> None:
    market = [quote for quote in _market() if not (
        quote.symbol == "RICH"
        and quote.snapshot_time == datetime(2026, 1, 1, 1, tzinfo=UTC)
    )]
    result = run_backtest(market, _model_price, _config())

    assert result.trades == ()
    assert result.diagnostics.count("missing_entry_leg") == 1


@pytest.mark.parametrize(
    ("hour", "reason"),
    [
        (1, "missing_or_invalid_entry_perpetual"),
        (2, "missing_or_invalid_exit_perpetual"),
    ],
)
def test_missing_perpetual_is_rejected_without_option_index_fallback(
    hour: int,
    reason: str,
) -> None:
    target = datetime(2026, 1, 1, hour, tzinfo=UTC)
    market = [
        quote
        for quote in _market()
        if not (quote.instrument_type == "perpetual" and quote.snapshot_time == target)
    ]

    result = run_backtest(market, _model_price, _config())

    assert result.trades == ()
    assert result.diagnostics.count(reason) == 1


@pytest.mark.parametrize(
    ("hour", "reason"),
    [(1, "stale_entry_perpetual"), (2, "stale_exit_perpetual")],
)
def test_stale_perpetual_is_rejected(hour: int, reason: str) -> None:
    target = datetime(2026, 1, 1, hour, tzinfo=UTC)
    market = [
        quote
        for quote in _market()
        if not (quote.instrument_type == "perpetual" and quote.snapshot_time == target)
    ]
    market.append(
        _perpetual(
            target,
            100.0 if hour == 1 else 103.0,
            available_at=target - timedelta(minutes=30),
        )
    )

    result = run_backtest(market, _model_price, _config())

    assert result.trades == ()
    assert result.diagnostics.count(reason) == 1


def test_inverse_option_fee_uses_notional_rate_and_premium_cap() -> None:
    config = _config()
    config = BacktestConfig(
        horizons_hours=config.horizons_hours,
        signal=config.signal,
        selection=config.selection,
        risk_limits=config.risk_limits,
        option_fee_rate=0.0003,
        option_fee_cap_fraction=0.125,
        perpetual_fee_rate=0.0,
    )
    trade = run_backtest(_market(), _model_price, config).trades[0]

    # Every side is below the 12.5%-of-premium cap, so the 3 bp underlying
    # notional rate binds: 2 contracts at entry * 100 and exit * 103.
    assert trade.option_fees == pytest.approx(0.0003 * (2.0 * 100.0 + 2.0 * 103.0))


def test_inverse_perpetual_fee_uses_fixed_usd_amount_on_each_execution() -> None:
    config = _config()
    config = BacktestConfig(
        horizons_hours=config.horizons_hours,
        signal=config.signal,
        selection=config.selection,
        risk_limits=config.risk_limits,
        option_fee_rate=0.0,
        perpetual_fee_rate=0.00035,
        constant_funding_rate_per_hour=0.0,
    )
    trade = run_backtest(_market(), _model_price, config).trades[0]

    assert trade.hedge_notional_usd == pytest.approx(-24.0)
    assert trade.perpetual_fees == pytest.approx(2.0 * 0.00035 * 24.0)


def test_long_inverse_perpetual_profits_on_price_rise_and_pays_positive_funding() -> None:
    market = [
        replace(
            quote,
            delta=0.20 if quote.symbol == "CHEAP" else 0.80,
        )
        if quote.instrument_type == "option"
        else quote
        for quote in _market()
    ]
    trade = run_backtest(market, _model_price, _config()).trades[0]

    # Option NTD is 0.095 - 0.655 = -0.56 BTC, hence a +56 USD perp hedge.
    assert trade.entry_net_delta_before_hedge == pytest.approx(-0.56)
    assert trade.hedge_quantity == pytest.approx(0.56)
    assert trade.hedge_notional_usd == pytest.approx(56.0)
    assert trade.hedge_contracts == pytest.approx(5.6)
    expected_coin_pnl = 56.0 * (1.0 / 100.0 - 1.0 / 103.0)
    assert trade.hedge_gross_pnl == pytest.approx(expected_coin_pnl * 103.0)
    assert trade.hedge_gross_pnl > 0.0
    assert trade.funding_pnl == pytest.approx(-56.0 * 0.001)


def test_eth_inverse_perpetual_uses_one_dollar_contracts() -> None:
    trade = run_backtest(_market(asset="ETH"), _model_price, _config()).trades[0]

    assert trade.hedge_notional_usd == pytest.approx(-24.0)
    assert trade.hedge_contracts == pytest.approx(-24.0)


def test_invalid_risk_limit_is_explicit() -> None:
    with pytest.raises(ValueError, match="max_gross_vega"):
        RiskLimits(max_gross_vega=0.0)

    with pytest.raises(ValueError, match="max_perpetual_quote_age"):
        BacktestConfig(max_perpetual_quote_age=timedelta(seconds=-1))
    with pytest.raises(ValueError, match="max_perpetual_quote_age"):
        BacktestConfig(max_perpetual_quote_age=None)  # type: ignore[arg-type]
