"""Event-time backtester for volatility-surface relative-value signals.

The engine is intentionally conservative:

* a signal observed on snapshot *t* can only execute on snapshot *t+1*;
* long options cross the ask and liquidate at the bid (vice versa for shorts);
* the initial option delta is converted to a signed USD amount of the matching
  coin-settled Deribit inverse perpetual and crossed at its executable quote;
* fees, funding, bid/ask turnover and risk scaling are explicit;
* one- and four-hour horizons are separate hypothetical trade cohorts.

The implementation is dependency-light and consumes structural quote protocols,
so it does not import the data ingestion layer at runtime.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

from .signals import (
    ModelPriceResolver,
    PortfolioSelection,
    QuoteSnapshot,
    ResidualSignalConfig,
    SelectionConfig,
    compute_residual_signals,
    quote_available_at,
    quote_bid_ask,
    quote_expiry,
    quote_instrument,
    quote_snapshot_time,
    select_vega_neutral_portfolios,
)


class FundingRateSnapshot(Protocol):
    timestamp: datetime
    rate: float


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Pre-trade limits in portfolio currency and underlying units."""

    max_gross_vega: float = 100.0
    max_gross_notional: float = 1_000_000.0
    max_abs_position_per_leg: float = 10.0
    max_scenario_loss: float = 100_000.0
    scenario_underlying_shock: float = 0.10
    scenario_volatility_shock: float = 0.20
    minimum_scale: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "max_gross_vega",
            "max_gross_notional",
            "max_abs_position_per_leg",
            "max_scenario_loss",
        ):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be strictly positive")
        if not 0.0 < self.scenario_underlying_shock < 1.0:
            raise ValueError("scenario_underlying_shock must lie in (0, 1)")
        if self.scenario_volatility_shock <= 0.0:
            raise ValueError("scenario_volatility_shock must be strictly positive")
        if not 0.0 < self.minimum_scale <= 1.0:
            raise ValueError("minimum_scale must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Execution, cost, holding-period and risk assumptions."""

    horizons_hours: tuple[float, ...] = (1.0, 4.0)
    signal: ResidualSignalConfig = field(default_factory=ResidualSignalConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    max_portfolios_per_snapshot: int = 1
    option_contract_multiplier: float = 1.0
    inverse_options: bool = True
    option_fee_rate: float = 0.0003
    option_fee_cap_fraction: float = 0.125
    option_fee_per_contract_per_side: float = 0.0
    perpetual_fee_rate: float = 0.00035
    max_perpetual_quote_age: timedelta = timedelta(minutes=15)
    constant_funding_rate_per_hour: float = 0.0
    max_exit_delay: timedelta | None = timedelta(hours=1)

    def __post_init__(self) -> None:
        if not self.horizons_hours or any(h <= 0.0 for h in self.horizons_hours):
            raise ValueError("horizons_hours must contain strictly positive horizons")
        if len(set(self.horizons_hours)) != len(self.horizons_hours):
            raise ValueError("horizons_hours cannot contain duplicates")
        if self.max_portfolios_per_snapshot < 1:
            raise ValueError("max_portfolios_per_snapshot must be at least one")
        if self.option_contract_multiplier <= 0.0:
            raise ValueError("option_contract_multiplier must be strictly positive")
        for name in (
            "option_fee_rate",
            "option_fee_cap_fraction",
            "option_fee_per_contract_per_side",
            "perpetual_fee_rate",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if not isinstance(self.max_perpetual_quote_age, timedelta):
            raise ValueError("max_perpetual_quote_age must be a timedelta")
        if self.max_perpetual_quote_age < timedelta(0):
            raise ValueError("max_perpetual_quote_age cannot be negative")
        if self.max_exit_delay is not None and self.max_exit_delay < timedelta(0):
            raise ValueError("max_exit_delay cannot be negative")


@dataclass(frozen=True, slots=True)
class ExecutedLeg:
    instrument: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_mid: float
    exit_mid: float
    entry_underlying: float
    exit_underlying: float
    entry_vega: float
    entry_delta: float

    @property
    def side(self) -> str:
        return "long" if self.quantity > 0.0 else "short"


@dataclass(frozen=True, slots=True)
class TradeResult:
    """One selected portfolio evaluated at one holding horizon.

    ``hedge_quantity`` is the signed BTC/ETH exposure targeted at entry,
    ``hedge_notional_usd`` is Deribit's signed inverse-perpetual order amount,
    and ``hedge_contracts`` divides that amount by the venue contract size.
    """

    signal_time: datetime
    decision_time: datetime
    entry_market_time: datetime
    entry_time: datetime
    exit_market_time: datetime
    exit_time: datetime
    requested_horizon_hours: float
    realized_holding_hours: float
    expiry: datetime
    legs: tuple[ExecutedLeg, ...]
    hedge_quantity: float
    hedge_notional_usd: float
    hedge_contracts: float
    hedge_entry_price: float
    hedge_exit_price: float
    option_gross_pnl: float
    hedge_gross_pnl: float
    option_fees: float
    perpetual_fees: float
    funding_pnl: float
    net_pnl: float
    turnover: float
    gross_vega: float
    net_vega: float
    entry_net_delta_before_hedge: float
    gross_notional: float
    scenario_loss: float
    risk_scale: float
    signal_strength: float

    @property
    def total_fees(self) -> float:
        return self.option_fees + self.perpetual_fees

    @property
    def pnl_attribution(self) -> Mapping[str, float]:
        return {
            "option": self.option_gross_pnl,
            "perpetual_hedge": self.hedge_gross_pnl,
            "option_fees": -self.option_fees,
            "perpetual_fees": -self.perpetual_fees,
            "funding": self.funding_pnl,
            "net": self.net_pnl,
        }


@dataclass(frozen=True, slots=True)
class BacktestDiagnostics:
    market_snapshots: int
    signal_snapshots_evaluated: int
    accepted_signals: int
    rejected_signal_quotes: int
    candidate_portfolios: int
    completed_trades: int
    risk_scaled_trades: int
    skip_counts: tuple[tuple[str, int], ...]

    def count(self, reason: str) -> int:
        return dict(self.skip_counts).get(reason, 0)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    trades: tuple[TradeResult, ...]
    diagnostics: BacktestDiagnostics

    def equity_curve(
        self, horizon_hours: float | None = None
    ) -> tuple[tuple[datetime, float], ...]:
        """Return cumulative realized PnL ordered by exit information time."""

        selected = (
            trade
            for trade in self.trades
            if horizon_hours is None or trade.requested_horizon_hours == horizon_hours
        )
        cumulative = 0.0
        curve: list[tuple[datetime, float]] = []
        for trade in sorted(selected, key=lambda item: (item.exit_time, item.signal_time)):
            cumulative += trade.net_pnl
            curve.append((trade.exit_time, cumulative))
        return tuple(curve)


@dataclass(frozen=True, slots=True)
class _PerpetualMarket:
    market_time: datetime
    available_at: datetime
    asset: str
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


@dataclass(frozen=True, slots=True)
class _EntryRisk:
    scale: float
    gross_vega: float
    net_vega: float
    net_delta: float
    gross_notional: float
    scenario_loss: float


def _first_attr(obj: object, names: Sequence[str], default: object = None) -> object:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _is_perpetual(quote: QuoteSnapshot) -> bool:
    instrument_type = _first_attr(quote, ("instrument_type",), "")
    return str(instrument_type).strip().lower() in {
        "perpetual",
        "perp",
        "future_perpetual",
    }


def _option_map(quotes: Iterable[QuoteSnapshot]) -> dict[str, QuoteSnapshot]:
    """Keep the latest-arriving quote when a snapshot contains duplicates."""

    result: dict[str, QuoteSnapshot] = {}
    for quote in quotes:
        if _is_perpetual(quote):
            continue
        instrument = quote_instrument(quote)
        previous = result.get(instrument)
        if previous is None or quote_available_at(quote) > quote_available_at(previous):
            result[instrument] = quote
    return result


def _group_snapshots(
    quotes: Iterable[QuoteSnapshot],
) -> tuple[tuple[datetime, tuple[QuoteSnapshot, ...]], ...]:
    grouped: defaultdict[datetime, list[QuoteSnapshot]] = defaultdict(list)
    for quote in quotes:
        grouped[quote_snapshot_time(quote)].append(quote)
    return tuple(
        (timestamp, tuple(grouped[timestamp])) for timestamp in sorted(grouped)
    )


def _underlying_price(quote: QuoteSnapshot) -> float:
    raw = _first_attr(quote, ("underlying_price", "forward_price", "forward"))
    if raw is None:
        raise ValueError(f"{quote_instrument(quote)} has no underlying/forward price")
    value = float(str(raw))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{quote_instrument(quote)} has invalid underlying price")
    return value


def _quote_asset(quote: QuoteSnapshot) -> str:
    """Return the base asset used to select the matching inverse perpetual."""

    raw = _first_attr(quote, ("asset",))
    if raw is not None and str(raw).strip():
        return str(raw).strip().upper()
    instrument = quote_instrument(quote).upper()
    for separator in ("-", "_"):
        if separator in instrument:
            return instrument.split(separator, maxsplit=1)[0]
    raise ValueError(f"cannot infer the base asset for {instrument}")


def _inverse_perpetual_contract_size_usd(asset: str) -> float:
    """Return Deribit's USD contract size for inverse BTC/ETH perpetuals."""

    sizes = {"BTC": 10.0, "ETH": 1.0}
    try:
        return sizes[asset.upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported inverse perpetual asset: {asset}") from exc


def _perpetual_market(
    market_time: datetime,
    snapshot: Sequence[QuoteSnapshot],
    asset: str,
) -> _PerpetualMarket:
    normalized_asset = asset.upper()
    perpetuals = [
        quote
        for quote in snapshot
        if _is_perpetual(quote) and _quote_asset(quote) == normalized_asset
    ]
    if not perpetuals:
        raise ValueError(f"snapshot has no executable {normalized_asset} perpetual quote")
    quote = max(
        perpetuals,
        key=lambda item: (quote_available_at(item), quote_instrument(item)),
    )
    bid, ask = quote_bid_ask(quote)
    available_at = quote_available_at(quote)
    if available_at > market_time:
        raise ValueError("perpetual quote arrives after its market snapshot")
    return _PerpetualMarket(
        market_time,
        available_at,
        normalized_asset,
        bid,
        ask,
    )


def _perpetual_is_stale(
    market: _PerpetualMarket,
    max_age: timedelta,
) -> bool:
    return market.market_time - market.available_at > max_age


def _quote_greek(quote: QuoteSnapshot, name: str, fallback: float = 0.0) -> float:
    raw = _first_attr(quote, (name,), fallback)
    try:
        result = float(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{quote_instrument(quote)} has invalid {name}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{quote_instrument(quote)} has non-finite {name}")
    return result


def _inverse_delta(quote: QuoteSnapshot) -> float:
    """Return a base-coin delta proxy for an inverse option.

    Deribit reports the Black delta alongside a coin-denominated premium.  The
    net-transaction delta proxy is ``Black delta - option coin mark``.  A
    two-sided midpoint is used because this is a hedge size, not a fill price.
    """

    bid, ask = quote_bid_ask(quote)
    return _quote_greek(quote, "delta") - 0.5 * (bid + ask)


def _effective_delta(quote: QuoteSnapshot, inverse_options: bool) -> float:
    return _inverse_delta(quote) if inverse_options else _quote_greek(quote, "delta")


def _cash_vega(
    quote: QuoteSnapshot,
    underlying: float,
    inverse_options: bool,
) -> float:
    # The normalized research schema stores Black-76 vega in USD per unit of
    # volatility even when the executable premium itself is quoted in coin.
    # Keep the arguments explicit so a future native-coin Greek adapter cannot
    # silently change units.
    _ = underlying, inverse_options
    return abs(_quote_greek(quote, "vega"))


def _scenario_loss(
    quantities: Sequence[float],
    entry_quotes: Sequence[QuoteSnapshot],
    underlying: float,
    config: BacktestConfig,
) -> float:
    shock_s = config.risk_limits.scenario_underlying_shock * underlying
    shock_vol = config.risk_limits.scenario_volatility_shock
    multiplier = config.option_contract_multiplier
    scenario_pnls: list[float] = []
    for direction_s in (-1.0, 1.0):
        for direction_vol in (-1.0, 1.0):
            ds = direction_s * shock_s
            dvol = direction_vol * shock_vol
            pnl = 0.0
            for quantity, quote in zip(quantities, entry_quotes, strict=True):
                delta = _effective_delta(quote, config.inverse_options)
                gamma = _quote_greek(quote, "gamma")
                vega = _cash_vega(quote, underlying, config.inverse_options)
                approximation = delta * ds + 0.5 * gamma * ds * ds + vega * dvol
                pnl += quantity * multiplier * approximation
            scenario_pnls.append(pnl)
    return max(0.0, -min(scenario_pnls))


def _entry_risk(
    selection: PortfolioSelection,
    entry_quotes: Sequence[QuoteSnapshot],
    underlying: float,
    config: BacktestConfig,
) -> _EntryRisk | None:
    base_quantities = [leg.quantity for leg in selection.legs]
    multiplier = config.option_contract_multiplier
    gross_vega = sum(
        abs(quantity)
        * _cash_vega(quote, underlying, config.inverse_options)
        * multiplier
        for quantity, quote in zip(base_quantities, entry_quotes, strict=True)
    )
    gross_notional = sum(
        abs(quantity) * underlying * multiplier for quantity in base_quantities
    )
    max_position = max(abs(quantity) for quantity in base_quantities)
    scenario_loss = _scenario_loss(base_quantities, entry_quotes, underlying, config)
    limits = config.risk_limits
    ratios = [
        1.0,
        limits.max_gross_vega / gross_vega if gross_vega > 0.0 else 1.0,
        limits.max_gross_notional / gross_notional if gross_notional > 0.0 else 1.0,
        limits.max_abs_position_per_leg / max_position if max_position > 0.0 else 1.0,
        limits.max_scenario_loss / scenario_loss if scenario_loss > 0.0 else 1.0,
    ]
    scale = min(ratios)
    if not math.isfinite(scale):
        scale = 1.0
    scale = min(1.0, max(0.0, scale))
    if scale < limits.minimum_scale:
        return None

    quantities = [quantity * scale for quantity in base_quantities]
    net_vega = sum(
        quantity * _cash_vega(quote, underlying, config.inverse_options) * multiplier
        for quantity, quote in zip(quantities, entry_quotes, strict=True)
    )
    net_delta = sum(
        quantity * _effective_delta(quote, config.inverse_options) * multiplier
        for quantity, quote in zip(quantities, entry_quotes, strict=True)
    )
    return _EntryRisk(
        scale=scale,
        gross_vega=gross_vega * scale,
        net_vega=net_vega,
        net_delta=net_delta,
        gross_notional=gross_notional * scale,
        scenario_loss=scenario_loss * scale,
    )


def _execution_price(bid: float, ask: float, quantity: float, *, entering: bool) -> float:
    if quantity == 0.0:
        return 0.5 * (bid + ask)
    # A long buys at entry and sells at exit; a short does the reverse.
    buy = quantity > 0.0 if entering else quantity < 0.0
    return ask if buy else bid


def _funding_event_time(event: FundingRateSnapshot) -> datetime:
    raw = _first_attr(event, ("timestamp", "funding_time"))
    if not isinstance(raw, datetime):
        raise ValueError("funding observation has no valid timestamp")
    return raw


def _funding_event_rate(event: FundingRateSnapshot) -> float:
    raw = _first_attr(event, ("rate", "funding_rate"))
    if raw is None:
        raise ValueError("funding observation has no rate")
    rate = float(str(raw))
    if not math.isfinite(rate):
        raise ValueError("funding observation has a non-finite rate")
    return rate


def _funding_pnl(
    hedge_notional_usd: float,
    entry_market_time: datetime,
    exit_market_time: datetime,
    funding_rates: Sequence[FundingRateSnapshot],
    constant_rate_per_hour: float,
) -> float:
    if funding_rates:
        total_rate = sum(
            _funding_event_rate(event)
            for event in funding_rates
            if entry_market_time < _funding_event_time(event) <= exit_market_time
        )
    else:
        hours = (exit_market_time - entry_market_time).total_seconds() / 3600.0
        total_rate = constant_rate_per_hour * hours
    # Deribit expresses inverse-perpetual order size in USD.  At index X the
    # base-coin position is N/X, so a funding fraction r transfers r*N/X coin,
    # whose contemporaneous USD value is r*N.  Positive funding is paid by
    # longs (positive signed USD notional) and received by shorts.
    return -hedge_notional_usd * total_rate


def _exit_index(
    timestamps: Sequence[datetime],
    entry_index: int,
    horizon_hours: float,
    max_exit_delay: timedelta | None,
) -> int | None:
    target = timestamps[entry_index] + timedelta(hours=horizon_hours)
    index = bisect.bisect_left(timestamps, target, lo=entry_index + 1)
    if index >= len(timestamps):
        return None
    if max_exit_delay is not None and timestamps[index] - target > max_exit_delay:
        return None
    return index


def _evaluate_trade(
    selection: PortfolioSelection,
    entry_market_time: datetime,
    entry_snapshot: Sequence[QuoteSnapshot],
    exit_market_time: datetime,
    exit_snapshot: Sequence[QuoteSnapshot],
    horizon_hours: float,
    config: BacktestConfig,
    funding_rates: Sequence[FundingRateSnapshot],
) -> tuple[TradeResult | None, str | None]:
    entry_options = _option_map(entry_snapshot)
    exit_options = _option_map(exit_snapshot)
    instruments = [leg.instrument for leg in selection.legs]
    if any(instrument not in entry_options for instrument in instruments):
        return None, "missing_entry_leg"
    if any(instrument not in exit_options for instrument in instruments):
        return None, "missing_exit_leg"
    entry_quotes = [entry_options[instrument] for instrument in instruments]
    exit_quotes = [exit_options[instrument] for instrument in instruments]

    try:
        assets = {_quote_asset(quote) for quote in entry_quotes}
    except ValueError:
        return None, "missing_option_asset"
    if len(assets) != 1:
        return None, "cross_asset_option_portfolio"
    asset = next(iter(assets))

    if any(quote_available_at(quote) <= selection.decision_time for quote in entry_quotes):
        return None, "noncausal_entry_quote"
    if any(quote_expiry(quote) <= exit_market_time for quote in entry_quotes):
        return None, "expiry_before_exit"

    try:
        entry_perp = _perpetual_market(entry_market_time, entry_snapshot, asset)
    except ValueError:
        return None, "missing_or_invalid_entry_perpetual"
    try:
        exit_perp = _perpetual_market(exit_market_time, exit_snapshot, asset)
    except ValueError:
        return None, "missing_or_invalid_exit_perpetual"
    if _perpetual_is_stale(entry_perp, config.max_perpetual_quote_age):
        return None, "stale_entry_perpetual"
    if _perpetual_is_stale(exit_perp, config.max_perpetual_quote_age):
        return None, "stale_exit_perpetual"
    if entry_perp.available_at <= selection.decision_time:
        return None, "noncausal_entry_perpetual"

    underlying = entry_perp.mid
    try:
        risk = _entry_risk(selection, entry_quotes, underlying, config)
    except ValueError:
        return None, "invalid_entry_greeks"
    if risk is None:
        return None, "risk_scale_below_minimum"

    multiplier = config.option_contract_multiplier
    quantities = [leg.quantity * risk.scale for leg in selection.legs]
    executed_legs: list[ExecutedLeg] = []
    option_gross_pnl = 0.0
    option_turnover = 0.0
    option_fees = 0.0
    for quantity, entry_quote, exit_quote in zip(
        quantities, entry_quotes, exit_quotes, strict=True
    ):
        try:
            entry_bid, entry_ask = quote_bid_ask(entry_quote)
            exit_bid, exit_ask = quote_bid_ask(exit_quote)
        except ValueError:
            return None, "invalid_execution_market"
        entry_price = _execution_price(entry_bid, entry_ask, quantity, entering=True)
        exit_price = _execution_price(exit_bid, exit_ask, quantity, entering=False)
        try:
            entry_underlying = _underlying_price(entry_quote)
        except ValueError:
            entry_underlying = entry_perp.mid
        try:
            exit_underlying = _underlying_price(exit_quote)
        except ValueError:
            exit_underlying = exit_perp.mid
        absolute_contracts = abs(quantity) * multiplier
        if config.inverse_options:
            # Native coin P&L is marked into USD at liquidation.  This avoids
            # mixing the USD index with coin-denominated premium differences.
            option_gross_pnl += (
                quantity
                * multiplier
                * (exit_price - entry_price)
                * exit_underlying
            )
            entry_premium_usd = absolute_contracts * entry_price * entry_underlying
            exit_premium_usd = absolute_contracts * exit_price * exit_underlying
            option_turnover += entry_premium_usd + exit_premium_usd
            for premium_coin, index_price in (
                (entry_price, entry_underlying),
                (exit_price, exit_underlying),
            ):
                notional_fee = (
                    config.option_fee_rate * index_price * absolute_contracts
                )
                premium_cap = (
                    config.option_fee_cap_fraction
                    * premium_coin
                    * index_price
                    * absolute_contracts
                )
                option_fees += min(notional_fee, premium_cap)
        else:
            option_gross_pnl += quantity * multiplier * (exit_price - entry_price)
            premium_turnover = absolute_contracts * (entry_price + exit_price)
            option_turnover += premium_turnover
            option_fees += config.option_fee_rate * premium_turnover
        executed_legs.append(
            ExecutedLeg(
                instrument=quote_instrument(entry_quote),
                quantity=quantity,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_mid=0.5 * (entry_bid + entry_ask),
                exit_mid=0.5 * (exit_bid + exit_ask),
                entry_underlying=entry_underlying,
                exit_underlying=exit_underlying,
                entry_vega=abs(_quote_greek(entry_quote, "vega")),
                entry_delta=_effective_delta(entry_quote, config.inverse_options),
            )
        )

    # Option NTD is measured in base coin.  Deribit inverse perpetual order
    # amounts are instead USD notionals: BTC-PERPETUAL contracts are $10 and
    # ETH-PERPETUAL contracts are $1.  Size the signed USD order so that N/P0
    # equals the desired base-coin hedge at the executable entry price P0.
    hedge_quantity = -risk.net_delta
    hedge_entry_price = _execution_price(
        entry_perp.bid, entry_perp.ask, hedge_quantity, entering=True
    )
    hedge_notional_usd = hedge_quantity * hedge_entry_price
    contract_size_usd = _inverse_perpetual_contract_size_usd(entry_perp.asset)
    hedge_contracts = hedge_notional_usd / contract_size_usd
    hedge_exit_price = _execution_price(
        exit_perp.bid, exit_perp.ask, hedge_notional_usd, entering=False
    )
    # A signed inverse-future USD amount N realizes
    # N * (1/P0 - 1/P1) in settlement coin.  Reports use USD, so mark that coin
    # cash flow at the contemporaneous exit midpoint (our index proxy).
    hedge_pnl_coin = hedge_notional_usd * (
        1.0 / hedge_entry_price - 1.0 / hedge_exit_price
    )
    hedge_gross_pnl = hedge_pnl_coin * exit_perp.mid
    perpetual_turnover = 2.0 * abs(hedge_notional_usd)

    option_fees += (
        config.option_fee_per_contract_per_side
        * 2.0
        * sum(abs(quantity) for quantity in quantities)
    )
    # Deribit charges each inverse-perpetual execution as a fraction of the
    # fixed USD order amount (debited in settlement coin at the fill price).
    # Its contemporaneous USD equivalent is therefore f*|N| on each side.
    perpetual_fees = config.perpetual_fee_rate * perpetual_turnover
    funding_pnl = _funding_pnl(
        hedge_notional_usd,
        entry_market_time,
        exit_market_time,
        funding_rates,
        config.constant_funding_rate_per_hour,
    )
    net_pnl = (
        option_gross_pnl
        + hedge_gross_pnl
        + funding_pnl
        - option_fees
        - perpetual_fees
    )
    entry_time = max(
        [entry_perp.available_at, *(quote_available_at(q) for q in entry_quotes)]
    )
    exit_time = max(
        [exit_perp.available_at, *(quote_available_at(q) for q in exit_quotes)]
    )
    realized_hours = (exit_market_time - entry_market_time).total_seconds() / 3600.0
    return (
        TradeResult(
            signal_time=selection.snapshot_time,
            decision_time=selection.decision_time,
            entry_market_time=entry_market_time,
            entry_time=entry_time,
            exit_market_time=exit_market_time,
            exit_time=exit_time,
            requested_horizon_hours=horizon_hours,
            realized_holding_hours=realized_hours,
            expiry=selection.expiry,
            legs=tuple(executed_legs),
            hedge_quantity=hedge_quantity,
            hedge_notional_usd=hedge_notional_usd,
            hedge_contracts=hedge_contracts,
            hedge_entry_price=hedge_entry_price,
            hedge_exit_price=hedge_exit_price,
            option_gross_pnl=option_gross_pnl,
            hedge_gross_pnl=hedge_gross_pnl,
            option_fees=option_fees,
            perpetual_fees=perpetual_fees,
            funding_pnl=funding_pnl,
            net_pnl=net_pnl,
            turnover=option_turnover + perpetual_turnover,
            gross_vega=risk.gross_vega,
            net_vega=risk.net_vega,
            entry_net_delta_before_hedge=risk.net_delta,
            gross_notional=risk.gross_notional,
            scenario_loss=risk.scenario_loss,
            risk_scale=risk.scale,
            signal_strength=selection.signal_strength,
        ),
        None,
    )


def run_backtest(
    quotes: Iterable[QuoteSnapshot],
    model_prices: ModelPriceResolver,
    config: BacktestConfig | None = None,
    *,
    perpetual_quotes: Iterable[QuoteSnapshot] = (),
    funding_rates: Iterable[FundingRateSnapshot] = (),
) -> BacktestResult:
    """Run a causal surface-residual backtest on a flat quote stream.

    ``model_prices`` must represent a surface fit available at each signal
    snapshot.  It may be a callback or a mapping keyed by
    ``(snapshot_time, instrument)`` (``instrument`` alone is also accepted for
    static/synthetic tests).  Quotes are grouped by ``snapshot_time`` and sorted
    internally; execution is always the immediately following market snapshot.
    """

    cfg = config or BacktestConfig()
    all_quotes = [*quotes, *perpetual_quotes]
    snapshots = _group_snapshots(all_quotes)
    timestamps = [timestamp for timestamp, _ in snapshots]
    funding = tuple(sorted(funding_rates, key=_funding_event_time))
    skips: Counter[str] = Counter()
    trades: list[TradeResult] = []
    signal_snapshots = 0
    accepted_signals = 0
    rejected_signal_quotes = 0
    candidate_portfolios = 0
    risk_scaled_trades = 0

    for signal_index in range(max(0, len(snapshots) - 1)):
        entry_index = signal_index + 1
        horizon_exits = [
            (horizon, _exit_index(timestamps, entry_index, horizon, cfg.max_exit_delay))
            for horizon in cfg.horizons_hours
        ]
        horizon_exits = [(h, i) for h, i in horizon_exits if i is not None]
        if not horizon_exits:
            skips["no_exit_snapshot_for_horizon"] += 1
            continue

        _, signal_snapshot = snapshots[signal_index]
        signal_options = tuple(_option_map(signal_snapshot).values())
        if not signal_options:
            skips["empty_option_signal_snapshot"] += 1
            continue
        signal_snapshots += 1
        batch = compute_residual_signals(
            signal_options,
            model_prices,
            cfg.signal,
            strict=False,
        )
        accepted_signals += len(batch.signals)
        rejected_signal_quotes += batch.diagnostics.rejected_quotes
        selections = select_vega_neutral_portfolios(batch.signals, cfg.selection)
        candidate_portfolios += len(selections)
        if not selections:
            skips["no_two_sided_signal_portfolio"] += 1
            continue

        # The selector sorts strongest first within a snapshot.  Explicitly
        # filter to this signal time in case callers supplied unusual timestamps.
        selections = tuple(
            sorted(selections, key=lambda item: -item.signal_strength)
        )[: cfg.max_portfolios_per_snapshot]
        entry_market_time, entry_snapshot = snapshots[entry_index]
        if entry_market_time <= timestamps[signal_index]:
            raise ValueError("market snapshots must advance strictly in time")

        for selection in selections:
            for horizon, exit_index_or_none in horizon_exits:
                if exit_index_or_none is None:  # narrowed above; keeps typing explicit
                    continue
                exit_market_time, exit_snapshot = snapshots[exit_index_or_none]
                trade, reason = _evaluate_trade(
                    selection,
                    entry_market_time,
                    entry_snapshot,
                    exit_market_time,
                    exit_snapshot,
                    horizon,
                    cfg,
                    funding,
                )
                if trade is None:
                    skips[reason or "unknown_trade_rejection"] += 1
                    continue
                if trade.entry_market_time <= trade.signal_time:
                    raise AssertionError("same-timestamp execution is forbidden")
                if trade.entry_time <= trade.decision_time:
                    raise AssertionError("entry quote must arrive after the decision")
                if trade.risk_scale < 1.0 - 1e-12:
                    risk_scaled_trades += 1
                trades.append(trade)

    trades.sort(
        key=lambda item: (
            item.signal_time,
            item.requested_horizon_hours,
            item.expiry,
        )
    )
    diagnostics = BacktestDiagnostics(
        market_snapshots=len(snapshots),
        signal_snapshots_evaluated=signal_snapshots,
        accepted_signals=accepted_signals,
        rejected_signal_quotes=rejected_signal_quotes,
        candidate_portfolios=candidate_portfolios,
        completed_trades=len(trades),
        risk_scaled_trades=risk_scaled_trades,
        skip_counts=tuple(sorted(skips.items())),
    )
    return BacktestResult(tuple(trades), diagnostics)


__all__ = [
    "BacktestConfig",
    "BacktestDiagnostics",
    "BacktestResult",
    "ExecutedLeg",
    "FundingRateSnapshot",
    "RiskLimits",
    "TradeResult",
    "run_backtest",
]
