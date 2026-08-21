"""Surface-residual signals and vega-neutral option portfolio construction.

The module is deliberately data-provider agnostic.  ``QuoteSnapshot`` is a
structural protocol matching the public fields used by the engine; callers may
pass the project's concrete data model or any compatible object.

Sign convention
---------------
``residual_price = observed_mid - fitted_surface_price``.  A negative residual
is therefore *cheap* (a long candidate), while a positive residual is *rich*
(a short candidate).  The score is expressed in half-spreads.  We compute it
through volatility units explicitly::

    executable_residual = observed_quote_edge - fitted_surface_price
    score = executable_residual / half_spread * reference_vega / vega

``executable_residual`` is zero whenever the model lies inside the bid/ask
interval.  ``reference_vega`` is the median valid vega in the same timestamp
and expiry, so low-vega wings cannot dominate simply because their dollar price
is small.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeAlias


class QuoteSnapshot(Protocol):
    """Minimum structural interface consumed by this module.

    The concrete data layer may expose additional fields.  A few compatibility
    fallbacks (``instrument``, ``bid``, ``ask``, ``expiry`` and ``strike``) are
    accepted internally, but the names below are the canonical interface.
    """

    symbol: str
    timestamp: datetime
    local_timestamp: datetime
    snapshot_time: datetime | None
    bid_price: float | None
    ask_price: float | None
    expiration: datetime | None
    strike_price: float | None
    option_type: str | None
    delta: float | None
    vega: float | None


ModelPriceResolver: TypeAlias = (
    Mapping[object, float] | Callable[[QuoteSnapshot], float]
)


@dataclass(frozen=True, slots=True)
class ResidualSignalConfig:
    """Validation and scaling rules for surface residuals."""

    min_vega: float = 1e-8
    min_half_spread: float = 1e-8
    min_bid: float = 0.0
    max_relative_spread: float | None = None
    max_abs_score: float | None = 25.0

    def __post_init__(self) -> None:
        if self.min_vega <= 0.0:
            raise ValueError("min_vega must be strictly positive")
        if self.min_half_spread <= 0.0:
            raise ValueError("min_half_spread must be strictly positive")
        if self.max_relative_spread is not None and self.max_relative_spread <= 0.0:
            raise ValueError("max_relative_spread must be positive when provided")
        if self.max_abs_score is not None and self.max_abs_score <= 0.0:
            raise ValueError("max_abs_score must be positive when provided")


@dataclass(frozen=True, slots=True)
class ResidualSignal:
    """A quote's deviation from an arbitrage-free fitted surface."""

    quote: QuoteSnapshot
    instrument: str
    snapshot_time: datetime
    available_at: datetime
    expiry: datetime
    option_type: str
    strike: float
    model_price: float
    mid_price: float
    half_spread: float
    vega: float
    delta: float
    residual_price: float
    executable_residual: float
    residual_vol: float
    half_spread_vol: float
    score: float

    @property
    def is_cheap(self) -> bool:
        return self.score < 0.0

    @property
    def is_rich(self) -> bool:
        return self.score > 0.0


@dataclass(frozen=True, slots=True)
class SignalDiagnostics:
    total_quotes: int
    accepted_quotes: int
    rejected_quotes: int
    rejection_counts: tuple[tuple[str, int], ...]

    def count(self, reason: str) -> int:
        return dict(self.rejection_counts).get(reason, 0)


@dataclass(frozen=True, slots=True)
class SignalBatch:
    signals: tuple[ResidualSignal, ...]
    diagnostics: SignalDiagnostics


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Rules for pairing cheap and rich options within one expiry."""

    minimum_abs_score: float = 1.0
    legs_per_side: int = 2
    target_gross_vega: float = 100.0
    max_abs_quantity_per_leg: float = math.inf

    def __post_init__(self) -> None:
        if self.minimum_abs_score < 0.0:
            raise ValueError("minimum_abs_score cannot be negative")
        if self.legs_per_side < 1:
            raise ValueError("legs_per_side must be at least one")
        if self.target_gross_vega <= 0.0:
            raise ValueError("target_gross_vega must be strictly positive")
        if self.max_abs_quantity_per_leg <= 0.0:
            raise ValueError("max_abs_quantity_per_leg must be strictly positive")


@dataclass(frozen=True, slots=True)
class PortfolioLeg:
    signal: ResidualSignal
    quantity: float

    @property
    def instrument(self) -> str:
        return self.signal.instrument

    @property
    def side(self) -> str:
        return "long" if self.quantity > 0.0 else "short"


@dataclass(frozen=True, slots=True)
class PortfolioSelection:
    """A same-expiry, long-cheap/short-rich portfolio."""

    snapshot_time: datetime
    decision_time: datetime
    expiry: datetime
    legs: tuple[PortfolioLeg, ...]
    gross_vega: float
    net_vega: float
    net_delta: float
    signal_strength: float

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("a portfolio selection must contain at least one leg")
        if any(leg.signal.expiry != self.expiry for leg in self.legs):
            raise ValueError("all portfolio legs must have the same expiry")
        if not any(leg.quantity > 0.0 for leg in self.legs):
            raise ValueError("portfolio must contain at least one long leg")
        if not any(leg.quantity < 0.0 for leg in self.legs):
            raise ValueError("portfolio must contain at least one short leg")


def _first_attr(obj: object, names: Sequence[str], default: object = None) -> object:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def quote_instrument(quote: QuoteSnapshot) -> str:
    value = _first_attr(quote, ("symbol", "instrument"))
    if value is None or not str(value):
        raise ValueError("quote is missing a non-empty symbol/instrument")
    return str(value)


def quote_snapshot_time(quote: QuoteSnapshot) -> datetime:
    value = _first_attr(quote, ("snapshot_time", "timestamp"))
    if not isinstance(value, datetime):
        raise ValueError(f"{quote_instrument(quote)} has no valid snapshot timestamp")
    return value


def quote_available_at(quote: QuoteSnapshot) -> datetime:
    value = _first_attr(quote, ("available_at", "local_timestamp", "timestamp"))
    if not isinstance(value, datetime):
        raise ValueError(f"{quote_instrument(quote)} has no valid availability timestamp")
    return value


def quote_expiry(quote: QuoteSnapshot) -> datetime:
    value = _first_attr(quote, ("expiration", "expiry"))
    if not isinstance(value, datetime):
        raise ValueError(f"{quote_instrument(quote)} has no valid option expiry")
    return value


def quote_bid_ask(quote: QuoteSnapshot) -> tuple[float, float]:
    bid = _first_attr(quote, ("bid_price", "bid"))
    ask = _first_attr(quote, ("ask_price", "ask"))
    if bid is None or ask is None:
        raise ValueError(f"{quote_instrument(quote)} has no two-sided market")
    try:
        bid_float, ask_float = float(str(bid)), float(str(ask))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{quote_instrument(quote)} has non-numeric bid/ask") from exc
    if not math.isfinite(bid_float) or not math.isfinite(ask_float):
        raise ValueError(f"{quote_instrument(quote)} has non-finite bid/ask")
    if bid_float < 0.0 or ask_float < bid_float:
        raise ValueError(f"{quote_instrument(quote)} has an invalid or crossed market")
    return bid_float, ask_float


def _resolve_model_price(
    quote: QuoteSnapshot,
    resolver: ModelPriceResolver,
) -> float:
    if callable(resolver):
        value = resolver(quote)
    else:
        instrument = quote_instrument(quote)
        snapshot_time = quote_snapshot_time(quote)
        candidate_keys = (
            (snapshot_time, instrument),
            (instrument, snapshot_time),
            instrument,
        )
        for key in candidate_keys:
            if key in resolver:
                value = resolver[key]
                break
        else:
            raise KeyError(
                f"missing model price for {instrument} at {snapshot_time.isoformat()}"
            )
    try:
        model_price = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"model price for {quote_instrument(quote)} is not numeric") from exc
    if not math.isfinite(model_price) or model_price < 0.0:
        raise ValueError(
            f"model price for {quote_instrument(quote)} must be finite and non-negative"
        )
    return model_price


def _signal_from_quote(
    quote: QuoteSnapshot,
    model_prices: ModelPriceResolver,
    config: ResidualSignalConfig,
) -> ResidualSignal:
    instrument = quote_instrument(quote)
    snapshot_time = quote_snapshot_time(quote)
    available_at = quote_available_at(quote)
    expiry = quote_expiry(quote)
    if expiry <= snapshot_time:
        raise ValueError(f"{instrument} is expired at the snapshot time")

    bid, ask = quote_bid_ask(quote)
    if bid < config.min_bid:
        raise ValueError(f"{instrument} bid is below min_bid")
    mid = 0.5 * (bid + ask)
    if mid <= 0.0:
        raise ValueError(f"{instrument} has a non-positive mid price")
    half_spread = 0.5 * (ask - bid)
    if config.max_relative_spread is not None:
        relative_spread = (ask - bid) / mid
        if relative_spread > config.max_relative_spread:
            raise ValueError(f"{instrument} exceeds max_relative_spread")

    vega_raw = _first_attr(quote, ("vega",))
    delta_raw = _first_attr(quote, ("delta",), 0.0)
    strike_raw = _first_attr(quote, ("strike_price", "strike"))
    option_type_raw = _first_attr(quote, ("option_type",))
    if vega_raw is None or strike_raw is None or option_type_raw is None:
        raise ValueError(f"{instrument} is missing vega, strike or option type")
    try:
        vega = abs(float(str(vega_raw)))
        delta = float(str(delta_raw))
        strike = float(str(strike_raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{instrument} has invalid Greeks or strike") from exc
    if not all(math.isfinite(x) for x in (vega, delta, strike)):
        raise ValueError(f"{instrument} has non-finite Greeks or strike")
    if vega < config.min_vega:
        raise ValueError(f"{instrument} vega is below min_vega")
    if strike <= 0.0:
        raise ValueError(f"{instrument} strike must be strictly positive")
    option_type = str(option_type_raw).strip().lower()
    if option_type in {"c", "call"}:
        option_type = "call"
    elif option_type in {"p", "put"}:
        option_type = "put"
    else:
        raise ValueError(f"{instrument} has unsupported option_type={option_type_raw!r}")

    model_price = _resolve_model_price(quote, model_prices)
    residual_price = mid - model_price
    residual_vol = residual_price / vega
    effective_half_spread = max(half_spread, config.min_half_spread)
    half_spread_vol = effective_half_spread / vega
    if model_price < bid:
        # Selling at the bid remains rich versus the fitted value.
        executable_residual = bid - model_price
    elif model_price > ask:
        # Buying at the ask remains cheap versus the fitted value.
        executable_residual = ask - model_price
    else:
        executable_residual = 0.0
    raw_score = executable_residual / effective_half_spread

    return ResidualSignal(
        quote=quote,
        instrument=instrument,
        snapshot_time=snapshot_time,
        available_at=available_at,
        expiry=expiry,
        option_type=option_type,
        strike=strike,
        model_price=model_price,
        mid_price=mid,
        half_spread=half_spread,
        vega=vega,
        delta=delta,
        residual_price=residual_price,
        executable_residual=executable_residual,
        residual_vol=residual_vol,
        half_spread_vol=half_spread_vol,
        # Cross-sectional reference-vega scaling is applied once all valid
        # quotes in this expiry snapshot are known.
        score=raw_score,
    )


def compute_residual_signals(
    quotes: Iterable[QuoteSnapshot],
    model_prices: ModelPriceResolver,
    config: ResidualSignalConfig | None = None,
    *,
    strict: bool = False,
) -> SignalBatch:
    """Compute executable-cost-normalized surface residuals.

    Invalid or incomplete quotes are counted and skipped by default because real
    option feeds routinely contain one-sided markets.  ``strict=True`` converts
    the first data-quality rejection into an explicit ``ValueError``.  A missing
    model-price key is always raised: silently dropping it could mask a surface
    calibration bug.
    """

    cfg = config or ResidualSignalConfig()
    accepted: list[ResidualSignal] = []
    rejection_counts: defaultdict[str, int] = defaultdict(int)
    total = 0
    for quote in quotes:
        total += 1
        try:
            accepted.append(_signal_from_quote(quote, model_prices, cfg))
        except KeyError:
            raise
        except ValueError as exc:
            if strict:
                raise
            reason = str(exc).split(" ", 1)[1] if " " in str(exc) else str(exc)
            rejection_counts[reason] += 1

    # Normalize by a same-snapshot, same-expiry reference vega.  This is fitted
    # without future observations and is therefore safe for point-in-time use.
    by_slice: defaultdict[tuple[datetime, datetime], list[ResidualSignal]] = defaultdict(list)
    for signal in accepted:
        by_slice[(signal.snapshot_time, signal.expiry)].append(signal)
    normalized: list[ResidualSignal] = []
    for group in by_slice.values():
        ordered_vegas = sorted(signal.vega for signal in group)
        count = len(ordered_vegas)
        if count % 2:
            reference_vega = ordered_vegas[count // 2]
        else:
            reference_vega = 0.5 * (
                ordered_vegas[count // 2 - 1] + ordered_vegas[count // 2]
            )
        for signal in group:
            raw_score = signal.score * reference_vega / signal.vega
            if cfg.max_abs_score is None:
                score = raw_score
            else:
                score = max(-cfg.max_abs_score, min(cfg.max_abs_score, raw_score))
            normalized.append(
                ResidualSignal(
                    quote=signal.quote,
                    instrument=signal.instrument,
                    snapshot_time=signal.snapshot_time,
                    available_at=signal.available_at,
                    expiry=signal.expiry,
                    option_type=signal.option_type,
                    strike=signal.strike,
                    model_price=signal.model_price,
                    mid_price=signal.mid_price,
                    half_spread=signal.half_spread,
                    vega=signal.vega,
                    delta=signal.delta,
                    residual_price=signal.residual_price,
                    executable_residual=signal.executable_residual,
                    residual_vol=signal.residual_vol,
                    half_spread_vol=signal.half_spread_vol,
                    score=score,
                )
            )
    accepted = normalized
    accepted.sort(key=lambda item: (item.snapshot_time, item.expiry, item.instrument))
    diagnostics = SignalDiagnostics(
        total_quotes=total,
        accepted_quotes=len(accepted),
        rejected_quotes=total - len(accepted),
        rejection_counts=tuple(sorted(rejection_counts.items())),
    )
    return SignalBatch(tuple(accepted), diagnostics)


def select_vega_neutral_portfolios(
    signals: Iterable[ResidualSignal],
    config: SelectionConfig | None = None,
) -> tuple[PortfolioSelection, ...]:
    """Build same-expiry long-cheap / short-rich vega-neutral portfolios.

    Each side receives the same aggregate absolute vega.  Within a side, vega
    is allocated equally across the selected legs.  If a per-leg quantity cap
    binds, both sides are scaled to the common feasible vega so neutrality is
    preserved rather than clipped asymmetrically.
    """

    cfg = config or SelectionConfig()
    grouped: defaultdict[tuple[datetime, datetime], list[ResidualSignal]] = defaultdict(list)
    for signal in signals:
        grouped[(signal.snapshot_time, signal.expiry)].append(signal)

    selections: list[PortfolioSelection] = []
    for (snapshot_time, expiry), group in grouped.items():
        cheap = sorted(
            (s for s in group if s.score <= -cfg.minimum_abs_score),
            key=lambda s: (s.score, s.instrument),
        )[: cfg.legs_per_side]
        rich = sorted(
            (s for s in group if s.score >= cfg.minimum_abs_score),
            key=lambda s: (-s.score, s.instrument),
        )[: cfg.legs_per_side]
        if not cheap or not rich:
            continue

        target_side_vega = 0.5 * cfg.target_gross_vega
        if math.isfinite(cfg.max_abs_quantity_per_leg):
            cheap_capacity = min(
                cfg.max_abs_quantity_per_leg * s.vega * len(cheap) for s in cheap
            )
            rich_capacity = min(
                cfg.max_abs_quantity_per_leg * s.vega * len(rich) for s in rich
            )
            target_side_vega = min(target_side_vega, cheap_capacity, rich_capacity)
        if target_side_vega <= 0.0:
            continue

        legs: list[PortfolioLeg] = []
        for signal in cheap:
            quantity = target_side_vega / (len(cheap) * signal.vega)
            legs.append(PortfolioLeg(signal, quantity))
        for signal in rich:
            quantity = -target_side_vega / (len(rich) * signal.vega)
            legs.append(PortfolioLeg(signal, quantity))

        gross_vega = sum(abs(leg.quantity) * leg.signal.vega for leg in legs)
        net_vega = sum(leg.quantity * leg.signal.vega for leg in legs)
        net_delta = sum(leg.quantity * leg.signal.delta for leg in legs)
        strength = sum(abs(leg.signal.score) for leg in legs) / len(legs)
        selections.append(
            PortfolioSelection(
                snapshot_time=snapshot_time,
                decision_time=max(leg.signal.available_at for leg in legs),
                expiry=expiry,
                legs=tuple(legs),
                gross_vega=gross_vega,
                net_vega=net_vega,
                net_delta=net_delta,
                signal_strength=strength,
            )
        )

    selections.sort(
        key=lambda item: (
            item.snapshot_time,
            -item.signal_strength,
            item.expiry,
        )
    )
    return tuple(selections)


def select_vega_neutral_portfolio(
    signals: Iterable[ResidualSignal],
    config: SelectionConfig | None = None,
) -> PortfolioSelection | None:
    """Return the strongest eligible portfolio from one signal snapshot."""

    portfolios = select_vega_neutral_portfolios(signals, config)
    if not portfolios:
        return None
    return max(portfolios, key=lambda item: item.signal_strength)


__all__ = [
    "ModelPriceResolver",
    "PortfolioLeg",
    "PortfolioSelection",
    "QuoteSnapshot",
    "ResidualSignal",
    "ResidualSignalConfig",
    "SelectionConfig",
    "SignalBatch",
    "SignalDiagnostics",
    "compute_residual_signals",
    "quote_available_at",
    "quote_bid_ask",
    "quote_expiry",
    "quote_instrument",
    "quote_snapshot_time",
    "select_vega_neutral_portfolio",
    "select_vega_neutral_portfolios",
]
