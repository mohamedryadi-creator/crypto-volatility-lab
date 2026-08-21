"""Typed, dependency-free data contracts used by the data pipeline.

The project deliberately keeps the normalized quote schema independent from
``pandas`` and ``pyarrow``.  This makes the ingestion path usable on a small
laptop and gives the research/backtest layers a stable, testable interface.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from math import isfinite
from typing import Any, ClassVar, Literal

InstrumentType = Literal["option", "perpetual"]
OptionType = Literal["call", "put"]


def _as_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class QuoteSnapshot:
    """A normalized, point-in-time best quote.

    ``timestamp`` is the exchange event time and ``local_timestamp`` is the
    collection/arrival time.  Research code must use ``local_timestamp`` to
    decide when an observation became available.  ``snapshot_time`` is set by
    the resampler and denotes the as-of grid point; the source timestamps are
    intentionally preserved for staleness diagnostics.

    Deribit option prices are denominated in the underlying asset, exactly as
    supplied by Tardis.  ``underlying_price`` remains USD-index denominated.
    Consumers converting option P&L to USD therefore multiply option prices by
    the contemporaneous underlying/index price.
    """

    exchange: str
    symbol: str
    asset: str
    instrument_type: InstrumentType
    timestamp: datetime
    local_timestamp: datetime
    bid_price: float | None
    ask_price: float | None
    bid_amount: float | None = None
    ask_amount: float | None = None
    option_type: OptionType | None = None
    strike_price: float | None = None
    expiration: datetime | None = None
    open_interest: float | None = None
    bid_iv: float | None = None
    ask_iv: float | None = None
    mark_price: float | None = None
    mark_iv: float | None = None
    underlying_index: str | None = None
    underlying_price: float | None = None
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    rho: float | None = None
    snapshot_time: datetime | None = None
    source: str = "tardis"

    DATETIME_FIELDS: ClassVar[tuple[str, ...]] = (
        "timestamp",
        "local_timestamp",
        "expiration",
        "snapshot_time",
    )

    def __post_init__(self) -> None:
        if not self.exchange:
            raise ValueError("exchange must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.asset:
            raise ValueError("asset must not be empty")
        if self.instrument_type not in ("option", "perpetual"):
            raise ValueError("instrument_type must be 'option' or 'perpetual'")
        if self.instrument_type == "option":
            if self.option_type not in ("call", "put"):
                raise ValueError("an option snapshot requires option_type")
            if self.strike_price is None or self.strike_price <= 0:
                raise ValueError("an option snapshot requires a positive strike_price")
            if self.expiration is None:
                raise ValueError("an option snapshot requires expiration")

        object.__setattr__(self, "timestamp", _as_utc(self.timestamp, "timestamp"))
        object.__setattr__(
            self,
            "local_timestamp",
            _as_utc(self.local_timestamp, "local_timestamp"),
        )
        if self.expiration is not None:
            object.__setattr__(
                self, "expiration", _as_utc(self.expiration, "expiration")
            )
        if self.snapshot_time is not None:
            object.__setattr__(
                self,
                "snapshot_time",
                _as_utc(self.snapshot_time, "snapshot_time"),
            )

        numeric_names = (
            "bid_price",
            "ask_price",
            "bid_amount",
            "ask_amount",
            "strike_price",
            "open_interest",
            "bid_iv",
            "ask_iv",
            "mark_price",
            "mark_iv",
            "underlying_price",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
        )
        for name in numeric_names:
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite when provided")

        for name in (
            "bid_price",
            "ask_price",
            "bid_amount",
            "ask_amount",
            "open_interest",
            "bid_iv",
            "ask_iv",
            "mark_price",
            "mark_iv",
            "underlying_price",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def available_at(self) -> datetime:
        """When the quote was available to a historical strategy."""

        return self.local_timestamp

    @property
    def key(self) -> tuple[str, str]:
        return (self.exchange, self.symbol)

    @property
    def mid_price(self) -> float | None:
        if self.bid_price is None or self.ask_price is None:
            return None
        return 0.5 * (self.bid_price + self.ask_price)

    @property
    def spread(self) -> float | None:
        if self.bid_price is None or self.ask_price is None:
            return None
        return self.ask_price - self.bid_price

    @property
    def relative_spread(self) -> float | None:
        mid = self.mid_price
        spread = self.spread
        if mid is None or spread is None or mid <= 0:
            return None
        return spread / mid

    @property
    def dte_days(self) -> float | None:
        if self.expiration is None:
            return None
        as_of = self.snapshot_time or self.local_timestamp
        return (self.expiration - as_of).total_seconds() / 86_400.0

    def at_grid(self, snapshot_time: datetime) -> QuoteSnapshot:
        """Return this observation tagged with a backward-looking as-of time."""

        snapshot_time = _as_utc(snapshot_time, "snapshot_time")
        if snapshot_time < self.local_timestamp:
            raise ValueError("snapshot_time cannot precede quote availability")
        return replace(self, snapshot_time=snapshot_time)

    def to_record(self) -> dict[str, Any]:
        """Return a flat record suitable for CSV/Parquet serialization."""

        record: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, datetime):
                record[field.name] = value.isoformat().replace("+00:00", "Z")
            elif value is None:
                record[field.name] = ""
            else:
                record[field.name] = value
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> QuoteSnapshot:
        """Rehydrate a snapshot written by :meth:`to_record`."""

        values = dict(record)
        for name in cls.DATETIME_FIELDS:
            value = values.get(name)
            if value in (None, ""):
                values[name] = None
            elif not isinstance(value, datetime):
                values[name] = datetime.fromisoformat(str(value).replace("Z", "+00:00"))

        float_names = {
            "bid_price",
            "ask_price",
            "bid_amount",
            "ask_amount",
            "strike_price",
            "open_interest",
            "bid_iv",
            "ask_iv",
            "mark_price",
            "mark_iv",
            "underlying_price",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
        }
        for name in float_names:
            value = values.get(name)
            values[name] = None if value in (None, "") else float(value)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class DataFilterConfig:
    """Point-in-time liquidity and data-quality filters.

    Any threshold may be disabled with ``None``.  Missing option fields fail a
    filter when the corresponding threshold is enabled; this prevents silent
    selection of incomplete contracts.
    """

    max_feed_lag_seconds: float | None = 60.0
    max_quote_age_seconds: float | None = 15.0 * 60.0
    max_relative_spread: float | None = 0.50
    min_abs_delta: float | None = 0.05
    max_abs_delta: float | None = 0.95
    min_dte_days: float | None = 1.0
    max_dte_days: float | None = 180.0
    min_open_interest: float | None = 1.0
    require_two_sided: bool = True
    allow_crossed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_feed_lag_seconds",
            "max_quote_age_seconds",
            "max_relative_spread",
            "min_abs_delta",
            "max_abs_delta",
            "min_dte_days",
            "max_dte_days",
            "min_open_interest",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")
        if (
            self.min_abs_delta is not None
            and self.max_abs_delta is not None
            and self.min_abs_delta > self.max_abs_delta
        ):
            raise ValueError("min_abs_delta cannot exceed max_abs_delta")
        if (
            self.min_dte_days is not None
            and self.max_dte_days is not None
            and self.min_dte_days > self.max_dte_days
        ):
            raise ValueError("min_dte_days cannot exceed max_dte_days")
