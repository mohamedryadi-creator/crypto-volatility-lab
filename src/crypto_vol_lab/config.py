"""Typed configuration loaded from TOML using only the Python standard library."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DataConfig:
    exchange: str = "deribit"
    assets: tuple[str, ...] = ("BTC", "ETH")
    start: str = "2020-03-01"
    end: str = "2026-08-01"
    interval_minutes: int = 15
    max_spread_fraction: float = 0.35
    min_days_to_expiry: float = 7.0
    max_days_to_expiry: float = 120.0

    def __post_init__(self) -> None:
        if self.exchange != "deribit":
            raise ValueError("only the deribit exchange is supported")
        if not self.assets or len(set(self.assets)) != len(self.assets):
            raise ValueError("assets must be a non-empty tuple without duplicates")
        unsupported = set(self.assets) - {"BTC", "ETH"}
        if unsupported:
            raise ValueError(f"unsupported assets: {sorted(unsupported)}")
        start = date.fromisoformat(self.start)
        end = date.fromisoformat(self.end)
        if end < start:
            raise ValueError("data end date cannot precede the start date")
        if self.interval_minutes not in (15, 60):
            raise ValueError("interval_minutes must be either 15 or 60")
        if self.max_spread_fraction <= 0.0:
            raise ValueError("max_spread_fraction must be strictly positive")
        if self.min_days_to_expiry < 0.0:
            raise ValueError("min_days_to_expiry cannot be negative")
        if self.max_days_to_expiry <= self.min_days_to_expiry:
            raise ValueError("max_days_to_expiry must exceed min_days_to_expiry")


@dataclass(frozen=True, slots=True)
class SurfaceConfig:
    min_quotes_per_expiry: int = 8
    log_moneyness_limit: float = 1.0
    robust_loss: str = "soft_l1"
    arbitrage_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.min_quotes_per_expiry < 5:
            raise ValueError("min_quotes_per_expiry must be at least five")
        if self.log_moneyness_limit <= 0.0:
            raise ValueError("log_moneyness_limit must be strictly positive")
        if self.robust_loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise ValueError("unsupported robust_loss")
        if self.arbitrage_tolerance < 0.0:
            raise ValueError("arbitrage_tolerance cannot be negative")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    entry_lag_minutes: int = 15
    holding_periods_minutes: tuple[int, ...] = (60, 240)
    signal_threshold: float = 1.5
    option_fee_rate: float = 0.0003
    perpetual_taker_fee_rate: float = 0.00035
    funding_rate_8h: float = 0.0
    max_gross_vega: float = 1_000.0
    max_scenario_loss: float = 1_000.0

    def __post_init__(self) -> None:
        if self.entry_lag_minutes <= 0:
            raise ValueError("entry_lag_minutes must be strictly positive")
        if not self.holding_periods_minutes or any(
            value <= 0 for value in self.holding_periods_minutes
        ):
            raise ValueError("holding_periods_minutes must contain positive values")
        if len(set(self.holding_periods_minutes)) != len(self.holding_periods_minutes):
            raise ValueError("holding_periods_minutes cannot contain duplicates")
        if self.signal_threshold < 0.0:
            raise ValueError("signal_threshold cannot be negative")
        if self.option_fee_rate < 0.0 or self.perpetual_taker_fee_rate < 0.0:
            raise ValueError("fee rates cannot be negative")
        if self.max_gross_vega <= 0.0 or self.max_scenario_loss <= 0.0:
            raise ValueError("risk limits must be strictly positive")


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    seed: int = 7
    data: DataConfig = field(default_factory=DataConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"TOML section [{name}] must be a table")
    return value


def load_config(path: str | Path) -> ResearchConfig:
    """Load a research config and reject misspelled top-level sections."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    allowed = {"research", "data", "surface", "backtest"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration section(s): {sorted(unknown)}")

    research = _section(raw, "research")
    data = _section(raw, "data")
    surface = _section(raw, "surface")
    backtest = _section(raw, "backtest")

    if "assets" in data:
        data["assets"] = tuple(str(asset) for asset in data["assets"])
    if "holding_periods_minutes" in backtest:
        backtest["holding_periods_minutes"] = tuple(
            int(value) for value in backtest["holding_periods_minutes"]
        )
    return ResearchConfig(
        seed=int(research.get("seed", 7)),
        data=DataConfig(**data),
        surface=SurfaceConfig(**surface),
        backtest=BacktestConfig(**backtest),
    )
