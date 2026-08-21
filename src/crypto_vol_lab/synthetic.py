"""Deterministic, arbitrage-free synthetic BTC/ETH option-chain fixtures.

Fixtures are generated from a mixture of Black lognormal distributions.  A
positive mixture of risk-neutral distributions preserves monotonicity and
convexity in strike; keeping each component's forward fixed and total variance
increasing with maturity also avoids calendar arbitrage.  The resulting smile
is more realistic than a flat-volatility toy while remaining dependency-free.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from .data import write_partitioned
from .data_models import QuoteSnapshot

_ASSET_PARAMETERS = {
    "BTC": {
        "spot": 30_000.0,
        "annual_vol": 0.62,
        "perp_half_spread_bps": 0.45,
        "strike_step": 500.0,
    },
    "ETH": {
        "spot": 1_900.0,
        "annual_vol": 0.72,
        "perp_half_spread_bps": 0.65,
        "strike_step": 50.0,
    },
}


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _black_price(
    forward: float,
    strike: float,
    maturity_years: float,
    volatility: float,
    option_type: Literal["call", "put"],
) -> float:
    if maturity_years <= 0 or volatility <= 0:
        intrinsic = forward - strike if option_type == "call" else strike - forward
        return max(intrinsic, 0.0)
    root_t = math.sqrt(maturity_years)
    d1 = (
        math.log(forward / strike) + 0.5 * volatility * volatility * maturity_years
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    if option_type == "call":
        return forward * _normal_cdf(d1) - strike * _normal_cdf(d2)
    return strike * _normal_cdf(-d2) - forward * _normal_cdf(-d1)


def _mixture_price(
    forward: float,
    strike: float,
    maturity_years: float,
    base_volatility: float,
    option_type: Literal["call", "put"],
) -> float:
    # Both components have the same mean/forward.  Their positive mixture is a
    # valid risk-neutral distribution and produces smooth smile curvature.
    weight = 0.68
    calm_volatility = 0.76 * base_volatility
    tail_volatility = 1.42 * base_volatility
    return weight * _black_price(
        forward, strike, maturity_years, calm_volatility, option_type
    ) + (1.0 - weight) * _black_price(
        forward, strike, maturity_years, tail_volatility, option_type
    )


def _implied_volatility(
    price: float,
    forward: float,
    strike: float,
    maturity_years: float,
    option_type: Literal["call", "put"],
) -> float:
    intrinsic = max(
        forward - strike if option_type == "call" else strike - forward,
        0.0,
    )
    target = min(max(price, intrinsic), forward if option_type == "call" else strike)
    lower, upper = 1e-4, 4.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        model_price = _black_price(
            forward, strike, maturity_years, midpoint, option_type
        )
        if model_price < target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _greeks(
    forward: float,
    strike: float,
    maturity_years: float,
    volatility: float,
    option_type: Literal["call", "put"],
) -> tuple[float, float, float, float, float]:
    root_t = math.sqrt(max(maturity_years, 1e-12))
    d1 = (
        math.log(forward / strike) + 0.5 * volatility * volatility * maturity_years
    ) / (volatility * root_t)
    density = _normal_pdf(d1)
    delta = _normal_cdf(d1) if option_type == "call" else _normal_cdf(d1) - 1.0
    gamma = density / (forward * volatility * root_t)
    # The normalized schema stores vega per unit volatility; raw Deribit vega
    # is converted from its displayed per-percentage-point convention on input.
    vega = forward * density * root_t
    theta = -forward * density * volatility / (2.0 * root_t * 365.0)
    rho = 0.0
    return delta, gamma, vega, theta, rho


def _expiry_token(expiration: datetime) -> str:
    return expiration.strftime("%d%b%y").upper().lstrip("0")


def _fixed_strikes(
    spot: float, moneyness: Sequence[float], strike_step: float
) -> tuple[float, ...]:
    rounded = {
        max(strike_step, round((spot * ratio) / strike_step) * strike_step)
        for ratio in moneyness
    }
    return tuple(sorted(rounded))


def generate_synthetic_snapshots(
    *,
    start: datetime | None = None,
    periods: int = 8,
    interval_minutes: Literal[15, 60] = 15,
    assets: Sequence[str] = ("BTC", "ETH"),
    expiries_days: Sequence[int] = (7, 30, 90),
    moneyness: Sequence[float] = (0.70, 0.85, 1.00, 1.15, 1.30),
    seed: int = 7,
) -> Iterator[QuoteSnapshot]:
    """Yield a small realistic chain plus hedge instrument at each grid point.

    Prices are deterministic for a fixed seed.  Option marks are denominated in
    BTC/ETH, matching Deribit; perpetual quotes and ``underlying_price`` are USD.
    """

    if periods < 1:
        raise ValueError("periods must be positive")
    if interval_minutes not in (15, 60):
        raise ValueError("interval_minutes must be either 15 or 60")
    if any(days <= 0 for days in expiries_days):
        raise ValueError("expiries_days must all be positive")
    if any(ratio <= 0 for ratio in moneyness):
        raise ValueError("moneyness values must all be positive")

    if start is None:
        start = datetime(2023, 6, 1, 0, 0, tzinfo=UTC)
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("start must be timezone-aware")
    start = start.astimezone(UTC)

    requested_assets = tuple(asset.upper() for asset in assets)
    unsupported = set(requested_assets) - set(_ASSET_PARAMETERS)
    if unsupported:
        raise ValueError(f"unsupported synthetic assets: {sorted(unsupported)}")

    generator = random.Random(seed)
    spots = {
        asset: float(_ASSET_PARAMETERS[asset]["spot"])
        for asset in requested_assets
    }
    strikes = {
        asset: _fixed_strikes(
            spots[asset],
            moneyness,
            float(_ASSET_PARAMETERS[asset]["strike_step"]),
        )
        for asset in requested_assets
    }
    expirations = tuple(
        # A fixed 08:00 UTC expiry mirrors Deribit conventions.
        (start + timedelta(days=days)).replace(hour=8, minute=0, second=0, microsecond=0)
        for days in expiries_days
    )

    dt_years = interval_minutes / (365.0 * 24.0 * 60.0)
    step = timedelta(minutes=interval_minutes)
    for period in range(periods):
        grid_time = start + period * step
        common_shock = generator.gauss(0.0, 1.0) if period else 0.0
        for asset in requested_assets:
            params = _ASSET_PARAMETERS[asset]
            annual_volatility = float(params["annual_vol"])
            if period:
                idiosyncratic_shock = generator.gauss(0.0, 1.0)
                correlated_shock = (
                    0.78 * common_shock
                    + math.sqrt(1.0 - 0.78**2) * idiosyncratic_shock
                )
                spots[asset] *= math.exp(
                    -0.5 * annual_volatility**2 * dt_years
                    + annual_volatility * math.sqrt(dt_years) * correlated_shock
                )
            spot = spots[asset]
            event_time = grid_time - timedelta(milliseconds=12)

            half_spread = (
                spot * float(params["perp_half_spread_bps"]) / 10_000.0
            )
            yield QuoteSnapshot(
                exchange="deribit",
                symbol=f"{asset}-PERPETUAL",
                asset=asset,
                instrument_type="perpetual",
                timestamp=event_time,
                local_timestamp=grid_time,
                bid_price=spot - half_spread,
                ask_price=spot + half_spread,
                bid_amount=25.0 if asset == "BTC" else 350.0,
                ask_amount=23.0 if asset == "BTC" else 330.0,
                snapshot_time=grid_time,
                source="synthetic_mixture_black",
            )

            for expiration in expirations:
                maturity_years = (expiration - grid_time).total_seconds() / (
                    365.0 * 86_400.0
                )
                if maturity_years <= 0:
                    continue
                for strike in strikes[asset]:
                    for option_type in ("call", "put"):
                        usd_mark = _mixture_price(
                            spot,
                            strike,
                            maturity_years,
                            annual_volatility,
                            option_type,
                        )
                        mark_price = usd_mark / spot
                        mark_iv = _implied_volatility(
                            usd_mark,
                            spot,
                            strike,
                            maturity_years,
                            option_type,
                        )
                        delta, gamma, vega, theta, rho = _greeks(
                            spot,
                            strike,
                            maturity_years,
                            mark_iv,
                            option_type,
                        )
                        distance = abs(math.log(strike / spot))
                        relative_width = 0.012 + 0.020 * distance + 0.002 / math.sqrt(
                            max(maturity_years, 1.0 / 365.0)
                        )
                        half_option_spread = max(0.00005, 0.5 * relative_width * mark_price)
                        bid_price = max(0.0, mark_price - half_option_spread)
                        ask_price = mark_price + half_option_spread
                        bid_iv = (
                            _implied_volatility(
                                bid_price * spot,
                                spot,
                                strike,
                                maturity_years,
                                option_type,
                            )
                            if bid_price > 0
                            else None
                        )
                        ask_iv = _implied_volatility(
                            ask_price * spot,
                            spot,
                            strike,
                            maturity_years,
                            option_type,
                        )
                        open_interest = max(
                            1.0,
                            (2_500.0 if asset == "BTC" else 18_000.0)
                            * math.exp(-3.0 * distance)
                            / math.sqrt(max((expiration - start).days, 1)),
                        )
                        strike_text = str(int(strike)) if strike.is_integer() else str(strike)
                        side = "C" if option_type == "call" else "P"
                        yield QuoteSnapshot(
                            exchange="deribit",
                            symbol=f"{asset}-{_expiry_token(expiration)}-{strike_text}-{side}",
                            asset=asset,
                            instrument_type="option",
                            timestamp=event_time,
                            local_timestamp=grid_time,
                            bid_price=bid_price,
                            ask_price=ask_price,
                            bid_amount=max(0.1, 4.0 * math.exp(-2.0 * distance)),
                            ask_amount=max(0.1, 3.8 * math.exp(-2.0 * distance)),
                            option_type=option_type,
                            strike_price=strike,
                            expiration=expiration,
                            open_interest=open_interest,
                            bid_iv=bid_iv,
                            ask_iv=ask_iv,
                            mark_price=mark_price,
                            mark_iv=mark_iv,
                            underlying_index=f"SYN.{asset}-{_expiry_token(expiration)}",
                            underlying_price=spot,
                            delta=delta,
                            gamma=gamma,
                            vega=vega,
                            theta=theta,
                            rho=rho,
                            snapshot_time=grid_time,
                            source="synthetic_mixture_black",
                        )


def write_synthetic_fixture(
    root: str | Path,
    *,
    file_format: Literal["csv", "parquet"] = "csv",
    overwrite: bool = False,
    **generator_options: object,
) -> tuple[Path, ...]:
    """Generate and partition an offline smoke/demo dataset."""

    snapshots = generate_synthetic_snapshots(**generator_options)  # type: ignore[arg-type]
    return write_partitioned(
        snapshots,
        root,
        file_format=file_format,
        overwrite=overwrite,
    )


__all__ = ["generate_synthetic_snapshots", "write_synthetic_fixture"]
