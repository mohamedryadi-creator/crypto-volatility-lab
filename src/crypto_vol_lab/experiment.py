"""Point-in-time SSVI calibration and reproducible research orchestration."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .backtest import (
    BacktestConfig as EngineBacktestConfig,
)
from .backtest import (
    BacktestDiagnostics,
    RiskLimits,
    TradeResult,
    run_backtest,
)
from .config import ResearchConfig
from .data import read_partitioned
from .data_models import DataFilterConfig, QuoteSnapshot
from .pricing import black76_greeks, black76_price, implied_volatility
from .reporting import (
    save_density_plot,
    save_equity_plot,
    save_smile_plot,
    write_json_report,
)
from .risk_neutral import density_from_ssvi
from .signals import (
    ModelPriceResolver,
    ResidualSignalConfig,
    SelectionConfig,
)
from .signals import (
    QuoteSnapshot as SignalQuoteSnapshot,
)
from .statistics import (
    DAILY_PERIODS_PER_YEAR,
    TradeLike,
    aggregate_daily_trades,
    day_clustered_bootstrap,
    summarize_trades,
)
from .svi import (
    RobustLoss,
    SSVIFitResult,
    butterfly_diagnostic,
    calendar_arbitrage_diagnostic,
    fit_raw_svi,
    fit_ssvi_surface,
    price_arbitrage_diagnostic,
    raw_svi_total_variance,
    ssvi_butterfly_g,
)
from .synthetic import generate_synthetic_snapshots

FloatArray = NDArray[np.float64]
ModelPriceMap = dict[tuple[datetime, str], float]
_SECONDS_PER_YEAR = 365.0 * 86_400.0
_PROTOCOL_START = date(2020, 3, 1)
_PROTOCOL_END = date(2026, 8, 1)
_ETH_LOCKBOX_START = date(2025, 1, 1)


@dataclass(frozen=True, slots=True)
class _CohortDefinition:
    name: str
    asset: str
    start: date | None
    end: date
    role: str
    claim_eligible: bool


_COHORTS = (
    _CohortDefinition(
        "btc_development",
        "BTC",
        date(2020, 3, 1),
        date(2023, 12, 31),
        "development",
        False,
    ),
    _CohortDefinition(
        "btc_validation",
        "BTC",
        date(2024, 1, 1),
        date(2024, 12, 31),
        "controlled_validation",
        False,
    ),
    _CohortDefinition(
        "btc_final_test",
        "BTC",
        date(2025, 1, 1),
        date(2026, 8, 31),
        "final_out_of_sample",
        True,
    ),
    _CohortDefinition(
        "eth_external_lockbox",
        "ETH",
        date(2025, 1, 1),
        date(2026, 8, 31),
        "external_lockbox",
        True,
    ),
)


def _parse_config_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"data.{field_name} must use ISO date format YYYY-MM-DD") from exc


def _month_starts(first: date, last: date) -> tuple[date, ...]:
    if first > last:
        raise ValueError("first expected month cannot be after last expected month")
    cursor = first.replace(day=1)
    final = last.replace(day=1)
    months: list[date] = []
    while cursor <= final:
        months.append(cursor)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return tuple(months)


def _expected_protocol_dates() -> dict[str, tuple[date, ...]]:
    return {
        "BTC": _month_starts(_PROTOCOL_START, _PROTOCOL_END),
        "ETH": _month_starts(_ETH_LOCKBOX_START, _PROTOCOL_END),
    }


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _partition_snapshot_count(path: Path) -> int:
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return 0
            return sum(1 for _ in reader)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        return int(pq.ParquetFile(path).metadata.num_rows)
    raise ValueError(f"unsupported partition format: {path}")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _manifest_positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"transformation manifest {field_name} must be a positive integer")
    return value


def _transformation_manifest_status(
    root: Path,
    config: ResearchConfig,
) -> dict[str, Any]:
    manifest_path = root / "transformation.manifest.json"
    if not manifest_path.exists():
        return {
            "status": "missing",
            "path": str(manifest_path),
            "expected_interval_minutes": config.data.interval_minutes,
        }
    if not manifest_path.is_file():
        raise ValueError(f"transformation manifest is not a file: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid transformation manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("transformation manifest must contain a JSON object")

    interval: object = payload.get("interval_minutes")
    if isinstance(interval, bool) or not isinstance(interval, int):
        raise ValueError(
            "transformation manifest must contain an integer interval_minutes"
        )
    if interval != config.data.interval_minutes:
        raise ValueError(
            "transformation manifest interval_minutes "
            f"({interval}) does not match config.data.interval_minutes "
            f"({config.data.interval_minutes})"
        )

    if payload.get("schema_version") != 1:
        raise ValueError("transformation manifest schema_version must be 1")
    if payload.get("artifact_type") != "normalized_snapshot_transformation":
        raise ValueError("transformation manifest has an unexpected artifact_type")
    if not isinstance(payload.get("package_version"), str):
        raise ValueError("transformation manifest package_version must be a string")

    file_format = payload.get("format")
    if file_format not in {"csv", "parquet"}:
        raise ValueError("transformation manifest format must be 'csv' or 'parquet'")
    if payload.get("strict") is not True:
        raise ValueError("confirmatory data require strict=true in the transformation manifest")

    config_record = payload.get("config")
    if not isinstance(config_record, dict):
        raise ValueError("transformation manifest config must be an object")
    if not isinstance(config_record.get("path"), str) or not _is_sha256(
        config_record.get("sha256")
    ):
        raise ValueError("transformation manifest config provenance is invalid")
    if _canonical_json(config_record.get("values")) != _canonical_json(asdict(config)):
        raise ValueError("transformation manifest config values do not match the active config")
    recorded_config_path = Path(config_record["path"])
    if recorded_config_path.is_file() and _sha256_file(recorded_config_path) != config_record[
        "sha256"
    ]:
        raise ValueError("transformation manifest config checksum does not match its file")

    expected_filters = asdict(
        DataFilterConfig(
            max_relative_spread=config.data.max_spread_fraction,
            min_dte_days=config.data.min_days_to_expiry,
            max_dte_days=config.data.max_days_to_expiry,
        )
    )
    if _canonical_json(payload.get("filters")) != _canonical_json(expected_filters):
        raise ValueError("transformation manifest filters do not match the active config")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("transformation manifest sources must be a non-empty list")
    source_snapshot_count = 0
    source_paths: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"transformation manifest source {index} must be an object")
        source_path = source.get("path")
        if not isinstance(source_path, str) or not source_path or source_path in source_paths:
            raise ValueError("transformation manifest source paths must be unique strings")
        source_paths.add(source_path)
        if not _is_sha256(source.get("sha256")):
            raise ValueError(f"transformation manifest source {index} has an invalid checksum")
        source_snapshot_count += _manifest_positive_integer(
            source.get("snapshot_count"), f"sources[{index}].snapshot_count"
        )
        retained_source = Path(source_path)
        if retained_source.is_file() and _sha256_file(retained_source) != source["sha256"]:
            raise ValueError(f"transformation manifest source checksum mismatch: {source_path}")

    partition_records = payload.get("output_partitions")
    if not isinstance(partition_records, list) or not partition_records:
        raise ValueError("transformation manifest output_partitions must be non-empty")
    root_resolved = root.resolve()
    recorded_paths: set[Path] = set()
    prepared_snapshot_count = 0
    expected_suffix = f".{file_format}"
    for index, record in enumerate(partition_records):
        if not isinstance(record, dict):
            raise ValueError(f"transformation manifest partition {index} must be an object")
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"transformation manifest partition {index} has no path")
        candidate = Path(raw_path)
        candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError("transformation manifest partition escapes the data root") from exc
        if candidate in recorded_paths:
            raise ValueError("transformation manifest contains a duplicate partition path")
        recorded_paths.add(candidate)
        if candidate.suffix != expected_suffix or not candidate.is_file():
            raise ValueError(
                "transformation manifest partition is missing or wrong format: "
                f"{raw_path}"
            )
        if candidate.stat().st_size <= 0 or not _is_sha256(record.get("sha256")):
            raise ValueError(f"transformation manifest partition is empty or unhashed: {raw_path}")
        if _sha256_file(candidate) != record["sha256"]:
            raise ValueError(f"transformation manifest partition checksum mismatch: {raw_path}")
        recorded_count = _manifest_positive_integer(
            record.get("snapshot_count"), f"output_partitions[{index}].snapshot_count"
        )
        if _partition_snapshot_count(candidate) != recorded_count:
            raise ValueError(f"transformation manifest partition row count mismatch: {raw_path}")
        prepared_snapshot_count += recorded_count

    actual_paths = {
        path.resolve()
        for path in root.rglob("part-*.*")
        if path.is_file() and path.suffix in {".csv", ".parquet"}
    }
    unsupported_paths = [
        path
        for path in root.rglob("part-*.*")
        if path.is_file() and path.suffix not in {".csv", ".parquet"}
    ]
    if unsupported_paths or actual_paths != recorded_paths:
        raise ValueError("partition tree does not exactly match the transformation manifest")

    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("transformation manifest counts must be an object")
    recorded_input_count = _manifest_positive_integer(
        counts.get("input_snapshots"), "counts.input_snapshots"
    )
    if recorded_input_count != source_snapshot_count:
        raise ValueError("transformation manifest input snapshot count is inconsistent")
    recorded_prepared_count = _manifest_positive_integer(
        counts.get("prepared_snapshots"), "counts.prepared_snapshots"
    )
    if recorded_prepared_count != prepared_snapshot_count:
        raise ValueError("transformation manifest prepared snapshot count is inconsistent")

    return {
        "status": "validated",
        "path": str(manifest_path),
        "interval_minutes": interval,
        "expected_interval_minutes": config.data.interval_minutes,
        "format": file_format,
        "strict": True,
        "source_files": len(sources),
        "partition_files": len(recorded_paths),
        "prepared_snapshots": prepared_snapshot_count,
        "checksums_validated": True,
        "config_validated": True,
    }


def _partition_component_present(
    root: Path,
    observation_date: date,
    asset: str,
    instrument_type: str,
) -> bool:
    partition = (
        root
        / f"date={observation_date.isoformat()}"
        / f"asset={asset}"
        / f"instrument_type={instrument_type}"
    )
    if not partition.is_dir():
        return False
    return any(
        path.is_file() and path.suffix in {".csv", ".parquet"}
        for path in partition.glob("part-*.*")
    )


def _audit_protocol_completeness(
    root: Path,
    config: ResearchConfig,
) -> dict[str, Any]:
    expected = _expected_protocol_dates()
    components = ("option", "perpetual")
    missing: dict[str, dict[str, list[str]]] = {}
    complete_months: dict[str, int] = {}
    for asset, dates in expected.items():
        asset_missing: dict[str, list[str]] = {
            component: [] for component in components
        }
        complete_count = 0
        for observation_date in dates:
            present = {
                component: _partition_component_present(
                    root, observation_date, asset, component
                )
                for component in components
            }
            for component, is_present in present.items():
                if not is_present:
                    asset_missing[component].append(observation_date.isoformat())
            complete_count += int(all(present.values()))
        missing[asset] = asset_missing
        complete_months[asset] = complete_count

    manifest = _transformation_manifest_status(root, config)
    partitions_complete = all(
        not dates
        for asset_missing in missing.values()
        for dates in asset_missing.values()
    )
    complete = partitions_complete and manifest["status"] == "validated"
    return {
        "complete": complete,
        "confirmatory_claim_eligible": complete,
        "required_components": list(components),
        "expected": {
            asset: {
                "start": dates[0].isoformat(),
                "end": dates[-1].isoformat(),
                "months": len(dates),
            }
            for asset, dates in expected.items()
        },
        "complete_months": complete_months,
        "missing_components": missing,
        "transformation_manifest": manifest,
    }


@dataclass(frozen=True, slots=True)
class NumericalArbitrageAudit:
    """Aggregated numerical static-arbitrage checks for one fitted surface."""

    passed: bool
    maturity_nodes: int
    log_moneyness_nodes: int
    log_moneyness_min: float
    log_moneyness_max: float
    minimum_butterfly_g: float
    butterfly_violations: int
    minimum_calendar_increment: float
    calendar_violations: int
    price_monotonicity_violations: int
    price_vertical_spread_violations: int
    price_convexity_violations: int
    price_negative_violations: int
    minimum_price_convexity_margin: float


@dataclass(frozen=True, slots=True)
class SurfaceCalibration:
    """One point-in-time, one-asset SSVI calibration and its observations."""

    snapshot_time: datetime
    asset: str
    forward: float
    quotes: tuple[QuoteSnapshot, ...]
    log_moneyness: FloatArray
    maturities: FloatArray
    observed_iv: FloatArray
    model_prices: Mapping[str, float]
    fit: SSVIFitResult
    arbitrage_audit: NumericalArbitrageAudit


@dataclass(frozen=True, slots=True)
class SurfaceBuildResult:
    model_prices: Mapping[tuple[datetime, str], float]
    calibrations: tuple[SurfaceCalibration, ...]
    failures: tuple[tuple[datetime, str, str], ...]


@dataclass(frozen=True, slots=True)
class ResearchRun:
    trades: tuple[TradeResult, ...]
    calibration_count: int
    calibration_failures: int
    processed_days: int


def _snapshot_time(quote: QuoteSnapshot) -> datetime:
    return quote.snapshot_time or quote.local_timestamp


def _maturity(quote: QuoteSnapshot, snapshot_time: datetime) -> float:
    if quote.expiration is None:
        raise ValueError("option quote has no expiration")
    maturity = (quote.expiration - snapshot_time).total_seconds() / _SECONDS_PER_YEAR
    if maturity <= 0.0:
        raise ValueError("option quote is expired")
    return maturity


def _option_type(quote: QuoteSnapshot) -> str:
    if quote.option_type not in {"call", "put"}:
        raise ValueError("option quote has no valid option type")
    return quote.option_type


def _coin_iv(price: float, quote: QuoteSnapshot, maturity: float) -> float:
    if quote.underlying_price is None or quote.strike_price is None:
        raise ValueError("option quote has no underlying or strike")
    return implied_volatility(
        price * quote.underlying_price,
        quote.underlying_price,
        quote.strike_price,
        maturity,
        option_type=_option_type(quote),  # type: ignore[arg-type]
    )


def _valid_option_quotes(quotes: Iterable[QuoteSnapshot]) -> tuple[QuoteSnapshot, ...]:
    accepted: list[QuoteSnapshot] = []
    for quote in quotes:
        if quote.instrument_type != "option":
            continue
        if (
            quote.bid_price is None
            or quote.ask_price is None
            or quote.ask_price <= quote.bid_price
            or quote.bid_price < 0.0
            or quote.mid_price is None
            or quote.mid_price <= 0.0
            or quote.underlying_price is None
            or quote.underlying_price <= 0.0
            or quote.strike_price is None
            or quote.expiration is None
        ):
            continue
        accepted.append(quote)
    return tuple(accepted)


def _numerical_arbitrage_audit(
    fit: SSVIFitResult,
    observed_log_moneyness: Sequence[float],
    *,
    tolerance: float,
) -> NumericalArbitrageAudit:
    """Audit butterfly, calendar and option-price shape on a dense grid."""

    observed = np.asarray(observed_log_moneyness, dtype=np.float64)
    observed_extent = float(np.max(np.abs(observed)))
    audit_extent = max(1.25, observed_extent + 0.25)
    log_moneyness_grid = np.linspace(-audit_extent, audit_extent, 401)
    fitted_maturities = fit.surface.maturities
    maturity_grid = np.linspace(
        float(fitted_maturities[0]),
        float(fitted_maturities[-1]),
        33,
    )

    minimum_butterfly_g = math.inf
    butterfly_violations = 0
    price_monotonicity_violations = 0
    price_vertical_spread_violations = 0
    price_convexity_violations = 0
    price_negative_violations = 0
    minimum_price_convexity_margin = math.inf
    strikes = np.exp(log_moneyness_grid)

    for maturity in maturity_grid:
        theta = float(fit.surface.theta(maturity))
        g_values = np.asarray(
            ssvi_butterfly_g(log_moneyness_grid, theta, fit.params),
            dtype=np.float64,
        )
        butterfly = butterfly_diagnostic(
            log_moneyness_grid,
            g_values,
            tolerance=tolerance,
        )
        minimum_butterfly_g = min(minimum_butterfly_g, butterfly.minimum_g)
        butterfly_violations += butterfly.violation_count

        volatility = np.asarray(
            fit.surface.implied_volatility(log_moneyness_grid, maturity),
            dtype=np.float64,
        )
        call_prices = np.asarray(
            black76_price(
                1.0,
                strikes,
                maturity,
                volatility,
                discount_factor=1.0,
                option_type="call",
            ),
            dtype=np.float64,
        )
        prices = price_arbitrage_diagnostic(
            strikes,
            call_prices,
            option_type="call",
            discount_factor=1.0,
            tolerance=tolerance,
        )
        price_monotonicity_violations += prices.monotonicity_violations
        price_vertical_spread_violations += prices.vertical_spread_violations
        price_convexity_violations += prices.convexity_violations
        price_negative_violations += prices.negative_price_violations
        minimum_price_convexity_margin = min(
            minimum_price_convexity_margin,
            prices.minimum_convexity_margin,
        )

    calendar = calendar_arbitrage_diagnostic(
        fit.surface,
        maturity_grid,
        log_moneyness_grid,
        tolerance=tolerance,
    )
    passed = (
        butterfly_violations == 0
        and calendar.violation_count == 0
        and price_monotonicity_violations == 0
        and price_vertical_spread_violations == 0
        and price_convexity_violations == 0
        and price_negative_violations == 0
    )
    return NumericalArbitrageAudit(
        passed=passed,
        maturity_nodes=int(maturity_grid.size),
        log_moneyness_nodes=int(log_moneyness_grid.size),
        log_moneyness_min=float(log_moneyness_grid[0]),
        log_moneyness_max=float(log_moneyness_grid[-1]),
        minimum_butterfly_g=minimum_butterfly_g,
        butterfly_violations=butterfly_violations,
        minimum_calendar_increment=calendar.minimum_increment,
        calendar_violations=calendar.violation_count,
        price_monotonicity_violations=price_monotonicity_violations,
        price_vertical_spread_violations=price_vertical_spread_violations,
        price_convexity_violations=price_convexity_violations,
        price_negative_violations=price_negative_violations,
        minimum_price_convexity_margin=minimum_price_convexity_margin,
    )


def calibrate_ssvi_snapshot(
    quotes: Iterable[QuoteSnapshot],
    *,
    min_quotes_per_expiry: int = 8,
    log_moneyness_limit: float = math.inf,
    robust_loss: str = "soft_l1",
    arbitrage_tolerance: float = 1e-8,
) -> SurfaceCalibration:
    """Fit a no-arbitrage SSVI surface using only one available snapshot."""

    if min_quotes_per_expiry < 5:
        raise ValueError("min_quotes_per_expiry must be at least five")
    if log_moneyness_limit <= 0.0:
        raise ValueError("log_moneyness_limit must be strictly positive")
    allowed_losses = {"linear", "soft_l1", "huber", "cauchy", "arctan"}
    if robust_loss not in allowed_losses:
        raise ValueError(f"unsupported robust_loss: {robust_loss}")
    if arbitrage_tolerance < 0.0 or not math.isfinite(arbitrage_tolerance):
        raise ValueError("arbitrage_tolerance must be finite and non-negative")
    loss = cast(RobustLoss, robust_loss)

    option_quotes = _valid_option_quotes(quotes)
    if not option_quotes:
        raise ValueError("snapshot contains no valid option quotes")
    snapshot_times = {_snapshot_time(quote) for quote in option_quotes}
    assets = {quote.asset for quote in option_quotes}
    if len(snapshot_times) != 1 or len(assets) != 1:
        raise ValueError("calibration requires exactly one snapshot time and asset")
    snapshot_time = next(iter(snapshot_times))
    asset = next(iter(assets))

    by_expiry: defaultdict[datetime, list[QuoteSnapshot]] = defaultdict(list)
    for quote in option_quotes:
        if quote.expiration is not None:
            by_expiry[quote.expiration].append(quote)

    accepted_quotes: list[QuoteSnapshot] = []
    log_moneyness: list[float] = []
    maturities: list[float] = []
    observed_iv: list[float] = []
    total_variance: list[float] = []
    vegas: list[float] = []
    atm_maturities: list[float] = []
    atm_total_variances: list[float] = []

    for _, expiry_quotes in sorted(by_expiry.items()):
        if len(expiry_quotes) < min_quotes_per_expiry:
            continue
        slice_quotes: list[QuoteSnapshot] = []
        slice_k: list[float] = []
        slice_w: list[float] = []
        slice_bid_w: list[float] = []
        slice_ask_w: list[float] = []
        slice_iv: list[float] = []
        slice_vega: list[float] = []
        slice_maturity = _maturity(expiry_quotes[0], snapshot_time)
        for quote in expiry_quotes:
            if quote.mid_price is None or quote.underlying_price is None:
                continue
            if quote.strike_price is None or quote.bid_price is None or quote.ask_price is None:
                continue
            k = math.log(quote.strike_price / quote.underlying_price)
            if abs(k) > log_moneyness_limit:
                continue
            try:
                mid_iv = _coin_iv(quote.mid_price, quote, slice_maturity)
                bid_iv = _coin_iv(quote.bid_price, quote, slice_maturity)
                ask_iv = _coin_iv(quote.ask_price, quote, slice_maturity)
            except ValueError:
                continue
            mid_w = mid_iv * mid_iv * slice_maturity
            bid_w = min(bid_iv, ask_iv) ** 2 * slice_maturity
            ask_w = max(bid_iv, ask_iv) ** 2 * slice_maturity
            greek = black76_greeks(
                quote.underlying_price,
                quote.strike_price,
                slice_maturity,
                mid_iv,
                option_type=_option_type(quote),  # type: ignore[arg-type]
            )
            slice_quotes.append(quote)
            slice_k.append(k)
            slice_w.append(mid_w)
            slice_bid_w.append(bid_w)
            slice_ask_w.append(ask_w)
            slice_iv.append(mid_iv)
            slice_vega.append(max(greek.vega, 1e-12))
        if len(slice_quotes) < min_quotes_per_expiry or len(set(slice_k)) < 5:
            continue
        raw_fit = fit_raw_svi(
            slice_k,
            slice_w,
            total_variance_bid=slice_bid_w,
            total_variance_ask=slice_ask_w,
            vega=slice_vega,
            loss=loss,
        )
        theta = float(raw_svi_total_variance(0.0, raw_fit.params))
        if not raw_fit.success or theta <= 0.0:
            continue
        atm_maturities.append(slice_maturity)
        atm_total_variances.append(theta)
        accepted_quotes.extend(slice_quotes)
        log_moneyness.extend(slice_k)
        maturities.extend([slice_maturity] * len(slice_quotes))
        observed_iv.extend(slice_iv)
        total_variance.extend(slice_w)
        vegas.extend(slice_vega)

    if len(atm_maturities) < 2 or len(accepted_quotes) < 12:
        raise ValueError("at least two liquid expiry slices are required for SSVI")
    order = np.argsort(np.asarray(atm_maturities))
    atm_t = np.asarray(atm_maturities, dtype=np.float64)[order]
    atm_theta_curve = np.asarray(atm_total_variances, dtype=np.float64)[order]
    atm_theta_curve = np.maximum.accumulate(atm_theta_curve)
    for index in range(1, atm_theta_curve.size):
        atm_theta_curve[index] = max(
            atm_theta_curve[index], atm_theta_curve[index - 1] + 1e-10
        )

    fit = fit_ssvi_surface(
        maturities,
        log_moneyness,
        total_variance,
        atm_t,
        atm_theta_curve,
        weights=vegas,
        loss=loss,
        enforce_sufficient_no_arbitrage=False,
    )
    if not fit.success:
        raise ValueError(f"SSVI optimizer did not converge: {fit.message}")
    report = fit.no_arbitrage_report
    if (
        report.minimum_linear_margin < -arbitrage_tolerance
        or report.minimum_quadratic_margin < -arbitrage_tolerance
    ):
        raise ValueError(
            "SSVI sufficient no-arbitrage conditions exceed configured tolerance"
        )

    numerical_audit = _numerical_arbitrage_audit(
        fit,
        log_moneyness,
        tolerance=arbitrage_tolerance,
    )
    if not numerical_audit.passed:
        raise ValueError(
            "numerical static-arbitrage audit failed: "
            f"butterfly={numerical_audit.butterfly_violations}, "
            f"calendar={numerical_audit.calendar_violations}, "
            f"monotonicity={numerical_audit.price_monotonicity_violations}, "
            f"vertical={numerical_audit.price_vertical_spread_violations}, "
            f"convexity={numerical_audit.price_convexity_violations}, "
            f"negative_prices={numerical_audit.price_negative_violations}"
        )

    prices: dict[str, float] = {}
    for quote, k, maturity in zip(
        accepted_quotes, log_moneyness, maturities, strict=True
    ):
        if quote.underlying_price is None or quote.strike_price is None:
            continue
        fitted_iv = float(fit.surface.implied_volatility(k, maturity))
        usd_price = float(
            black76_price(
                quote.underlying_price,
                quote.strike_price,
                maturity,
                fitted_iv,
                option_type=_option_type(quote),  # type: ignore[arg-type]
            )
        )
        prices[quote.symbol] = usd_price / quote.underlying_price

    forwards = [
        quote.underlying_price
        for quote in accepted_quotes
        if quote.underlying_price is not None
    ]
    return SurfaceCalibration(
        snapshot_time=snapshot_time,
        asset=asset,
        forward=float(np.median(forwards)),
        quotes=tuple(accepted_quotes),
        log_moneyness=np.asarray(log_moneyness, dtype=np.float64),
        maturities=np.asarray(maturities, dtype=np.float64),
        observed_iv=np.asarray(observed_iv, dtype=np.float64),
        model_prices=prices,
        fit=fit,
        arbitrage_audit=numerical_audit,
    )


def build_surface_model_prices(
    quotes: Iterable[QuoteSnapshot],
    *,
    min_quotes_per_expiry: int = 8,
    log_moneyness_limit: float = math.inf,
    robust_loss: str = "soft_l1",
    arbitrage_tolerance: float = 1e-8,
) -> SurfaceBuildResult:
    """Calibrate each asset/snapshot independently and return causal model marks."""

    groups: defaultdict[tuple[datetime, str], list[QuoteSnapshot]] = defaultdict(list)
    for quote in quotes:
        if quote.instrument_type == "option":
            groups[(_snapshot_time(quote), quote.asset)].append(quote)

    prices: ModelPriceMap = {}
    calibrations: list[SurfaceCalibration] = []
    failures: list[tuple[datetime, str, str]] = []
    for (snapshot_time, asset), group in sorted(groups.items()):
        try:
            calibration = calibrate_ssvi_snapshot(
                group,
                min_quotes_per_expiry=min_quotes_per_expiry,
                log_moneyness_limit=log_moneyness_limit,
                robust_loss=robust_loss,
                arbitrage_tolerance=arbitrage_tolerance,
            )
        except ValueError as exc:
            failures.append((snapshot_time, asset, str(exc)))
            continue
        calibrations.append(calibration)
        prices.update(
            {
                (snapshot_time, symbol): price
                for symbol, price in calibration.model_prices.items()
            }
        )
    return SurfaceBuildResult(prices, tuple(calibrations), tuple(failures))


def _complete_model_prices(
    quotes: Iterable[QuoteSnapshot],
    fitted_prices: Mapping[tuple[datetime, str], float],
) -> tuple[ModelPriceMap, int]:
    """Fill uncalibrated contracts at midpoint so they carry zero signal.

    A surface can legitimately exclude deep-ITM quotes whose bid or ask fails
    Black-price bounds.  The execution engine still needs those contracts in
    later snapshots to close a position, but its strict model-price resolver
    must never receive a missing key.  Midpoint fallback is causal and makes an
    excluded quote ineligible for selection rather than fabricating an edge.
    """

    complete = dict(fitted_prices)
    fallback_keys: set[tuple[datetime, str]] = set()
    for quote in quotes:
        if quote.instrument_type != "option" or quote.mid_price is None:
            continue
        key = (_snapshot_time(quote), quote.symbol)
        if key not in complete:
            complete[key] = quote.mid_price
            fallback_keys.add(key)
    return complete, len(fallback_keys)


def _protocol_cohort(asset: str, timestamp: datetime) -> str | None:
    """Return the predeclared cohort, or ``None`` outside the protocol.

    In particular, ETH observations before 2025 are never relabeled as a
    development sample: ETH is reserved exclusively for the external lockbox.
    """

    observation_date = timestamp.date()
    normalized_asset = asset.upper()
    for cohort in _COHORTS:
        if cohort.asset != normalized_asset:
            continue
        if cohort.start is not None and observation_date < cohort.start:
            continue
        if observation_date <= cohort.end:
            return cohort.name
    return None


def _protocol_exclusion_reason(asset: str, timestamp: datetime) -> str:
    normalized_asset = asset.upper()
    observation_date = timestamp.date()
    if normalized_asset == "ETH" and observation_date < date(2025, 1, 1):
        return "eth_pre_lockbox"
    if normalized_asset in {"BTC", "ETH"} and observation_date > date(2026, 8, 31):
        return f"{normalized_asset.lower()}_after_protocol_end"
    return "asset_or_date_outside_protocol"


def _trade_asset(trade: TradeResult) -> str:
    if not trade.legs:
        raise ValueError("cannot assign a cohort to a trade without option legs")
    return trade.legs[0].instrument.split("-", maxsplit=1)[0].upper()


def _equity_curve(
    trades: Iterable[TradeResult], horizon_hours: float
) -> tuple[tuple[datetime, float], ...]:
    cumulative = 0.0
    curve: list[tuple[datetime, float]] = []
    selected = sorted(
        (
            trade
            for trade in trades
            if trade.requested_horizon_hours == horizon_hours
        ),
        key=lambda trade: (trade.exit_time, trade.signal_time),
    )
    for trade in selected:
        cumulative += trade.net_pnl
        curve.append((trade.exit_time, cumulative))
    return tuple(curve)


def _cohort_payloads(
    trades_by_cohort: Mapping[str, Sequence[TradeResult]],
    *,
    bootstrap_seed: int,
    confirmatory_data_complete: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for cohort in _COHORTS:
        trades = tuple(trades_by_cohort.get(cohort.name, ()))
        payload[cohort.name] = {
            "asset": cohort.asset,
            "role": cohort.role,
        "start": cohort.start.isoformat() if cohort.start is not None else None,
            "end": cohort.end.isoformat(),
            "claim_eligible": cohort.claim_eligible and confirmatory_data_complete,
            "trades": len(trades),
            "performance": _performance_payload(trades, seed=bootstrap_seed),
            "pnl_attribution": _pnl_attribution_payload(trades),
        }
    return payload


def engine_config(config: ResearchConfig) -> EngineBacktestConfig:
    """Translate the repository TOML contract to the execution engine config."""

    if config.backtest.entry_lag_minutes != config.data.interval_minutes:
        raise ValueError(
            "entry_lag_minutes must equal data.interval_minutes because execution "
            "occurs on the immediately following snapshot"
        )

    return EngineBacktestConfig(
        horizons_hours=tuple(
            minutes / 60.0 for minutes in config.backtest.holding_periods_minutes
        ),
        signal=ResidualSignalConfig(
            max_relative_spread=config.data.max_spread_fraction
        ),
        selection=SelectionConfig(
            minimum_abs_score=config.backtest.signal_threshold,
            legs_per_side=1,
            target_gross_vega=min(config.backtest.max_gross_vega, 100.0),
        ),
        risk_limits=RiskLimits(
            max_gross_vega=config.backtest.max_gross_vega,
            max_scenario_loss=config.backtest.max_scenario_loss,
        ),
        option_fee_rate=config.backtest.option_fee_rate,
        perpetual_fee_rate=config.backtest.perpetual_taker_fee_rate,
        constant_funding_rate_per_hour=config.backtest.funding_rate_8h / 8.0,
        max_perpetual_quote_age=timedelta(minutes=config.data.interval_minutes),
        inverse_options=True,
    )


def _distort_synthetic_quotes(
    snapshots: Sequence[QuoteSnapshot],
) -> tuple[QuoteSnapshot, ...]:
    times = sorted({_snapshot_time(quote) for quote in snapshots})
    if len(times) < 8:
        return tuple(snapshots)
    base_time = times[0]
    candidates = [
        quote
        for quote in snapshots
        if quote.instrument_type == "option"
        and _snapshot_time(quote) == base_time
        and quote.option_type == "call"
    ]
    expiries = sorted({quote.expiration for quote in candidates if quote.expiration is not None})
    if len(expiries) < 2:
        return tuple(snapshots)
    target_expiry = expiries[1]
    expiry_candidates = sorted(
        (quote for quote in candidates if quote.expiration == target_expiry),
        key=lambda quote: quote.strike_price or 0.0,
    )
    if len(expiry_candidates) < 4:
        return tuple(snapshots)
    cheap_symbol = expiry_candidates[1].symbol
    rich_symbol = expiry_candidates[-2].symbol
    distorted_times = {times[4], times[5]}
    output: list[QuoteSnapshot] = []
    for quote in snapshots:
        if _snapshot_time(quote) not in distorted_times:
            output.append(quote)
            continue
        if quote.symbol == cheap_symbol:
            direction = -1.0
        elif quote.symbol == rich_symbol:
            direction = 1.0
        else:
            direction = 0.0
        if direction == 0.0 or quote.bid_price is None or quote.ask_price is None:
            output.append(quote)
            continue
        factor = 1.0 + direction * 0.12
        output.append(
            replace(
                quote,
                bid_price=max(0.0, quote.bid_price * factor),
                ask_price=quote.ask_price * factor,
                source="synthetic_injected_relative_value",
            )
        )
    return tuple(output)


def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _performance_payload(
    trades: Sequence[TradeResult],
    *,
    seed: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    trade_view = cast(Iterable[TradeLike], trades)
    daily_by_horizon = aggregate_daily_trades(trade_view)
    for horizon, summary in summarize_trades(trade_view).items():
        values = asdict(summary)
        payload[f"{horizon:g}h"] = {
            key: _json_number(value) if isinstance(value, float) else value
            for key, value in values.items()
        }
        daily_cohort = daily_by_horizon[horizon]
        payload[f"{horizon:g}h"].update(
            {
                "trade_count": sum(aggregate.trade_count for aggregate in daily_cohort),
                "observation_unit": "utc_exit_day",
                "annualization_periods_per_year": DAILY_PERIODS_PER_YEAR,
            }
        )
        if daily_cohort:
            bootstrap = day_clustered_bootstrap(
                [aggregate.net_pnl for aggregate in daily_cohort],
                [aggregate.day for aggregate in daily_cohort],
                n_resamples=500,
                seed=seed,
            )
            payload[f"{horizon:g}h"]["day_clustered_mean_ci"] = [
                bootstrap.ci_low,
                bootstrap.ci_high,
            ]
    return payload


def _pnl_attribution_payload(trades: Sequence[TradeResult]) -> dict[str, float]:
    components = (
        "option",
        "perpetual_hedge",
        "option_fees",
        "perpetual_fees",
        "funding",
        "net",
    )
    return {
        component: float(
            sum(trade.pnl_attribution[component] for trade in trades)
        )
        for component in components
    }


def _aggregate_arbitrage_audits(
    items: Sequence[NumericalArbitrageAudit],
) -> dict[str, Any]:
    """Return a JSON-safe aggregate of the audits actually applied."""

    violation_fields = (
        "butterfly_violations",
        "calendar_violations",
        "price_monotonicity_violations",
        "price_vertical_spread_violations",
        "price_convexity_violations",
        "price_negative_violations",
    )
    violations = {
        field: sum(int(getattr(item, field)) for item in items)
        for field in violation_fields
    }
    if not items:
        return {
            "surfaces_audited": 0,
            "all_passed": False,
            "grid": None,
            "minimums": {
                "butterfly_g": None,
                "calendar_total_variance_increment": None,
                "price_convexity_margin": None,
            },
            "violations": violations,
        }

    maturity_node_counts = {item.maturity_nodes for item in items}
    log_moneyness_node_counts = {item.log_moneyness_nodes for item in items}
    return {
        "surfaces_audited": len(items),
        "all_passed": all(item.passed for item in items),
        "grid": {
            "maturity_nodes_per_surface": (
                next(iter(maturity_node_counts))
                if len(maturity_node_counts) == 1
                else sorted(maturity_node_counts)
            ),
            "log_moneyness_nodes_per_maturity": (
                next(iter(log_moneyness_node_counts))
                if len(log_moneyness_node_counts) == 1
                else sorted(log_moneyness_node_counts)
            ),
            "log_moneyness_min": min(item.log_moneyness_min for item in items),
            "log_moneyness_max": max(item.log_moneyness_max for item in items),
        },
        "minimums": {
            "butterfly_g": min(item.minimum_butterfly_g for item in items),
            "calendar_total_variance_increment": min(
                item.minimum_calendar_increment for item in items
            ),
            "price_convexity_margin": min(
                item.minimum_price_convexity_margin for item in items
            ),
        },
        "violations": violations,
    }


def run_synthetic_demo(
    output: str | Path,
    *,
    seed: int = 7,
) -> dict[str, Any]:
    """Run an offline SSVI/backtest demo with explicitly injected synthetic edge."""

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    fair_snapshots = tuple(
        generate_synthetic_snapshots(
            periods=40,
            interval_minutes=15,
            assets=("BTC",),
            seed=seed,
        )
    )
    observed = _distort_synthetic_quotes(fair_snapshots)
    options = tuple(quote for quote in observed if quote.instrument_type == "option")
    perpetuals = tuple(
        quote for quote in observed if quote.instrument_type == "perpetual"
    )
    surfaces = build_surface_model_prices(options)
    if not surfaces.calibrations:
        raise RuntimeError("synthetic SSVI calibration produced no valid surface")
    model_prices, model_price_fallbacks = _complete_model_prices(
        options, surfaces.model_prices
    )
    config = EngineBacktestConfig(
        selection=SelectionConfig(
            minimum_abs_score=0.75,
            legs_per_side=1,
            target_gross_vega=20.0,
            max_abs_quantity_per_leg=5.0,
        ),
        risk_limits=RiskLimits(
            max_gross_vega=100.0,
            max_gross_notional=250_000.0,
            max_abs_position_per_leg=5.0,
            max_scenario_loss=25_000.0,
        ),
        inverse_options=True,
    )
    backtest = run_backtest(
        cast(Iterable[SignalQuoteSnapshot], options),
        cast(ModelPriceResolver, model_prices),
        config,
        perpetual_quotes=cast(Iterable[SignalQuoteSnapshot], perpetuals),
    )

    calibration = surfaces.calibrations[min(4, len(surfaces.calibrations) - 1)]
    unique_maturities = np.unique(calibration.maturities)
    maturity = float(unique_maturities[min(1, unique_maturities.size - 1)])
    maturity_mask = np.isclose(calibration.maturities, maturity)
    observed_k = calibration.log_moneyness[maturity_mask]
    observed_iv = calibration.observed_iv[maturity_mask]
    k_grid = np.linspace(-0.6, 0.6, 241)
    fitted_iv = np.asarray(
        calibration.fit.surface.implied_volatility(k_grid, maturity),
        dtype=np.float64,
    )
    save_smile_plot(
        observed_k,
        observed_iv,
        k_grid,
        fitted_iv,
        output_path / "synthetic_smile.png",
        title="Synthetic BTC smile: observed quotes and arbitrage-free SSVI",
    )

    strike_grid = np.geomspace(0.20 * calibration.forward, 3.0 * calibration.forward, 401)
    distribution = density_from_ssvi(
        calibration.fit.surface,
        maturity,
        calibration.forward,
        strike_grid,
        evaluation_strikes=strike_grid,
    )
    save_density_plot(
        distribution.strikes,
        distribution.density,
        output_path / "synthetic_risk_neutral_density.png",
        forward=calibration.forward,
        title="Synthetic BTC risk-neutral density",
    )
    curves = {
        f"{horizon:g}h": backtest.equity_curve(horizon)
        for horizon in config.horizons_hours
        if backtest.equity_curve(horizon)
    }
    if curves:
        save_equity_plot(
            curves,
            output_path / "synthetic_equity.png",
            title="Synthetic execution test (not empirical performance)",
        )

    numerical_audit = _aggregate_arbitrage_audits(
        [item.arbitrage_audit for item in surfaces.calibrations]
    )

    summary: dict[str, Any] = {
        "synthetic_demo": True,
        "economic_claim": False,
        "seed": seed,
        "quotes": len(options),
        "surface_calibrations": len(surfaces.calibrations),
        "surface_failures": len(surfaces.failures),
        "model_price_midpoint_fallbacks": model_price_fallbacks,
        "mean_ssvi_rmse_total_variance": float(
            np.mean([item.fit.rmse for item in surfaces.calibrations])
        ),
        "all_sufficient_no_arbitrage_checks_pass": all(
            item.fit.no_arbitrage_report.passed for item in surfaces.calibrations
        ),
        "all_numerical_static_arbitrage_checks_pass": numerical_audit[
            "all_passed"
        ],
        "numerical_static_arbitrage_audit": numerical_audit,
        "risk_neutral_density": {
            "raw_mass": distribution.raw_mass,
            "mean": distribution.mean,
            "standard_deviation": distribution.standard_deviation,
            "skewness": distribution.skewness,
        },
        "backtest": {
            "trades": len(backtest.trades),
            "diagnostics": asdict(backtest.diagnostics),
            "performance": _performance_payload(backtest.trades, seed=seed),
            "pnl_attribution": _pnl_attribution_payload(backtest.trades),
        },
    }
    write_json_report(summary, output_path / "synthetic_summary.json")
    return summary


def _aggregate_diagnostics(items: Sequence[BacktestDiagnostics]) -> dict[str, Any]:
    scalar_names = (
        "market_snapshots",
        "signal_snapshots_evaluated",
        "accepted_signals",
        "rejected_signal_quotes",
        "candidate_portfolios",
        "completed_trades",
        "risk_scaled_trades",
    )
    payload: dict[str, Any] = {
        name: sum(int(getattr(item, name)) for item in items) for name in scalar_names
    }
    skips: defaultdict[str, int] = defaultdict(int)
    for item in items:
        for reason, count in item.skip_counts:
            skips[reason] += count
    payload["skip_counts"] = dict(sorted(skips.items()))
    return payload


def run_research_directory(
    input_root: str | Path,
    output: str | Path,
    config: ResearchConfig,
) -> dict[str, Any]:
    """Run the fixed protocol one date/asset partition at a time."""

    root = Path(input_root)
    requested_start = _parse_config_date(config.data.start, "start")
    requested_end = _parse_config_date(config.data.end, "end")
    if requested_start > requested_end:
        raise ValueError("data.start cannot be after data.end")
    completeness = _audit_protocol_completeness(root, config)

    discovered_date_directories = sorted(
        path for path in root.glob("date=*") if path.is_dir()
    )
    if not discovered_date_directories:
        raise FileNotFoundError(f"no date partitions found under {root}")
    date_directories: list[Path] = []
    skipped_partition_dates_outside_config = 0
    for path in discovered_date_directories:
        raw_date = path.name.removeprefix("date=")
        try:
            partition_date = date.fromisoformat(raw_date)
        except ValueError as exc:
            raise ValueError(f"invalid date partition directory: {path}") from exc
        if requested_start <= partition_date <= requested_end:
            date_directories.append(path)
        else:
            skipped_partition_dates_outside_config += 1
    all_trades: list[TradeResult] = []
    trades_by_cohort: dict[str, list[TradeResult]] = {
        cohort.name: [] for cohort in _COHORTS
    }
    diagnostics: list[BacktestDiagnostics] = []
    excluded_snapshots: Counter[str] = Counter()
    calibration_count = 0
    calibration_failures = 0
    numerical_arbitrage_rejections = 0
    numerical_arbitrage_audits: list[NumericalArbitrageAudit] = []
    model_price_fallbacks = 0
    processed_days = 0
    engine = engine_config(config)
    for date_directory in date_directories:
        day_used = False
        for asset in config.data.assets:
            asset_directory = date_directory / f"asset={asset}"
            if not asset_directory.exists():
                continue
            raw_snapshots = tuple(read_partitioned(asset_directory))
            if not raw_snapshots:
                continue
            eligible_snapshots: list[QuoteSnapshot] = []
            for quote in raw_snapshots:
                observation_date = _snapshot_time(quote).date()
                if not requested_start <= observation_date <= requested_end:
                    excluded_snapshots["outside_config_window"] += 1
                    continue
                if _protocol_cohort(quote.asset, _snapshot_time(quote)) is None:
                    reason = _protocol_exclusion_reason(
                        quote.asset, _snapshot_time(quote)
                    )
                    excluded_snapshots[reason] += 1
                    continue
                eligible_snapshots.append(quote)
            snapshots = tuple(eligible_snapshots)
            if not snapshots:
                continue
            options = tuple(
                quote for quote in snapshots if quote.instrument_type == "option"
            )
            perpetuals = tuple(
                quote for quote in snapshots if quote.instrument_type == "perpetual"
            )
            surfaces = build_surface_model_prices(
                options,
                min_quotes_per_expiry=config.surface.min_quotes_per_expiry,
                log_moneyness_limit=config.surface.log_moneyness_limit,
                robust_loss=config.surface.robust_loss,
                arbitrage_tolerance=config.surface.arbitrage_tolerance,
            )
            calibration_count += len(surfaces.calibrations)
            calibration_failures += len(surfaces.failures)
            numerical_arbitrage_audits.extend(
                item.arbitrage_audit for item in surfaces.calibrations
            )
            numerical_arbitrage_rejections += sum(
                reason.startswith("numerical static-arbitrage audit failed:")
                for _, _, reason in surfaces.failures
            )
            if not surfaces.model_prices:
                continue
            model_prices, fallback_count = _complete_model_prices(
                options, surfaces.model_prices
            )
            model_price_fallbacks += fallback_count
            result = run_backtest(
                cast(Iterable[SignalQuoteSnapshot], options),
                cast(ModelPriceResolver, model_prices),
                engine,
                perpetual_quotes=cast(Iterable[SignalQuoteSnapshot], perpetuals),
            )
            for trade in result.trades:
                cohort_name = _protocol_cohort(
                    _trade_asset(trade), trade.signal_time
                )
                if cohort_name is None:
                    excluded_snapshots["trade_outside_protocol"] += 1
                    continue
                all_trades.append(trade)
                trades_by_cohort[cohort_name].append(trade)
            diagnostics.append(result.diagnostics)
            day_used = True
        processed_days += int(day_used)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    trade_path = output_path / "trades.csv"
    if all_trades:
        with trade_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "signal_time",
                "exit_time",
                "horizon_hours",
                "asset",
                "cohort",
                "net_pnl",
                "option_pnl",
                "hedge_pnl",
                "fees",
                "funding",
                "turnover",
                "signal_strength",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trade in sorted(all_trades, key=lambda item: item.exit_time):
                instrument = trade.legs[0].instrument if trade.legs else ""
                asset = instrument.split("-", maxsplit=1)[0]
                writer.writerow(
                    {
                        "signal_time": trade.signal_time.isoformat(),
                        "exit_time": trade.exit_time.isoformat(),
                        "horizon_hours": trade.requested_horizon_hours,
                        "asset": asset,
                        "cohort": _protocol_cohort(asset, trade.signal_time),
                        "net_pnl": trade.net_pnl,
                        "option_pnl": trade.option_gross_pnl,
                        "hedge_pnl": trade.hedge_gross_pnl,
                        "fees": trade.total_fees,
                        "funding": trade.funding_pnl,
                        "turnover": trade.turnover,
                        "signal_strength": trade.signal_strength,
                    }
                )

    configured_assets = {asset.upper() for asset in config.data.assets}
    required_assets = {"BTC", "ETH"}
    assets_cover_protocol = required_assets.issubset(configured_assets)
    window_covers_protocol = (
        requested_start <= _PROTOCOL_START and requested_end >= _PROTOCOL_END
    )
    run_covers_protocol = assets_cover_protocol and window_covers_protocol
    confirmatory_data_complete = bool(completeness["complete"]) and run_covers_protocol
    cohort_payloads = _cohort_payloads(
        trades_by_cohort,
        bootstrap_seed=config.seed,
        confirmatory_data_complete=confirmatory_data_complete,
    )
    confirmatory_performance = {
        cohort.name: {
            "claim_eligible": cohort_payloads[cohort.name]["claim_eligible"],
            "performance": cohort_payloads[cohort.name]["performance"],
        }
        for cohort in _COHORTS
        if cohort.claim_eligible
    }
    numerical_arbitrage_audit = _aggregate_arbitrage_audits(
        numerical_arbitrage_audits
    )
    summary = {
        "synthetic_demo": False,
        "config_data_window": {
            "start": requested_start.isoformat(),
            "end": requested_end.isoformat(),
            "skipped_partition_dates": skipped_partition_dates_outside_config,
        },
        "data_completeness": completeness,
        "confirmatory_claim_eligible": confirmatory_data_complete,
        "protocol_run_coverage": {
            "complete": run_covers_protocol,
            "required_assets": sorted(required_assets),
            "configured_assets": sorted(configured_assets),
            "assets_complete": assets_cover_protocol,
            "required_start": _PROTOCOL_START.isoformat(),
            "required_end": _PROTOCOL_END.isoformat(),
            "window_complete": window_covers_protocol,
        },
        "processed_days": processed_days,
        "surface_calibrations": calibration_count,
        "surface_failures": calibration_failures,
        "all_numerical_static_arbitrage_checks_pass": (
            numerical_arbitrage_audit["all_passed"]
        ),
        "numerical_static_arbitrage_audit": numerical_arbitrage_audit,
        "numerical_static_arbitrage_rejections": numerical_arbitrage_rejections,
        "model_price_midpoint_fallbacks": model_price_fallbacks,
        "trades": len(all_trades),
        "performance": _performance_payload(all_trades, seed=config.seed),
        "pnl_attribution": _pnl_attribution_payload(all_trades),
        "performance_scope": (
            "descriptive aggregate across protocol cohorts; confirmatory cohorts "
            "are reported separately"
        ),
        "cohorts": cohort_payloads,
        "confirmatory_performance": confirmatory_performance,
        "protocol_enforcement": {
            "eth_pre_lockbox_excluded": True,
            "excluded_snapshot_counts": dict(sorted(excluded_snapshots.items())),
        },
        "surface_configuration": {
            "min_quotes_per_expiry": config.surface.min_quotes_per_expiry,
            "log_moneyness_limit": config.surface.log_moneyness_limit,
            "robust_loss": config.surface.robust_loss,
            "arbitrage_tolerance": config.surface.arbitrage_tolerance,
        },
        "diagnostics": _aggregate_diagnostics(diagnostics),
        "sampling_warning": "Inference is conditional on free first-of-month samples.",
    }
    write_json_report(summary, output_path / "research_summary.json")
    curves = {
        f"{cohort.name} {horizon:g}h": _equity_curve(
            trades_by_cohort[cohort.name], horizon
        )
        for cohort in _COHORTS
        for horizon in engine.horizons_hours
    }
    curves = {label: curve for label, curve in curves.items() if curve}
    if curves:
        save_equity_plot(
            curves,
            output_path / "research_equity.png",
            title="Out-of-sample relative-value P&L",
        )
    return summary


__all__ = [
    "NumericalArbitrageAudit",
    "ResearchRun",
    "SurfaceBuildResult",
    "SurfaceCalibration",
    "build_surface_model_prices",
    "calibrate_ssvi_snapshot",
    "engine_config",
    "run_research_directory",
    "run_synthetic_demo",
]
