from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import crypto_vol_lab.experiment as experiment
from crypto_vol_lab.config import (
    BacktestConfig as ResearchBacktestConfig,
)
from crypto_vol_lab.config import DataConfig, ResearchConfig
from crypto_vol_lab.data_models import DataFilterConfig
from crypto_vol_lab.experiment import (
    _audit_protocol_completeness,
    _expected_protocol_dates,
    _protocol_cohort,
    _protocol_exclusion_reason,
    calibrate_ssvi_snapshot,
    engine_config,
    run_research_directory,
    run_synthetic_demo,
)
from crypto_vol_lab.synthetic import generate_synthetic_snapshots


def test_calibration_honors_moneyness_limit_and_robust_loss() -> None:
    quotes = tuple(
        quote
        for quote in generate_synthetic_snapshots(
            periods=1,
            assets=("BTC",),
            moneyness=(0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25),
            seed=3,
        )
        if quote.instrument_type == "option"
    )
    calibration = calibrate_ssvi_snapshot(
        quotes,
        min_quotes_per_expiry=8,
        log_moneyness_limit=0.20,
        robust_loss="huber",
        arbitrage_tolerance=1e-8,
    )

    assert calibration.fit.success
    assert calibration.quotes
    assert max(abs(value) for value in calibration.log_moneyness) <= 0.20
    audit = calibration.arbitrage_audit
    assert audit.passed
    assert audit.butterfly_violations == 0
    assert audit.calendar_violations == 0
    assert audit.price_monotonicity_violations == 0
    assert audit.price_vertical_spread_violations == 0
    assert audit.price_convexity_violations == 0
    assert audit.price_negative_violations == 0
    assert audit.log_moneyness_min < min(calibration.log_moneyness)
    assert audit.log_moneyness_max > max(calibration.log_moneyness)


def test_calibration_rejects_a_failed_numerical_arbitrage_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quotes = tuple(
        quote
        for quote in generate_synthetic_snapshots(
            periods=1,
            assets=("BTC",),
            seed=13,
        )
        if quote.instrument_type == "option"
    )
    valid = calibrate_ssvi_snapshot(quotes)
    failed = replace(
        valid.arbitrage_audit,
        passed=False,
        price_convexity_violations=1,
    )
    monkeypatch.setattr(
        experiment,
        "_numerical_arbitrage_audit",
        lambda *_args, **_kwargs: failed,
    )

    with pytest.raises(ValueError, match="numerical static-arbitrage audit failed"):
        calibrate_ssvi_snapshot(quotes)


def test_invalid_surface_controls_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="robust_loss"):
        calibrate_ssvi_snapshot((), robust_loss="not-a-loss")
    with pytest.raises(ValueError, match="log_moneyness_limit"):
        calibrate_ssvi_snapshot((), log_moneyness_limit=0.0)
    with pytest.raises(ValueError, match="arbitrage_tolerance"):
        calibrate_ssvi_snapshot((), arbitrage_tolerance=-1.0)


def test_engine_config_rejects_ignored_entry_lag_semantics() -> None:
    config = ResearchConfig(
        data=DataConfig(interval_minutes=15),
        backtest=ResearchBacktestConfig(entry_lag_minutes=30),
    )
    with pytest.raises(ValueError, match="entry_lag_minutes"):
        engine_config(config)


def test_engine_config_bounds_perpetual_quote_age_by_resampling_interval() -> None:
    config = ResearchConfig(
        data=DataConfig(interval_minutes=60),
        backtest=ResearchBacktestConfig(entry_lag_minutes=60),
    )

    translated = engine_config(config)

    assert translated.max_perpetual_quote_age.total_seconds() == 60 * 60


@pytest.mark.parametrize(
    ("asset", "timestamp", "expected"),
    [
        ("BTC", datetime(2020, 2, 1, tzinfo=UTC), None),
        ("BTC", datetime(2020, 3, 1, tzinfo=UTC), "btc_development"),
        ("BTC", datetime(2023, 12, 31, 23, tzinfo=UTC), "btc_development"),
        ("BTC", datetime(2024, 1, 1, tzinfo=UTC), "btc_validation"),
        ("BTC", datetime(2025, 1, 1, tzinfo=UTC), "btc_final_test"),
        ("BTC", datetime(2026, 9, 1, tzinfo=UTC), None),
        ("ETH", datetime(2024, 12, 31, tzinfo=UTC), None),
        ("ETH", datetime(2025, 1, 1, tzinfo=UTC), "eth_external_lockbox"),
        ("ETH", datetime(2026, 9, 1, tzinfo=UTC), None),
    ],
)
def test_predeclared_research_cohorts_are_enforced(
    asset: str,
    timestamp: datetime,
    expected: str | None,
) -> None:
    assert _protocol_cohort(asset, timestamp) == expected


def test_pre_lockbox_eth_has_an_explicit_exclusion_reason() -> None:
    timestamp = datetime(2024, 6, 1, tzinfo=UTC)
    assert _protocol_exclusion_reason("ETH", timestamp) == "eth_pre_lockbox"


def _write_partition_stub(
    root: Path,
    observation_date: object,
    asset: str,
    component: str,
) -> None:
    partition = (
        root
        / f"date={observation_date}"
        / f"asset={asset}"
        / f"instrument_type={component}"
    )
    partition.mkdir(parents=True, exist_ok=True)
    (partition / "part-00000.csv").write_text("fixture\n1\n", encoding="utf-8")


def _write_valid_transformation_manifest(root: Path, config: ResearchConfig) -> Path:
    partitions = sorted(root.rglob("part-*.csv"))
    prepared_count = len(partitions)
    payload = {
        "schema_version": 1,
        "artifact_type": "normalized_snapshot_transformation",
        "package_version": "0.1.0",
        "config": {
            "path": "unretained-research.toml",
            "sha256": "1" * 64,
            "values": asdict(config),
        },
        "sources": [
            {
                "path": "unretained-source.csv.gz",
                "sha256": "2" * 64,
                "snapshot_count": prepared_count,
            }
        ],
        "counts": {
            "input_snapshots": prepared_count,
            "prepared_snapshots": prepared_count,
        },
        "filters": asdict(
            DataFilterConfig(
                max_relative_spread=config.data.max_spread_fraction,
                min_dte_days=config.data.min_days_to_expiry,
                max_dte_days=config.data.max_days_to_expiry,
            )
        ),
        "interval_minutes": config.data.interval_minutes,
        "format": "csv",
        "strict": True,
        "output_partitions": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "snapshot_count": 1,
            }
            for path in partitions
        ],
    }
    manifest_path = root / "transformation.manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def test_protocol_completeness_reports_missing_manifest_and_components(
    tmp_path: Path,
) -> None:
    audit = _audit_protocol_completeness(tmp_path, ResearchConfig())

    assert audit["complete"] is False
    assert audit["confirmatory_claim_eligible"] is False
    assert audit["expected"]["BTC"]["months"] == 78
    assert audit["expected"]["ETH"]["months"] == 20
    assert len(audit["missing_components"]["BTC"]["option"]) == 78
    assert len(audit["missing_components"]["ETH"]["perpetual"]) == 20
    assert audit["transformation_manifest"]["status"] == "missing"


def test_protocol_completeness_requires_every_component_and_valid_manifest(
    tmp_path: Path,
) -> None:
    config = ResearchConfig()
    for asset, dates in _expected_protocol_dates().items():
        for observation_date in dates:
            for component in ("option", "perpetual"):
                _write_partition_stub(tmp_path, observation_date, asset, component)
    _write_valid_transformation_manifest(tmp_path, config)

    audit = _audit_protocol_completeness(tmp_path, config)

    assert audit["complete"] is True
    assert audit["confirmatory_claim_eligible"] is True
    assert audit["complete_months"] == {"BTC": 78, "ETH": 20}
    assert audit["missing_components"]["BTC"] == {
        "option": [],
        "perpetual": [],
    }
    assert audit["transformation_manifest"]["status"] == "validated"
    assert audit["transformation_manifest"]["checksums_validated"] is True


def test_protocol_completeness_rejects_tampering_and_unmanifested_partitions(
    tmp_path: Path,
) -> None:
    config = ResearchConfig()
    for asset, dates in _expected_protocol_dates().items():
        for observation_date in dates:
            for component in ("option", "perpetual"):
                _write_partition_stub(tmp_path, observation_date, asset, component)
    _write_valid_transformation_manifest(tmp_path, config)
    first_partition = next(tmp_path.rglob("part-*.csv"))
    first_partition.write_text("fixture\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        _audit_protocol_completeness(tmp_path, config)

    _write_valid_transformation_manifest(tmp_path, config)
    extra = tmp_path / "date=2019-01-01" / "asset=BTC" / "instrument_type=option"
    extra.mkdir(parents=True)
    (extra / "part-00000.csv").write_text("fixture\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        _audit_protocol_completeness(tmp_path, config)


def test_transformation_manifest_interval_mismatch_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "transformation.manifest.json").write_text(
        json.dumps({"interval_minutes": 60}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        _audit_protocol_completeness(tmp_path, ResearchConfig())


def test_research_directory_enforces_config_window_and_threads_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    for day in ("2023-12-01", "2024-01-01"):
        (input_root / f"date={day}" / "asset=BTC").mkdir(parents=True)

    read_paths: list[Path] = []

    def fake_read_partitioned(path: str | Path) -> object:
        read_paths.append(Path(path))
        return iter(())

    bootstrap_seeds: list[int] = []

    def fake_performance_payload(trades: object, *, seed: int) -> dict[str, object]:
        del trades
        bootstrap_seeds.append(seed)
        return {}

    monkeypatch.setattr(experiment, "read_partitioned", fake_read_partitioned)
    monkeypatch.setattr(experiment, "_performance_payload", fake_performance_payload)
    config = ResearchConfig(
        seed=41,
        data=DataConfig(
            assets=("BTC",),
            start="2024-01-01",
            end="2024-12-31",
            interval_minutes=15,
        ),
    )
    summary = run_research_directory(input_root, tmp_path / "output", config)

    assert [path.parent.name for path in read_paths] == ["date=2024-01-01"]
    assert summary["config_data_window"] == {
        "start": "2024-01-01",
        "end": "2024-12-31",
        "skipped_partition_dates": 1,
    }
    assert summary["confirmatory_claim_eligible"] is False
    assert summary["protocol_run_coverage"]["complete"] is False
    assert summary["protocol_run_coverage"]["assets_complete"] is False
    assert summary["protocol_run_coverage"]["window_complete"] is False
    assert summary["data_completeness"]["transformation_manifest"]["status"] == "missing"
    assert set(bootstrap_seeds) == {41}


def test_synthetic_demo_end_to_end_is_nonempty_and_writes_sensible_reports(
    tmp_path: Path,
) -> None:
    summary = run_synthetic_demo(tmp_path, seed=7)

    assert summary["synthetic_demo"] is True
    assert summary["economic_claim"] is False
    assert summary["surface_calibrations"] == 40
    assert summary["surface_failures"] == 0
    assert summary["all_sufficient_no_arbitrage_checks_pass"] is True
    assert summary["all_numerical_static_arbitrage_checks_pass"] is True
    numerical_audit = summary["numerical_static_arbitrage_audit"]
    assert numerical_audit["surfaces_audited"] == 40
    assert numerical_audit["all_passed"] is True
    assert all(count == 0 for count in numerical_audit["violations"].values())
    assert summary["model_price_midpoint_fallbacks"] > 0

    density = summary["risk_neutral_density"]
    assert 0.98 < density["raw_mass"] < 1.02
    assert density["mean"] > 0.0
    assert density["standard_deviation"] > 0.0

    backtest = summary["backtest"]
    # This exact count pins the seeded edge injection and guards against a
    # silently empty demo after interface changes.
    assert backtest["trades"] == 4
    assert backtest["diagnostics"]["completed_trades"] == 4
    assert set(backtest["performance"]) == {"1h", "4h"}
    for horizon in ("1h", "4h"):
        performance = backtest["performance"][horizon]
        assert performance["observations"] == 1
        assert performance["trade_count"] == 2
        assert performance["observation_unit"] == "utc_exit_day"
        assert performance["annualization_periods_per_year"] == 365.0
        assert performance["annualized_sharpe"] is None

    attribution = backtest["pnl_attribution"]
    recomposed = (
        attribution["option"]
        + attribution["perpetual_hedge"]
        + attribution["option_fees"]
        + attribution["perpetual_fees"]
        + attribution["funding"]
    )
    assert attribution["net"] == pytest.approx(recomposed)
    assert all(math.isfinite(value) for value in attribution.values())

    expected_reports = (
        "synthetic_summary.json",
        "synthetic_smile.png",
        "synthetic_risk_neutral_density.png",
        "synthetic_equity.png",
    )
    for filename in expected_reports:
        artifact = tmp_path / filename
        assert artifact.is_file()
        assert artifact.stat().st_size > 500

    persisted = json.loads((tmp_path / "synthetic_summary.json").read_text())
    assert persisted["seed"] == 7
    assert persisted["backtest"]["trades"] == 4
