from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crypto_vol_lab import __version__, cli
from crypto_vol_lab.config import ResearchConfig
from crypto_vol_lab.data import read_partitioned_csv
from crypto_vol_lab.data_models import DataFilterConfig


def _stdout_json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)  # type: ignore[no-any-return]


def _write_quote_file(path: Path, symbol: str) -> None:
    first = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    second = datetime(2024, 1, 1, 0, 15, tzinfo=UTC)
    header = (
        "exchange,symbol,timestamp,local_timestamp,bid_price,bid_amount,"
        "ask_price,ask_amount\n"
    )
    rows = []
    for index, timestamp in enumerate((first, second)):
        micros = int(timestamp.timestamp() * 1_000_000)
        mid = 30_000.0 + 10.0 * index if symbol.startswith("BTC") else 2_000.0 + index
        rows.append(
            f"deribit,{symbol},{micros},{micros},{mid - 1.0},5,{mid + 1.0},5\n"
        )
    path.write_bytes(gzip.compress((header + "".join(rows)).encode("utf-8")))


def test_parser_exposes_all_readme_commands() -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()
    for command in ("config", "demo", "download", "prepare", "backtest"):
        assert command in help_text


def test_config_command_emits_typed_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "research.toml"
    config_path.write_text("[research]\nseed = 11\n", encoding="utf-8")

    assert cli.main(["config", "--path", str(config_path)]) == 0
    payload = _stdout_json(capsys)

    assert payload["command"] == "config"
    assert payload["valid"] is True
    assert payload["config"]["seed"] == 11
    assert payload["config"]["data"]["assets"] == ["BTC", "ETH"]


def test_config_error_is_actionable_and_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text("[data]\ninterval_minutes = 17\n", encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        cli.main(["config", "--path", str(config_path)])

    captured = capsys.readouterr()
    assert raised.value.code == cli.EXIT_USAGE
    assert captured.out == ""
    assert "interval_minutes must be either 15 or 60" in captured.err


def test_download_refuses_network_without_explicit_terms(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    class FailingClient:
        def __init__(self, **_: object) -> None:
            nonlocal called
            called = True

    monkeypatch.setattr(cli, "TardisClient", FailingClient)
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "download",
                "--date",
                "2024-01-01",
                "--output",
                str(tmp_path / "raw"),
            ]
        )

    captured = capsys.readouterr()
    assert raised.value.code == cli.EXIT_TERMS
    assert called is False
    assert captured.out == ""
    assert "--accept-provider-terms" in captured.err
    assert "terms-of-service" in captured.err


def test_download_keeps_randomized_raw_name_and_json_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, *, accept_terms: bool, api_key: str | None) -> None:
            assert accept_terms is True
            assert api_key is None

        @contextmanager
        def download_day(
            self,
            request: object,
            *,
            manifest_dir: Path,
            temporary_dir: Path,
            keep_raw: bool,
            expected_sha256: str | None,
        ) -> Iterator[SimpleNamespace]:
            assert keep_raw is True
            assert expected_sha256 is None
            temporary_dir.mkdir(parents=True, exist_ok=True)
            manifest_dir.mkdir(parents=True, exist_ok=True)
            raw_path = (
                temporary_dir
                / "deribit_options_chain_2024-01-01_OPTIONS_a1b2c3.csv.gz"
            )
            raw_path.write_bytes(gzip.compress(b"a,b\n1,2\n"))
            manifest_path = manifest_dir / "fixture.manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            yield SimpleNamespace(
                raw_path=raw_path,
                manifest_path=manifest_path,
                url="https://example.test/OPTIONS.csv.gz",
                sha256="a" * 64,
                size_bytes=42,
            )

    monkeypatch.delenv("TARDIS_API_KEY", raising=False)
    monkeypatch.setattr(cli, "TardisClient", FakeClient)
    raw_root = tmp_path / "raw"
    assert (
        cli.main(
            [
                "download",
                "--date",
                "2024-01-01",
                "--output",
                str(raw_root),
                "--keep-raw",
                "--accept-provider-terms",
            ]
        )
        == 0
    )
    payload = _stdout_json(capsys)

    expected = raw_root / "deribit_options_chain_2024-01-01_OPTIONS_a1b2c3.csv.gz"
    assert expected.exists()
    assert payload["raw_path"] == str(expected)
    assert payload["raw_retained"] is True
    assert payload["manifest"] == str(tmp_path / "manifests" / "fixture.manifest.json")


def test_prepare_accepts_repeated_inputs_and_writes_csv_partitions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    btc = tmp_path / "btc.csv.gz"
    eth = tmp_path / "eth.csv.gz"
    btc_second = tmp_path / "btc_second.csv.gz"
    _write_quote_file(btc, "BTC-PERPETUAL")
    _write_quote_file(eth, "ETH-PERPETUAL")
    _write_quote_file(btc_second, "BTC-SECOND")
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(
        """
[data]
interval_minutes = 15
max_spread_fraction = 0.12
min_days_to_expiry = 3.0
max_days_to_expiry = 45.0
""".lstrip(),
        encoding="utf-8",
    )
    output = tmp_path / "processed"
    observed: dict[str, object] = {}
    original_resample = cli.resample_snapshots

    def recording_resample(snapshots: Any, **kwargs: Any) -> Any:
        observed["interval_minutes"] = kwargs["interval_minutes"]
        observed["filters"] = kwargs["filters"]
        return original_resample(snapshots, **kwargs)

    monkeypatch.setattr(cli, "resample_snapshots", recording_resample)

    assert (
        cli.main(
            [
                "prepare",
                "--input",
                str(btc),
                "--input",
                str(eth),
                "--input",
                str(btc_second),
                "--output",
                str(output),
                "--config",
                str(config_path),
                "--interval-minutes",
                "15",
                "--format",
                "csv",
            ]
        )
        == 0
    )
    payload = _stdout_json(capsys)
    snapshots = tuple(read_partitioned_csv(output))
    manifest_path = Path(payload["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["inputs"] == [str(btc), str(eth), str(btc_second)]
    assert payload["format"] == "csv"
    assert payload["input_snapshots"] == 6
    assert payload["prepared_snapshots"] == 6
    assert len(payload["partitions"]) == 2
    assert len(snapshots) == 6
    assert {snapshot.asset for snapshot in snapshots} == {"BTC", "ETH"}
    assert observed["interval_minutes"] == 15
    assert isinstance(observed["filters"], DataFilterConfig)
    filters = observed["filters"]
    assert filters.max_relative_spread == 0.12
    assert filters.min_dte_days == 3.0
    assert filters.max_dte_days == 45.0

    assert manifest_path == output / "transformation.manifest.json"
    assert manifest["package_version"] == __version__
    assert manifest["config"]["path"] == str(config_path)
    assert manifest["config"]["sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    assert manifest["config"]["values"]["data"]["interval_minutes"] == 15
    assert manifest["config"]["values"]["surface"]["robust_loss"] == "soft_l1"
    assert manifest["counts"] == {
        "input_snapshots": 6,
        "prepared_snapshots": 6,
    }
    assert manifest["interval_minutes"] == 15
    assert manifest["format"] == "csv"
    assert manifest["filters"]["max_relative_spread"] == 0.12
    assert manifest["filters"]["min_dte_days"] == 3.0
    assert manifest["filters"]["max_dte_days"] == 45.0
    assert [source["path"] for source in manifest["sources"]] == [
        str(btc),
        str(eth),
        str(btc_second),
    ]
    assert [source["snapshot_count"] for source in manifest["sources"]] == [2, 2, 2]
    for source, path in zip(manifest["sources"], (btc, eth, btc_second), strict=True):
        assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert [item["path"] for item in manifest["output_partitions"]] == [
        str(Path(path).relative_to(output)) for path in payload["partitions"]
    ]
    for partition in manifest["output_partitions"]:
        path = output / partition["path"]
        assert partition["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert sorted(
        partition["snapshot_count"] for partition in manifest["output_partitions"]
    ) == [2, 4]
    assert not tuple(output.glob(".transformation.manifest.json.*.tmp"))

    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "prepare",
                "--input",
                str(btc),
                "--output",
                str(output),
                "--config",
                str(config_path),
                "--format",
                "csv",
            ]
        )
    captured = capsys.readouterr()
    assert raised.value.code == cli.EXIT_DATA
    assert "--overwrite" in captured.err


def test_prepare_rejects_explicit_interval_that_conflicts_with_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "quotes.csv.gz"
    _write_quote_file(source, "BTC-PERPETUAL")
    config_path = tmp_path / "hourly.toml"
    config_path.write_text(
        "[data]\ninterval_minutes = 60\n",
        encoding="utf-8",
    )
    derived_output = tmp_path / "derived"
    assert (
        cli.main(
            [
                "prepare",
                "--input",
                str(source),
                "--output",
                str(derived_output),
                "--config",
                str(config_path),
                "--format",
                "csv",
            ]
        )
        == 0
    )
    derived_payload = _stdout_json(capsys)
    derived_manifest = json.loads(
        Path(derived_payload["manifest"]).read_text(encoding="utf-8")
    )
    assert derived_payload["interval_minutes"] == 60
    assert derived_manifest["interval_minutes"] == 60

    output = tmp_path / "mismatch"

    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "prepare",
                "--input",
                str(source),
                "--output",
                str(output),
                "--config",
                str(config_path),
                "--interval-minutes",
                "15",
                "--format",
                "csv",
            ]
        )

    captured = capsys.readouterr()
    assert raised.value.code == cli.EXIT_USAGE
    assert captured.out == ""
    assert "conflicts with data.interval_minutes=60" in captured.err
    assert not output.exists()


def test_demo_and_backtest_delegate_and_emit_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_demo(output: Path, *, seed: int) -> dict[str, Any]:
        calls["demo"] = (output, seed)
        return {"synthetic_demo": True, "seed": seed}

    def fake_backtest(
        input_root: Path, output: Path, config: ResearchConfig
    ) -> dict[str, Any]:
        calls["backtest"] = (input_root, output, config.seed)
        return {"trades": 3, "processed_days": 1}

    monkeypatch.setattr(cli, "run_synthetic_demo", fake_demo)
    monkeypatch.setattr(cli, "run_research_directory", fake_backtest)
    demo_output = tmp_path / "demo"
    assert cli.main(["demo", "--output", str(demo_output), "--seed", "19"]) == 0
    demo_payload = _stdout_json(capsys)
    assert demo_payload["command"] == "demo"
    assert calls["demo"] == (demo_output, 19)

    input_root = tmp_path / "processed"
    input_root.mkdir()
    config_path = tmp_path / "research.toml"
    config_path.write_text("[research]\nseed = 23\n", encoding="utf-8")
    backtest_output = tmp_path / "reports"
    assert (
        cli.main(
            [
                "backtest",
                "--input",
                str(input_root),
                "--config",
                str(config_path),
                "--output",
                str(backtest_output),
            ]
        )
        == 0
    )
    backtest_payload = _stdout_json(capsys)
    assert backtest_payload["command"] == "backtest"
    assert backtest_payload["trades"] == 3
    assert calls["backtest"] == (input_root, backtest_output, 23)
