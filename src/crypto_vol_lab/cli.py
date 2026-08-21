"""Command-line interface for the reproducible volatility research workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from . import __version__
from .config import ResearchConfig, load_config
from .data import (
    TARDIS_TERMS_URL,
    DataPipelineError,
    TardisClient,
    TardisDownloadRequest,
    TermsNotAcceptedError,
    iter_tardis_snapshots,
    merge_snapshot_streams,
    resample_snapshots,
    write_partitioned,
)
from .data_models import DataFilterConfig, QuoteSnapshot
from .experiment import run_research_directory, run_synthetic_demo

EXIT_USAGE = 2
EXIT_TERMS = 3
EXIT_DATA = 4
EXIT_RESEARCH = 5

Payload = dict[str, Any]
CommandHandler = Callable[[argparse.Namespace], Payload]


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid ISO date (expected YYYY-MM-DD)"
        ) from exc


def _existing_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{label} must be a file: {path}")
    return path


def _existing_directory(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} must be a directory: {path}")
    return path


def _validate_config(config: ResearchConfig) -> None:
    """Reject internally inconsistent values that TOML decoding alone cannot catch."""

    if config.data.exchange != "deribit":
        raise ValueError("data.exchange must be 'deribit'")
    if not config.data.assets or set(config.data.assets) - {"BTC", "ETH"}:
        raise ValueError("data.assets must be a non-empty subset of ['BTC', 'ETH']")
    start = _iso_date(config.data.start)
    end = _iso_date(config.data.end)
    if end < start:
        raise ValueError("data.end cannot precede data.start")
    if config.data.interval_minutes not in {15, 60}:
        raise ValueError("data.interval_minutes must be either 15 or 60")
    if config.data.max_spread_fraction < 0.0:
        raise ValueError("data.max_spread_fraction cannot be negative")
    if (
        config.data.min_days_to_expiry < 0.0
        or config.data.max_days_to_expiry < config.data.min_days_to_expiry
    ):
        raise ValueError("data expiry limits must satisfy 0 <= min <= max")
    if config.surface.min_quotes_per_expiry < 5:
        raise ValueError("surface.min_quotes_per_expiry must be at least 5")
    if config.surface.log_moneyness_limit <= 0.0:
        raise ValueError("surface.log_moneyness_limit must be positive")
    if config.surface.robust_loss not in {
        "linear",
        "soft_l1",
        "huber",
        "cauchy",
        "arctan",
    }:
        raise ValueError("surface.robust_loss is not supported")
    if config.surface.arbitrage_tolerance < 0.0:
        raise ValueError("surface.arbitrage_tolerance cannot be negative")
    if config.backtest.entry_lag_minutes <= 0:
        raise ValueError("backtest.entry_lag_minutes must be positive")
    if not config.backtest.holding_periods_minutes or any(
        value <= 0 for value in config.backtest.holding_periods_minutes
    ):
        raise ValueError("backtest.holding_periods_minutes must contain positive values")
    if config.backtest.signal_threshold < 0.0:
        raise ValueError("backtest.signal_threshold cannot be negative")
    fee_values = (
        config.backtest.option_fee_rate,
        config.backtest.perpetual_taker_fee_rate,
    )
    if any(value < 0.0 or value >= 1.0 for value in fee_values):
        raise ValueError("backtest fee rates must lie in [0, 1)")
    if (
        config.backtest.max_gross_vega <= 0.0
        or config.backtest.max_scenario_loss <= 0.0
    ):
        raise ValueError("backtest risk limits must be positive")


def _load_valid_config(path: Path) -> ResearchConfig:
    _existing_file(path, "configuration path")
    config = load_config(path)
    _validate_config(config)
    return config


def _config_command(args: argparse.Namespace) -> Payload:
    path = cast(Path, args.path)
    config = _load_valid_config(path)
    return {
        "command": "config",
        "path": str(path),
        "valid": True,
        "config": asdict(config),
    }


def _demo_command(args: argparse.Namespace) -> Payload:
    output = cast(Path, args.output)
    seed = cast(int, args.seed)
    result = dict(run_synthetic_demo(output, seed=seed))
    result["command"] = "demo"
    result["output"] = str(output)
    return result


def _download_command(args: argparse.Namespace) -> Payload:
    accepted = cast(bool, args.accept_provider_terms)
    if not accepted:
        raise TermsNotAcceptedError(
            "download refused: review the current Tardis terms at "
            f"{TARDIS_TERMS_URL}, then pass --accept-provider-terms explicitly"
        )

    day = cast(date, args.day)
    data_type = cast(Literal["options_chain", "quotes"], args.data_type)
    requested_symbol = cast(str | None, args.symbol)
    symbol = requested_symbol or (
        "OPTIONS" if data_type == "options_chain" else "BTC-PERPETUAL"
    )
    request = TardisDownloadRequest(day=day, data_type=data_type, symbol=symbol)
    output = cast(Path, args.output)
    keep_raw = cast(bool, args.keep_raw)

    output.mkdir(parents=True, exist_ok=True)
    manifest_directory = output.parent / "manifests"
    client = TardisClient(
        accept_terms=True,
        api_key=os.environ.get("TARDIS_API_KEY"),
    )
    with client.download_day(
        request,
        manifest_dir=manifest_directory,
        temporary_dir=output,
        keep_raw=keep_raw,
        expected_sha256=cast(str | None, args.expected_sha256),
    ) as artifact:
        # TardisClient deliberately gives every retained raw payload a unique,
        # randomized pathname so a second download cannot overwrite the first.
        raw_path = artifact.raw_path if keep_raw else None
        payload: Payload = {
            "command": "download",
            "provider": "Tardis.dev",
            "date": request.day.isoformat(),
            "data_type": request.data_type,
            "symbol": request.symbol,
            "url": artifact.url,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
            "manifest": str(artifact.manifest_path),
            "raw_path": str(raw_path) if raw_path is not None else None,
            "raw_retained": raw_path is not None,
        }
    return payload


def _counting_stream(
    stream: Iterator[QuoteSnapshot], counter: list[int]
) -> Iterator[QuoteSnapshot]:
    for snapshot in stream:
        counter[0] += 1
        yield snapshot


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file in bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _partition_snapshot_count(path: Path) -> int:
    """Count persisted snapshots without loading a partition into memory."""

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


def _write_json_atomic(path: Path, payload: Payload) -> Path:
    """Persist JSON by fsyncing a sibling temporary file then replacing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                default=str,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def _prepare_command(args: argparse.Namespace) -> Payload:
    inputs = tuple(cast(list[Path], args.inputs))
    if not inputs:
        raise ValueError("at least one --input is required")
    for path in inputs:
        _existing_file(path, "input")
    if len(set(inputs)) != len(inputs):
        raise ValueError("duplicate --input paths are not allowed")

    config_path = cast(Path, args.config)
    config = _load_valid_config(config_path)
    explicit_interval = cast(int | None, args.interval_minutes)
    configured_interval = config.data.interval_minutes
    if explicit_interval is not None and explicit_interval != configured_interval:
        raise ValueError(
            f"--interval-minutes={explicit_interval} conflicts with "
            f"data.interval_minutes={configured_interval} in {config_path}"
        )
    interval_minutes = cast(Literal[15, 60], configured_interval)

    output = cast(Path, args.output)
    overwrite = cast(bool, args.overwrite)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"output must be a directory: {output}")
    if not overwrite and output.exists() and any(output.rglob("part-*.*")):
        raise FileExistsError(
            f"output already contains partitions: {output}; pass --overwrite to replace them"
        )

    file_format = cast(Literal["csv", "parquet"], args.file_format)
    strict = cast(bool, args.strict)
    source_hashes = tuple(_sha256_file(path) for path in inputs)
    input_counters = tuple([0] for _ in inputs)
    streams = tuple(
        _counting_stream(
            iter_tardis_snapshots(path, strict=strict),
            counter,
        )
        for path, counter in zip(inputs, input_counters, strict=True)
    )
    merged = merge_snapshot_streams(*streams)
    filters = DataFilterConfig(
        max_relative_spread=config.data.max_spread_fraction,
        min_dte_days=config.data.min_days_to_expiry,
        max_dte_days=config.data.max_days_to_expiry,
    )
    sampled = resample_snapshots(
        merged,
        interval_minutes=interval_minutes,
        filters=filters,
    )
    output_counter = [0]
    partitions = write_partitioned(
        _counting_stream(sampled, output_counter),
        output,
        file_format=file_format,
        overwrite=overwrite,
    )
    if not partitions:
        raise DataPipelineError(
            "preparation produced no snapshots; check the input schema, timestamps, and filters"
        )
    source_records = [
        {
            "path": str(path),
            "sha256": digest,
            "snapshot_count": counter[0],
        }
        for path, digest, counter in zip(
            inputs, source_hashes, input_counters, strict=True
        )
    ]
    partition_records = [
        {
            "path": path.relative_to(output).as_posix(),
            "sha256": _sha256_file(path),
            "snapshot_count": _partition_snapshot_count(path),
        }
        for path in partitions
    ]
    total_input_snapshots = sum(counter[0] for counter in input_counters)
    manifest_path = output / "transformation.manifest.json"
    manifest: Payload = {
        "schema_version": 1,
        "artifact_type": "normalized_snapshot_transformation",
        "package_version": __version__,
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "values": asdict(config),
        },
        "sources": source_records,
        "counts": {
            "input_snapshots": total_input_snapshots,
            "prepared_snapshots": output_counter[0],
        },
        "filters": asdict(filters),
        "interval_minutes": interval_minutes,
        "format": file_format,
        "strict": strict,
        "output_partitions": partition_records,
    }
    _write_json_atomic(manifest_path, manifest)
    return {
        "command": "prepare",
        "inputs": [str(path) for path in inputs],
        "config": str(config_path),
        "output": str(output),
        "format": file_format,
        "interval_minutes": interval_minutes,
        "input_snapshots": total_input_snapshots,
        "prepared_snapshots": output_counter[0],
        "partitions": [str(path) for path in partitions],
        "manifest": str(manifest_path),
    }


def _backtest_command(args: argparse.Namespace) -> Payload:
    input_root = _existing_directory(cast(Path, args.input_root), "input")
    config_path = cast(Path, args.config)
    config = _load_valid_config(config_path)
    output = cast(Path, args.output)
    result = dict(run_research_directory(input_root, output, config))
    result["command"] = "backtest"
    result["input"] = str(input_root)
    result["config"] = str(config_path)
    result["output"] = str(output)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser (also useful for documentation/tests)."""

    parser = argparse.ArgumentParser(
        prog="crypto-vol-lab",
        description="Arbitrage-free crypto-volatility research toolkit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser(
        "config", help="validate and print a research TOML configuration"
    )
    config_parser.add_argument("--path", required=True, type=Path)
    config_parser.set_defaults(handler=_config_command)

    demo_parser = subparsers.add_parser(
        "demo", help="run the deterministic synthetic end-to-end demonstration"
    )
    demo_parser.add_argument("--output", required=True, type=Path)
    demo_parser.add_argument("--seed", type=int, default=7)
    demo_parser.set_defaults(handler=_demo_command)

    download_parser = subparsers.add_parser(
        "download", help="stream and verify one Tardis daily dataset"
    )
    download_parser.add_argument("--date", dest="day", required=True, type=_iso_date)
    download_parser.add_argument("--output", required=True, type=Path)
    download_parser.add_argument(
        "--data-type",
        choices=("options_chain", "quotes"),
        default="options_chain",
    )
    download_parser.add_argument(
        "--symbol",
        help="defaults to OPTIONS, or BTC-PERPETUAL for --data-type quotes",
    )
    download_parser.add_argument("--expected-sha256")
    download_parser.add_argument("--keep-raw", action="store_true")
    download_parser.add_argument(
        "--accept-provider-terms",
        action="store_true",
        help=f"confirm that you reviewed {TARDIS_TERMS_URL}",
    )
    download_parser.set_defaults(handler=_download_command)

    prepare_parser = subparsers.add_parser(
        "prepare", help="normalize and causally resample one or more raw files"
    )
    prepare_parser.add_argument(
        "--input",
        dest="inputs",
        action="extend",
        nargs="+",
        required=True,
        type=Path,
        help="raw Tardis .csv.gz path; repeat for options and hedge quotes",
    )
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.add_argument(
        "--config", type=Path, default=Path("configs/research.toml")
    )
    prepare_parser.add_argument(
        "--interval-minutes", choices=(15, 60), type=int
    )
    prepare_parser.add_argument(
        "--format",
        dest="file_format",
        choices=("csv", "parquet"),
        default="parquet",
    )
    prepare_parser.add_argument("--strict", action="store_true")
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(handler=_prepare_command)

    backtest_parser = subparsers.add_parser(
        "backtest", help="run the predeclared research protocol"
    )
    backtest_parser.add_argument("--input", dest="input_root", required=True, type=Path)
    backtest_parser.add_argument("--config", required=True, type=Path)
    backtest_parser.add_argument("--output", required=True, type=Path)
    backtest_parser.set_defaults(handler=_backtest_command)
    return parser


def _exit(parser: argparse.ArgumentParser, status: int, message: str) -> NoReturn:
    parser.exit(status, f"crypto-vol-lab: error: {message}\n")


def _write_payload(payload: Payload) -> None:
    json.dump(
        payload,
        sys.stdout,
        indent=2,
        sort_keys=True,
        default=str,
        allow_nan=False,
    )
    sys.stdout.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, returning zero on success and raising ``SystemExit`` on errors."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(CommandHandler, args.handler)
    try:
        payload = handler(args)
        _write_payload(payload)
    except TermsNotAcceptedError as exc:
        _exit(parser, EXIT_TERMS, str(exc))
    except (
        FileExistsError,
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        PermissionError,
    ) as exc:
        _exit(parser, EXIT_DATA, str(exc))
    except (DataPipelineError, ImportError) as exc:
        _exit(parser, EXIT_DATA, str(exc))
    except (argparse.ArgumentTypeError, TypeError, ValueError) as exc:
        _exit(parser, EXIT_USAGE, str(exc))
    except RuntimeError as exc:
        _exit(parser, EXIT_RESEARCH, str(exc))
    return 0


__all__ = ["build_parser", "main"]
