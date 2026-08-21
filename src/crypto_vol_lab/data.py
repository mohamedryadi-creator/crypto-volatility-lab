"""Streaming ingestion, normalization, filtering, and point-in-time storage.

Only the first UTC day of each month is available from Tardis without an API
key.  The client enforces this rather than allowing a research run to fail late
or, worse, silently use a different data source.  Raw licensed files live in a
temporary location and are deleted when the download context exits by default.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO, Literal

from .data_models import DataFilterConfig, QuoteSnapshot

TARDIS_DATASET_BASE_URL = "https://datasets.tardis.dev/v1"
TARDIS_TERMS_URL = "https://docs.tardis.dev/legal/terms-of-service"
SUPPORTED_DATASETS: dict[str, frozenset[str]] = {
    "options_chain": frozenset({"OPTIONS"}),
    "quotes": frozenset({"BTC-PERPETUAL", "ETH-PERPETUAL"}),
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


class DataPipelineError(RuntimeError):
    """Base class for actionable ingestion errors."""


class TermsNotAcceptedError(DataPipelineError):
    """Raised before network I/O unless the caller explicitly accepts terms."""


class DownloadError(DataPipelineError):
    """Raised after all streaming download attempts fail."""


class IntegrityError(DataPipelineError):
    """Raised when length, digest, or gzip integrity validation fails."""


@dataclass(frozen=True, slots=True)
class TardisDownloadRequest:
    """A single daily file in the public Tardis datasets API."""

    day: date
    data_type: Literal["options_chain", "quotes"]
    symbol: str
    exchange: str = "deribit"

    def __post_init__(self) -> None:
        if self.exchange != "deribit":
            raise ValueError("this project intentionally supports Deribit only")
        allowed = SUPPORTED_DATASETS.get(self.data_type)
        if allowed is None:
            raise ValueError(f"unsupported Tardis data type: {self.data_type!r}")
        if self.symbol not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(
                f"{self.data_type} requires one of [{choices}], got {self.symbol!r}"
            )

    @property
    def is_free_sample(self) -> bool:
        return self.day.day == 1

    def url(self, base_url: str = TARDIS_DATASET_BASE_URL) -> str:
        safe_symbol = urllib.parse.quote(self.symbol.replace("/", "-").replace(":", "-"))
        return (
            f"{base_url.rstrip('/')}/{self.exchange}/{self.data_type}/"
            f"{self.day:%Y/%m/%d}/{safe_symbol}.csv.gz"
        )

    @property
    def stem(self) -> str:
        safe_symbol = self.symbol.replace("/", "-").replace(":", "-")
        return f"{self.exchange}_{self.data_type}_{self.day.isoformat()}_{safe_symbol}"


@dataclass(frozen=True, slots=True)
class DownloadArtifact:
    """Verified raw file plus the persistent provenance manifest."""

    request: TardisDownloadRequest
    raw_path: Path
    manifest_path: Path
    sha256: str
    size_bytes: int
    content_length: int | None
    url: str


class TardisClient:
    """Minimal stdlib Tardis client with bounded, retrying stream downloads.

    Parameters
    ----------
    accept_terms:
        Must be explicitly ``True``.  This is never inferred from an environment
        variable because Tardis imposes redistribution restrictions on raw data.
    api_key:
        Optional paid-access key.  Without it, only first-of-month samples are
        permitted by the client and by Tardis.
    opener, sleeper, clock:
        Dependency-injection seams used by offline tests.
    """

    def __init__(
        self,
        *,
        accept_terms: bool,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        chunk_size: int = 1024 * 1024,
        base_url: str = TARDIS_DATASET_BASE_URL,
        opener: Callable[..., Any] = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not accept_terms:
            raise TermsNotAcceptedError(
                "Tardis terms must be accepted explicitly with accept_terms=True; "
                f"review {TARDIS_TERMS_URL}"
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.chunk_size = chunk_size
        self.base_url = base_url
        self._opener = opener
        self._sleeper = sleeper
        self._clock = clock

    @contextmanager
    def download_day(
        self,
        request: TardisDownloadRequest,
        *,
        manifest_dir: str | Path,
        temporary_dir: str | Path | None = None,
        keep_raw: bool = False,
        expected_sha256: str | None = None,
        validate_gzip: bool = True,
    ) -> Iterator[DownloadArtifact]:
        """Download and verify one file, deleting the raw payload on exit.

        One and only one temporary raw pathname is reused across all retries.
        Manifests are retained because they contain provenance but no market
        records.  ``keep_raw=True`` is an explicit expert escape hatch; callers
        remain responsible for keeping licensed raw files outside Git.
        """

        if self.api_key is None and not request.is_free_sample:
            raise DownloadError(
                "Tardis free access is restricted to the first UTC day of each "
                "month; provide an API key or choose YYYY-MM-01"
            )

        manifest_root = Path(manifest_dir)
        manifest_root.mkdir(parents=True, exist_ok=True)
        temp_root = Path(temporary_dir) if temporary_dir is not None else None
        if temp_root is not None:
            temp_root.mkdir(parents=True, exist_ok=True)

        descriptor, raw_name = tempfile.mkstemp(
            prefix=f"{request.stem}_",
            suffix=".csv.gz",
            dir=temp_root,
        )
        os.close(descriptor)
        raw_path = Path(raw_name)
        artifact: DownloadArtifact | None = None
        try:
            artifact = self._download_with_retries(
                request=request,
                raw_path=raw_path,
                manifest_dir=manifest_root,
                expected_sha256=expected_sha256,
                validate_gzip=validate_gzip,
            )
            yield artifact
        finally:
            if not keep_raw or artifact is None:
                raw_path.unlink(missing_ok=True)

    def _download_with_retries(
        self,
        *,
        request: TardisDownloadRequest,
        raw_path: Path,
        manifest_dir: Path,
        expected_sha256: str | None,
        validate_gzip: bool,
    ) -> DownloadArtifact:
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._download_once(
                    request=request,
                    raw_path=raw_path,
                    manifest_dir=manifest_dir,
                    expected_sha256=expected_sha256,
                    validate_gzip=validate_gzip,
                    attempt=attempt,
                )
            except (
                OSError,
                urllib.error.URLError,
                urllib.error.HTTPError,
                IntegrityError,
            ) as exc:
                last_error = exc
                # Truncate the same path; never accumulate failed raw payloads.
                raw_path.write_bytes(b"")
                if attempt < self.max_attempts:
                    self._sleeper(self.backoff_seconds * (2 ** (attempt - 1)))

        raise DownloadError(
            f"failed to download {request.url(self.base_url)} after "
            f"{self.max_attempts} attempt(s): {last_error}"
        ) from last_error

    def _download_once(
        self,
        *,
        request: TardisDownloadRequest,
        raw_path: Path,
        manifest_dir: Path,
        expected_sha256: str | None,
        validate_gzip: bool,
        attempt: int,
    ) -> DownloadArtifact:
        url = request.url(self.base_url)
        headers = {
            "Accept": "application/gzip, application/octet-stream",
            "User-Agent": "crypto-vol-lab/1.0 (+research; stdlib-streaming-client)",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(url, headers=headers, method="GET")

        digest = hashlib.sha256()
        byte_count = 0
        content_length: int | None = None
        response_etag: str | None = None
        with self._opener(http_request, timeout=self.timeout_seconds) as response:
            raw_length = response.headers.get("Content-Length")
            if raw_length not in (None, ""):
                try:
                    content_length = int(raw_length)
                except ValueError as exc:
                    raise IntegrityError(
                        f"invalid Content-Length header: {raw_length!r}"
                    ) from exc
            response_etag = response.headers.get("ETag")
            with raw_path.open("wb") as output:
                while True:
                    block = response.read(self.chunk_size)
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
                    byte_count += len(block)
                output.flush()
                os.fsync(output.fileno())

        if content_length is not None and byte_count != content_length:
            raise IntegrityError(
                f"Content-Length mismatch: expected {content_length}, got {byte_count}"
            )
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256.lower() != expected_sha256.lower():
            raise IntegrityError(
                f"SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        if validate_gzip:
            self._validate_gzip(raw_path)

        downloaded_at = self._clock()
        if downloaded_at.tzinfo is None or downloaded_at.utcoffset() is None:
            raise DataPipelineError("manifest clock must return a timezone-aware datetime")
        downloaded_at = downloaded_at.astimezone(UTC)
        manifest = {
            "schema_version": 1,
            "provider": "Tardis.dev",
            "terms_url": TARDIS_TERMS_URL,
            "terms_accepted": True,
            "raw_redistribution_allowed": False,
            "exchange": request.exchange,
            "data_type": request.data_type,
            "symbol": request.symbol,
            "date": request.day.isoformat(),
            "free_first_of_month_sample": request.is_free_sample and self.api_key is None,
            "url": url,
            "downloaded_at": downloaded_at.isoformat().replace("+00:00", "Z"),
            "sha256": actual_sha256,
            "size_bytes": byte_count,
            "content_length": content_length,
            "etag": response_etag,
            "download_attempt": attempt,
        }
        timestamp_token = downloaded_at.strftime("%Y%m%dT%H%M%S.%fZ")
        manifest_base = f"{request.stem}_{timestamp_token}_{actual_sha256[:16]}"
        manifest_path = _write_json_unique_atomic(
            manifest_dir,
            manifest_base,
            manifest,
        )
        return DownloadArtifact(
            request=request,
            raw_path=raw_path,
            manifest_path=manifest_path,
            sha256=actual_sha256,
            size_bytes=byte_count,
            content_length=content_length,
            url=url,
        )

    def _validate_gzip(self, path: Path) -> None:
        try:
            with gzip.open(path, "rb") as compressed:
                while compressed.read(self.chunk_size):
                    pass
        except (OSError, EOFError) as exc:
            raise IntegrityError(f"invalid or truncated gzip payload: {exc}") from exc


def _write_json_unique_atomic(
    directory: Path,
    base_name: str,
    payload: dict[str, Any],
) -> Path:
    """Publish a complete JSON file once, without replacing prior manifests."""

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{base_name}.", suffix=".tmp", dir=directory
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        sequence = 0
        while True:
            suffix = "" if sequence == 0 else f"-{sequence:03d}"
            destination = directory / f"{base_name}{suffix}.manifest.json"
            try:
                # A hard link atomically exposes the already-fsynced inode and
                # fails instead of replacing an existing immutable manifest.
                os.link(temp_path, destination)
            except FileExistsError:
                sequence += 1
                continue
            return destination
    finally:
        temp_path.unlink(missing_ok=True)


def _micros_to_datetime(value: str) -> datetime:
    return datetime.fromtimestamp(int(value) / 1_000_000, tz=UTC)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_iv(value: Any) -> float | None:
    """Convert Tardis/Deribit percentage-point IVs to decimal volatility."""

    parsed = _optional_float(value)
    return None if parsed is None else parsed / 100.0


def _optional_vega(value: Any) -> float | None:
    """Convert Deribit USD-per-vol-point vega to USD per unit volatility."""

    parsed = _optional_float(value)
    return None if parsed is None else parsed * 100.0


def _asset_from_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    for asset in ("BTC", "ETH"):
        if symbol == f"{asset}-PERPETUAL" or symbol.startswith(f"{asset}-"):
            return asset
    # Newer Deribit USDC instruments can use BTC_USDC / ETH_USDC prefixes.
    for asset in ("BTC", "ETH"):
        if symbol.startswith(f"{asset}_"):
            return asset
    return symbol.split("-", 1)[0].split("_", 1)[0]


def _option_from_row(row: dict[str, str], *, source: str) -> QuoteSnapshot:
    return QuoteSnapshot(
        exchange=row["exchange"].lower(),
        symbol=row["symbol"].upper(),
        asset=_asset_from_symbol(row["symbol"]),
        instrument_type="option",
        timestamp=_micros_to_datetime(row["timestamp"]),
        local_timestamp=_micros_to_datetime(row["local_timestamp"]),
        bid_price=_optional_float(row.get("bid_price")),
        ask_price=_optional_float(row.get("ask_price")),
        bid_amount=_optional_float(row.get("bid_amount")),
        ask_amount=_optional_float(row.get("ask_amount")),
        option_type=row["type"].lower(),  # type: ignore[arg-type]
        strike_price=float(row["strike_price"]),
        expiration=_micros_to_datetime(row["expiration"]),
        open_interest=_optional_float(row.get("open_interest")),
        bid_iv=_optional_iv(row.get("bid_iv")),
        ask_iv=_optional_iv(row.get("ask_iv")),
        mark_price=_optional_float(row.get("mark_price")),
        mark_iv=_optional_iv(row.get("mark_iv")),
        underlying_index=row.get("underlying_index") or None,
        underlying_price=_optional_float(row.get("underlying_price")),
        delta=_optional_float(row.get("delta")),
        gamma=_optional_float(row.get("gamma")),
        vega=_optional_vega(row.get("vega")),
        theta=_optional_float(row.get("theta")),
        rho=_optional_float(row.get("rho")),
        source=source,
    )


def _perpetual_from_row(row: dict[str, str], *, source: str) -> QuoteSnapshot:
    return QuoteSnapshot(
        exchange=row["exchange"].lower(),
        symbol=row["symbol"].upper(),
        asset=_asset_from_symbol(row["symbol"]),
        instrument_type="perpetual",
        timestamp=_micros_to_datetime(row["timestamp"]),
        local_timestamp=_micros_to_datetime(row["local_timestamp"]),
        bid_price=_optional_float(row.get("bid_price")),
        ask_price=_optional_float(row.get("ask_price")),
        bid_amount=_optional_float(row.get("bid_amount")),
        ask_amount=_optional_float(row.get("ask_amount")),
        source=source,
    )


def normalize_tardis_csv(
    path: str | Path,
    *,
    data_type: Literal["options_chain", "quotes"] | None = None,
    chunk_size: int = 50_000,
    assets: Sequence[str] = ("BTC", "ETH"),
    filters: DataFilterConfig | None = None,
    strict: bool = False,
    source: str = "tardis",
    inverse_options_only: bool = True,
) -> Iterator[list[QuoteSnapshot]]:
    """Parse a Tardis ``.csv.gz`` stream into bounded normalized chunks.

    ``data_type`` is inferred from the header when omitted.  Malformed records
    are skipped in research mode and raised immediately with ``strict=True``.
    Filtering is row-local here; quote-age filtering is additionally applied at
    each as-of grid by :func:`resample_snapshots`.  Inverse BTC/ETH options are
    selected by default so that USDC-linear and coin-denominated prices can
    never be mixed in one surface by accident.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    allowed_assets = {asset.upper() for asset in assets}
    batch: list[QuoteSnapshot] = []
    with gzip.open(Path(path), "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Tardis CSV has no header")
        inferred = (
            "options_chain"
            if {"type", "strike_price", "expiration"}.issubset(reader.fieldnames)
            else "quotes"
        )
        selected_type = data_type or inferred
        if selected_type != inferred:
            raise ValueError(
                f"CSV header looks like {inferred}, not requested {selected_type}"
            )
        parser = _option_from_row if selected_type == "options_chain" else _perpetual_from_row

        for line_number, row in enumerate(reader, start=2):
            try:
                snapshot = parser(row, source=source)
                if snapshot.asset not in allowed_assets:
                    continue
                if (
                    selected_type == "options_chain"
                    and inverse_options_only
                    and not snapshot.symbol.startswith(f"{snapshot.asset}-")
                ):
                    # USDC-linear options have a different quote convention and
                    # cannot be mixed with the inverse BTC/ETH contracts.
                    continue
                if filters is not None and not passes_filters(snapshot, filters):
                    continue
            except (KeyError, TypeError, ValueError) as exc:
                if strict:
                    raise ValueError(f"invalid Tardis row {line_number}: {exc}") from exc
                continue
            batch.append(snapshot)
            if len(batch) >= chunk_size:
                yield batch
                batch = []
    if batch:
        yield batch


def iter_tardis_snapshots(
    path: str | Path,
    **kwargs: Any,
) -> Iterator[QuoteSnapshot]:
    """Flatten :func:`normalize_tardis_csv` without materializing a full day."""

    for chunk in normalize_tardis_csv(path, **kwargs):
        yield from chunk


def passes_filters(snapshot: QuoteSnapshot, config: DataFilterConfig) -> bool:
    """Apply point-in-time quality/liquidity filters to one observation."""

    lag = (snapshot.local_timestamp - snapshot.timestamp).total_seconds()
    if lag < 0:
        return False
    if config.max_feed_lag_seconds is not None and lag > config.max_feed_lag_seconds:
        return False

    bid = snapshot.bid_price
    ask = snapshot.ask_price
    if config.require_two_sided and (bid is None or ask is None):
        return False
    if bid is not None and ask is not None:
        if not config.allow_crossed and ask < bid:
            return False
        relative_spread = snapshot.relative_spread
        if config.max_relative_spread is not None and (
            relative_spread is None or relative_spread > config.max_relative_spread
        ):
            return False

    if snapshot.snapshot_time is not None and config.max_quote_age_seconds is not None:
        age = (snapshot.snapshot_time - snapshot.local_timestamp).total_seconds()
        if age < 0 or age > config.max_quote_age_seconds:
            return False

    if snapshot.instrument_type == "option":
        absolute_delta = abs(snapshot.delta) if snapshot.delta is not None else None
        if config.min_abs_delta is not None and (
            absolute_delta is None or absolute_delta < config.min_abs_delta
        ):
            return False
        if config.max_abs_delta is not None and (
            absolute_delta is None or absolute_delta > config.max_abs_delta
        ):
            return False
        dte = snapshot.dte_days
        if config.min_dte_days is not None and (dte is None or dte < config.min_dte_days):
            return False
        if config.max_dte_days is not None and (dte is None or dte > config.max_dte_days):
            return False
        if config.min_open_interest is not None and (
            snapshot.open_interest is None
            or snapshot.open_interest < config.min_open_interest
        ):
            return False

    return True


def _ceil_grid(value: datetime, interval_minutes: int) -> datetime:
    value = value.astimezone(UTC)
    interval_seconds = interval_minutes * 60
    epoch_seconds = int(value.timestamp())
    remainder = epoch_seconds % interval_seconds
    if remainder == 0 and value.microsecond == 0:
        return value
    increment = interval_seconds - remainder
    if value.microsecond:
        increment = interval_seconds - remainder
    return datetime.fromtimestamp(epoch_seconds + increment, tz=UTC).replace(
        microsecond=0
    )


def _floor_grid(value: datetime, interval_minutes: int) -> datetime:
    value = value.astimezone(UTC)
    interval_seconds = interval_minutes * 60
    epoch_seconds = int(value.timestamp())
    return datetime.fromtimestamp(
        epoch_seconds - (epoch_seconds % interval_seconds), tz=UTC
    )


def resample_snapshots(
    snapshots: Iterable[QuoteSnapshot],
    *,
    interval_minutes: Literal[15, 60] = 15,
    filters: DataFilterConfig | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Iterator[QuoteSnapshot]:
    """Carry each symbol backward-looking onto a strict 15m or 60m grid.

    Input must be ordered by ``local_timestamp``.  An update stamped exactly at
    the grid is available and may be selected; a later update never is.  The
    function retains only the latest row per ``(exchange, symbol)`` within one
    UTC date. State is reset when the date advances, so isolated research days
    are never bridged by fabricated snapshots and expired symbols cannot build
    up across the monthly sample.
    """

    if interval_minutes not in (15, 60):
        raise ValueError("interval_minutes must be either 15 or 60")
    if start is not None:
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("start must be timezone-aware")
        start = start.astimezone(UTC)
    if end is not None:
        if end.tzinfo is None or end.utcoffset() is None:
            raise ValueError("end must be timezone-aware")
        end = end.astimezone(UTC)
    if start is not None and end is not None and end < start:
        raise ValueError("end cannot precede start")

    state: dict[tuple[str, str], QuoteSnapshot] = {}
    grid: datetime | None = None
    previous_available: datetime | None = None
    last_available: datetime | None = None
    session_date: date | None = None
    step = timedelta(minutes=interval_minutes)

    def emit(grid_time: datetime) -> Iterator[QuoteSnapshot]:
        for key in sorted(state):
            tagged = state[key].at_grid(grid_time)
            if filters is None or passes_filters(tagged, filters):
                yield tagged

    def drain(terminal: datetime) -> Iterator[QuoteSnapshot]:
        nonlocal grid
        while grid is not None and grid <= terminal:
            yield from emit(grid)
            grid += step

    for snapshot in snapshots:
        available = snapshot.available_at
        if previous_available is not None and available < previous_available:
            raise ValueError(
                "snapshots must be sorted by local_timestamp for strict as-of sampling"
            )
        previous_available = available
        available_date = available.date()

        if session_date is not None and available_date != session_date:
            if grid is not None and last_available is not None:
                terminal = _floor_grid(last_available, interval_minutes)
                if end is not None and end.date() == session_date:
                    terminal = end
                yield from drain(terminal)
            state.clear()
            grid = None
            last_available = None

        session_date = available_date
        if start is not None and available_date < start.date():
            continue
        if end is not None and available_date > end.date():
            break
        if grid is None:
            anchor = available
            if start is not None and start.date() == available_date and start > anchor:
                anchor = start
            grid = _ceil_grid(anchor, interval_minutes)
        last_available = available

        while grid < available and (end is None or grid <= end):
            yield from emit(grid)
            grid += step
        if end is not None and available > end:
            break
        state[snapshot.key] = snapshot

    if grid is None or last_available is None:
        return
    terminal = _floor_grid(last_available, interval_minutes)
    if end is not None and session_date is not None and end.date() == session_date:
        terminal = end
    yield from drain(terminal)


def merge_snapshot_streams(
    *streams: Iterable[QuoteSnapshot],
) -> Iterator[QuoteSnapshot]:
    """Merge already-sorted option/perpetual streams by availability time."""

    yield from heapq.merge(
        *streams,
        key=lambda snapshot: (
            snapshot.local_timestamp,
            snapshot.exchange,
            snapshot.symbol,
        ),
    )


def _partition_date(snapshot: QuoteSnapshot) -> date:
    as_of = snapshot.snapshot_time or snapshot.local_timestamp
    return as_of.astimezone(UTC).date()


def _partition_path(root: Path, snapshot: QuoteSnapshot, suffix: str) -> Path:
    return (
        root
        / f"date={_partition_date(snapshot).isoformat()}"
        / f"asset={snapshot.asset}"
        / f"instrument_type={snapshot.instrument_type}"
        / f"part-00000.{suffix}"
    )


def write_partitioned(
    snapshots: Iterable[QuoteSnapshot],
    root: str | Path,
    *,
    file_format: Literal["csv", "parquet"] = "csv",
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write date/asset/instrument partitions without mandatory dependencies."""

    root_path = Path(root)
    if file_format == "csv":
        return _write_partitioned_csv(snapshots, root_path, overwrite=overwrite)
    if file_format == "parquet":
        return _write_partitioned_parquet(snapshots, root_path, overwrite=overwrite)
    raise ValueError("file_format must be 'csv' or 'parquet'")


def _write_partitioned_csv(
    snapshots: Iterable[QuoteSnapshot],
    root: Path,
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    field_names = [field.name for field in fields(QuoteSnapshot)]
    handles: dict[Path, tuple[Any, csv.DictWriter[Any]]] = {}
    touched: list[Path] = []

    def close_current_date() -> None:
        try:
            for handle, _ in handles.values():
                handle.close()
        finally:
            handles.clear()

    current_date: date | None = None
    try:
        for snapshot in snapshots:
            partition_date = _partition_date(snapshot)
            if current_date is not None and partition_date < current_date:
                raise ValueError(
                    "snapshots must be ordered by non-decreasing UTC partition date"
                )
            if current_date is not None and partition_date > current_date:
                close_current_date()
            current_date = partition_date

            path = _partition_path(root, snapshot, "csv")
            if path not in handles:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and not overwrite:
                    raise FileExistsError(f"partition already exists: {path}")
                handle = path.open("w", encoding="utf-8", newline="")
                writer = csv.DictWriter(handle, fieldnames=field_names)
                writer.writeheader()
                handles[path] = (handle, writer)
                touched.append(path)
            handles[path][1].writerow(snapshot.to_record())
    finally:
        close_current_date()
    return tuple(sorted(touched))


def _write_partitioned_parquet(
    snapshots: Iterable[QuoteSnapshot],
    root: Path,
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Parquet output is optional; install the project's 'parquet' extra "
            "or use file_format='csv'"
        ) from exc

    batch_size = 10_000
    string_fields = {
        "exchange",
        "symbol",
        "asset",
        "instrument_type",
        "option_type",
        "underlying_index",
        "source",
    }
    datetime_fields = set(QuoteSnapshot.DATETIME_FIELDS)
    parquet_schema = pa.schema(
        [
            pa.field(
                field.name,
                (
                    pa.string()
                    if field.name in string_fields
                    else (
                        pa.timestamp("us", tz="UTC")
                        if field.name in datetime_fields
                        else pa.float64()
                    )
                ),
            )
            for field in fields(QuoteSnapshot)
        ]
    )
    buffers: dict[Path, list[dict[str, Any]]] = {}
    writers: dict[Path, Any] = {}
    touched: list[Path] = []

    def flush(path: Path) -> None:
        records = buffers.get(path)
        if not records:
            return
        table = pa.Table.from_pylist(records, schema=parquet_schema)
        writer = writers.get(path)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writers[path] = writer
        writer.write_table(table)
        records.clear()

    def close_current_date() -> None:
        try:
            for path in sorted(buffers, key=str):
                flush(path)
        finally:
            try:
                for writer in writers.values():
                    writer.close()
            finally:
                writers.clear()
                buffers.clear()

    current_date: date | None = None
    try:
        for snapshot in snapshots:
            partition_date = _partition_date(snapshot)
            if current_date is not None and partition_date < current_date:
                raise ValueError(
                    "snapshots must be ordered by non-decreasing UTC partition date"
                )
            if current_date is not None and partition_date > current_date:
                close_current_date()
            current_date = partition_date

            path = _partition_path(root, snapshot, "parquet")
            if path not in buffers:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and not overwrite:
                    raise FileExistsError(f"partition already exists: {path}")
                buffers[path] = []
                touched.append(path)
            buffers[path].append(
                {
                    field.name: getattr(snapshot, field.name)
                    for field in fields(QuoteSnapshot)
                }
            )
            if len(buffers[path]) >= batch_size:
                flush(path)
    finally:
        close_current_date()
    return tuple(sorted(touched))


def read_partitioned_csv(root: str | Path) -> Iterator[QuoteSnapshot]:
    """Read normalized CSV partitions, mainly for smoke tests and demos."""

    for path in sorted(Path(root).rglob("part-*.csv")):
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                yield _snapshot_from_storage_record(row)


def _snapshot_from_storage_record(record: dict[str, Any]) -> QuoteSnapshot:
    """Normalize format-specific nulls before rebuilding the data contract."""

    normalized = dict(record)
    # CSV represents nulls as empty fields, while Parquet returns ``None``.
    # Leaving an empty optional string in place breaks an exact round trip for
    # perpetual rows (``option_type`` and ``underlying_index`` are both null).
    for name in ("option_type", "underlying_index"):
        if normalized.get(name) == "":
            normalized[name] = None
    return QuoteSnapshot.from_record(normalized)


def read_partitioned_parquet(root: str | Path) -> Iterator[QuoteSnapshot]:
    """Read normalized Parquet partitions without loading the full dataset."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Parquet input requires pyarrow; install the project dependencies "
            "or prepare CSV partitions instead"
        ) from exc
    for path in sorted(Path(root).rglob("part-*.parquet")):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=10_000):
            for row in batch.to_pylist():
                yield _snapshot_from_storage_record(row)


def read_partitioned(root: str | Path) -> Iterator[QuoteSnapshot]:
    """Read one partition tree, rejecting ambiguous mixed storage formats."""

    root_path = Path(root)
    csv_files = tuple(root_path.rglob("part-*.csv"))
    parquet_files = tuple(root_path.rglob("part-*.parquet"))
    if csv_files and parquet_files:
        raise DataPipelineError("partition root mixes CSV and Parquet files")
    if parquet_files:
        yield from read_partitioned_parquet(root_path)
    elif csv_files:
        yield from read_partitioned_csv(root_path)
    else:
        raise DataPipelineError(f"no CSV or Parquet partitions found under {root_path}")


def copy_derived_artifact(source: BinaryIO, destination: str | Path) -> Path:
    """Copy a non-raw derived stream atomically (utility for CLI integrations)."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", dir=destination_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination_path
