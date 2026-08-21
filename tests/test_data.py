from __future__ import annotations

import gzip
import hashlib
import json
from datetime import UTC, date, datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from crypto_vol_lab.data import (
    DataPipelineError,
    DownloadError,
    TardisClient,
    TardisDownloadRequest,
    TermsNotAcceptedError,
    iter_tardis_snapshots,
    normalize_tardis_csv,
    passes_filters,
    read_partitioned,
    read_partitioned_csv,
    read_partitioned_parquet,
    resample_snapshots,
    write_partitioned,
)
from crypto_vol_lab.data_models import DataFilterConfig, QuoteSnapshot
from crypto_vol_lab.synthetic import generate_synthetic_snapshots


class _FakeResponse:
    def __init__(
        self, payload: bytes, *, content_length: int | None = None, etag: str = '"abc"'
    ) -> None:
        self._payload = payload
        self._offset = 0
        self.headers = {
            "Content-Length": str(len(payload) if content_length is None else content_length),
            "ETag": etag,
        }

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload) - self._offset
        block = self._payload[self._offset : self._offset + size]
        self._offset += len(block)
        return block


def _perpetual(
    *,
    local_minute: int,
    bid: float,
    symbol: str = "BTC-PERPETUAL",
) -> QuoteSnapshot:
    available = datetime(2023, 6, 1, 0, local_minute, tzinfo=UTC)
    return QuoteSnapshot(
        exchange="deribit",
        symbol=symbol,
        asset=symbol[:3],
        instrument_type="perpetual",
        timestamp=available - timedelta(milliseconds=10),
        local_timestamp=available,
        bid_price=bid,
        ask_price=bid + 1.0,
    )


def _multi_date_snapshots(days: int = 12) -> list[QuoteSnapshot]:
    start = datetime(2023, 6, 1, tzinfo=UTC)
    return [
        snapshot
        for day_offset in range(days)
        for snapshot in generate_synthetic_snapshots(
            start=start + timedelta(days=day_offset),
            periods=1,
            expiries_days=(30,),
            moneyness=(1.0,),
        )
    ]


def _write_gzip(path: Path, text: str) -> None:
    path.write_bytes(gzip.compress(text.encode("utf-8")))


def test_quote_snapshot_metrics_and_utc_normalization() -> None:
    paris = timezone(timedelta(hours=2))
    snapshot = QuoteSnapshot(
        exchange="deribit",
        symbol="BTC-PERPETUAL",
        asset="BTC",
        instrument_type="perpetual",
        timestamp=datetime(2023, 6, 1, 2, 0, tzinfo=paris),
        local_timestamp=datetime(2023, 6, 1, 2, 0, 1, tzinfo=paris),
        bid_price=99.0,
        ask_price=101.0,
    )

    assert snapshot.timestamp == datetime(2023, 6, 1, 0, 0, tzinfo=UTC)
    assert snapshot.mid_price == 100.0
    assert snapshot.spread == 2.0
    assert snapshot.relative_spread == pytest.approx(0.02)
    with pytest.raises(ValueError, match="cannot precede"):
        snapshot.at_grid(snapshot.local_timestamp - timedelta(seconds=1))


def test_tardis_request_urls_and_explicit_terms() -> None:
    request = TardisDownloadRequest(
        day=date(2024, 2, 1), data_type="options_chain", symbol="OPTIONS"
    )
    assert request.url() == (
        "https://datasets.tardis.dev/v1/deribit/options_chain/2024/02/01/OPTIONS.csv.gz"
    )
    with pytest.raises(TermsNotAcceptedError, match="accept_terms=True"):
        TardisClient(accept_terms=False)
    with pytest.raises(ValueError, match="requires one of"):
        TardisDownloadRequest(
            day=date(2024, 2, 1), data_type="quotes", symbol="OPTIONS"
        )


def test_download_stream_manifest_sha_length_and_default_cleanup(tmp_path: Path) -> None:
    payload = gzip.compress(b"exchange,symbol,timestamp,local_timestamp\n")
    calls: list[tuple[str, float]] = []
    downloaded_at = datetime(2024, 2, 2, 3, 4, 5, 678901, tzinfo=UTC)

    def opener(request: Any, *, timeout: float) -> _FakeResponse:
        calls.append((request.full_url, timeout))
        return _FakeResponse(payload)

    client = TardisClient(
        accept_terms=True,
        opener=opener,
        sleeper=lambda _: None,
        chunk_size=7,
        clock=lambda: downloaded_at,
    )
    request = TardisDownloadRequest(
        day=date(2024, 2, 1), data_type="quotes", symbol="BTC-PERPETUAL"
    )
    manifest_dir = tmp_path / "manifests"
    raw_path: Path
    with client.download_day(
        request,
        manifest_dir=manifest_dir,
        temporary_dir=tmp_path / "raw-temp",
    ) as artifact:
        raw_path = artifact.raw_path
        assert raw_path.exists()
        assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
        assert artifact.size_bytes == len(payload)
        manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        assert manifest["terms_accepted"] is True
        assert manifest["raw_redistribution_allowed"] is False
        assert manifest["free_first_of_month_sample"] is True
        assert manifest["content_length"] == len(payload)
        assert manifest["sha256"] == artifact.sha256
        assert manifest["downloaded_at"] == "2024-02-02T03:04:05.678901Z"
        assert artifact.manifest_path.name == (
            f"{request.stem}_20240202T030405.678901Z_"
            f"{artifact.sha256[:16]}.manifest.json"
        )

    assert not raw_path.exists()
    assert len(list((tmp_path / "raw-temp").iterdir())) == 0
    assert len(calls) == 1
    assert calls[0][1] == 60.0


def test_repeated_downloads_create_distinct_immutable_manifests(tmp_path: Path) -> None:
    payload = gzip.compress(b"exchange,symbol,timestamp,local_timestamp\n")
    downloaded_at = datetime(2024, 4, 5, 6, 7, 8, 901234, tzinfo=UTC)

    def opener(_: Any, *, timeout: float) -> _FakeResponse:
        return _FakeResponse(payload)

    client = TardisClient(
        accept_terms=True,
        opener=opener,
        sleeper=lambda _: None,
        clock=lambda: downloaded_at,
    )
    request = TardisDownloadRequest(
        day=date(2024, 4, 1), data_type="quotes", symbol="BTC-PERPETUAL"
    )
    manifest_dir = tmp_path / "manifests"

    with client.download_day(request, manifest_dir=manifest_dir) as first:
        first_path = first.manifest_path
        first_contents = first_path.read_bytes()
    with client.download_day(request, manifest_dir=manifest_dir) as second:
        second_path = second.manifest_path

    assert first_path != second_path
    assert first_path.exists()
    assert second_path.exists()
    assert first_path.read_bytes() == first_contents
    assert json.loads(second_path.read_text(encoding="utf-8"))["terms_accepted"] is True
    assert second_path.name.endswith("-001.manifest.json")
    assert len(tuple(manifest_dir.glob("*.manifest.json"))) == 2


def test_download_retries_same_temp_file_after_length_mismatch(tmp_path: Path) -> None:
    payload = gzip.compress(b"a,b\n1,2\n")
    calls = 0
    sleeps: list[float] = []

    def opener(_: Any, *, timeout: float) -> _FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _FakeResponse(payload, content_length=len(payload) + 3)
        return _FakeResponse(payload)

    client = TardisClient(
        accept_terms=True,
        opener=opener,
        sleeper=sleeps.append,
        max_attempts=2,
        backoff_seconds=0.25,
    )
    request = TardisDownloadRequest(
        day=date(2024, 3, 1), data_type="quotes", symbol="ETH-PERPETUAL"
    )
    with client.download_day(request, manifest_dir=tmp_path / "manifests") as artifact:
        assert artifact.size_bytes == len(payload)
    assert calls == 2
    assert sleeps == [0.25]


def test_free_client_rejects_non_first_day_without_network(tmp_path: Path) -> None:
    called = False

    def opener(_: Any, *, timeout: float) -> _FakeResponse:
        nonlocal called
        called = True
        return _FakeResponse(b"")

    client = TardisClient(accept_terms=True, opener=opener)
    request = TardisDownloadRequest(
        day=date(2024, 3, 2), data_type="quotes", symbol="ETH-PERPETUAL"
    )
    with (
        pytest.raises(DownloadError, match="first UTC day"),
        client.download_day(request, manifest_dir=tmp_path),
    ):
        pass
    assert called is False


def test_parse_options_chain_in_bounded_chunks_and_filter_assets(tmp_path: Path) -> None:
    event = int(datetime(2023, 6, 1, tzinfo=UTC).timestamp() * 1_000_000)
    local = event + 100_000
    expiration = int(datetime(2023, 7, 1, 8, tzinfo=UTC).timestamp() * 1_000_000)
    header = (
        "exchange,symbol,timestamp,local_timestamp,type,strike_price,expiration,"
        "open_interest,last_price,bid_price,bid_amount,bid_iv,ask_price,ask_amount,"
        "ask_iv,mark_price,mark_iv,underlying_index,underlying_price,delta,gamma,"
        "vega,theta,rho\n"
    )
    rows = [
        f"deribit,BTC-30JUN23-30000-C,{event},{local},call,30000,{expiration},10,,"
        "0.09,2,58,0.10,2,61,0.095,60,SYN.BTC,30000,0.52,0.0001,30,-20,0\n",
        f"deribit,ETH-30JUN23-1900-P,{event},{local},put,1900,{expiration},20,,"
        "0.08,3,68,0.09,3,72,0.085,70,SYN.ETH,1900,-0.48,0.001,4,-2,0\n",
        f"deribit,SOL-30JUN23-20-C,{event},{local},call,20,{expiration},30,,"
        "0.1,2,70,0.11,2,72,0.105,71,SYN.SOL,20,0.5,0.01,1,-1,0\n",
        f"deribit,BTC_USDC-30JUN23-30000-C,{event},{local},call,30000,{expiration},"
        "30,,100,2,58,110,2,61,105,60,BTC_USDC,30000,0.5,0.01,1,-1,0\n",
        "malformed,row\n",
    ]
    path = tmp_path / "options.csv.gz"
    _write_gzip(path, header + "".join(rows))

    chunks = list(normalize_tardis_csv(path, chunk_size=1))
    assert [len(chunk) for chunk in chunks] == [1, 1]
    snapshots = [snapshot for chunk in chunks for snapshot in chunk]
    assert [snapshot.asset for snapshot in snapshots] == ["BTC", "ETH"]
    assert snapshots[0].option_type == "call"
    assert snapshots[0].expiration == datetime(2023, 7, 1, 8, tzinfo=UTC)
    assert snapshots[0].bid_iv == pytest.approx(0.58)
    assert snapshots[0].mark_iv == pytest.approx(0.60)
    assert snapshots[0].vega == pytest.approx(3_000.0)
    assert snapshots[1].delta == -0.48


def test_parse_perpetual_quotes_and_strict_malformed_row(tmp_path: Path) -> None:
    event = int(datetime(2023, 6, 1, tzinfo=UTC).timestamp() * 1_000_000)
    header = (
        "exchange,symbol,timestamp,local_timestamp,bid_price,bid_amount,"
        "ask_price,ask_amount\n"
    )
    path = tmp_path / "quotes.csv.gz"
    _write_gzip(
        path,
        header
        + f"deribit,BTC-PERPETUAL,{event},{event + 1000},29999,10,30001,11\n"
        + "deribit,ETH-PERPETUAL,not-a-time,also-bad,1900,1,1901,1\n",
    )

    snapshots = list(iter_tardis_snapshots(path))
    assert len(snapshots) == 1
    assert snapshots[0].instrument_type == "perpetual"
    assert snapshots[0].mid_price == 30_000.0
    with pytest.raises(ValueError, match="row 3"):
        list(iter_tardis_snapshots(path, strict=True))


def test_liquidity_filters_cover_spread_delta_dte_oi_and_feed_lag() -> None:
    snapshots = list(
        generate_synthetic_snapshots(
            periods=1,
            assets=("BTC",),
            expiries_days=(30,),
            moneyness=(1.0,),
        )
    )
    option = next(row for row in snapshots if row.instrument_type == "option")
    permissive = DataFilterConfig(
        max_feed_lag_seconds=1.0,
        max_quote_age_seconds=60.0,
        max_relative_spread=1.0,
        min_abs_delta=0.0,
        max_abs_delta=1.0,
        min_dte_days=1.0,
        max_dte_days=60.0,
        min_open_interest=1.0,
    )
    assert passes_filters(option, permissive)
    assert not passes_filters(
        option,
        DataFilterConfig(max_relative_spread=0.0),
    )
    assert not passes_filters(
        option,
        DataFilterConfig(min_open_interest=(option.open_interest or 0.0) + 1.0),
    )
    assert not passes_filters(option, DataFilterConfig(max_dte_days=2.0))
    assert not passes_filters(
        option, DataFilterConfig(min_abs_delta=0.99, max_abs_delta=1.0)
    )


def test_resampling_is_backward_looking_and_applies_grid_staleness() -> None:
    first = _perpetual(local_minute=10, bid=100.0)
    future = _perpetual(local_minute=16, bid=200.0)
    end = datetime(2023, 6, 1, 0, 30, tzinfo=UTC)

    sampled = list(resample_snapshots([first, future], interval_minutes=15, end=end))
    assert [(row.snapshot_time.minute, row.bid_price) for row in sampled] == [
        (15, 100.0),
        (30, 200.0),
    ]
    assert all(row.local_timestamp <= row.snapshot_time for row in sampled)

    stale_filter = DataFilterConfig(max_quote_age_seconds=10 * 60.0)
    fresh_only = list(
        resample_snapshots(
            [first, future], interval_minutes=15, end=end, filters=stale_filter
        )
    )
    assert [(row.snapshot_time.minute, row.bid_price) for row in fresh_only] == [
        (15, 100.0)
    ]


def test_update_exactly_on_grid_is_available_but_unsorted_input_fails() -> None:
    exact = _perpetual(local_minute=15, bid=123.0)
    end = datetime(2023, 6, 1, 0, 15, tzinfo=UTC)
    sampled = list(resample_snapshots([exact], interval_minutes=15, end=end))
    assert len(sampled) == 1
    assert sampled[0].bid_price == 123.0

    with pytest.raises(ValueError, match="sorted"):
        list(
            resample_snapshots(
                [_perpetual(local_minute=16, bid=2.0), exact],
                interval_minutes=15,
            )
        )


def test_resampling_does_not_bridge_isolated_utc_dates_or_retain_stale_symbols() -> None:
    first_day = [
        _perpetual(local_minute=0, bid=100.0),
        _perpetual(local_minute=15, bid=101.0),
    ]
    second_start = datetime(2023, 7, 1, tzinfo=UTC)
    second_day = [
        QuoteSnapshot(
            exchange="deribit",
            symbol="ETH-PERPETUAL",
            asset="ETH",
            instrument_type="perpetual",
            timestamp=second_start - timedelta(milliseconds=10),
            local_timestamp=second_start,
            bid_price=2_000.0,
            ask_price=2_001.0,
        ),
        QuoteSnapshot(
            exchange="deribit",
            symbol="ETH-PERPETUAL",
            asset="ETH",
            instrument_type="perpetual",
            timestamp=second_start + timedelta(minutes=15, milliseconds=-10),
            local_timestamp=second_start + timedelta(minutes=15),
            bid_price=2_010.0,
            ask_price=2_011.0,
        ),
    ]

    sampled = list(resample_snapshots([*first_day, *second_day], interval_minutes=15))

    assert [snapshot.snapshot_time for snapshot in sampled] == [
        datetime(2023, 6, 1, 0, 0, tzinfo=UTC),
        datetime(2023, 6, 1, 0, 15, tzinfo=UTC),
        datetime(2023, 7, 1, 0, 0, tzinfo=UTC),
        datetime(2023, 7, 1, 0, 15, tzinfo=UTC),
    ]
    assert [snapshot.asset for snapshot in sampled] == ["BTC", "BTC", "ETH", "ETH"]


def test_partitioned_csv_round_trip(tmp_path: Path) -> None:
    snapshots = list(generate_synthetic_snapshots(periods=1))
    root = tmp_path / "normalized"
    paths = write_partitioned(snapshots, root)

    assert len(paths) == 4
    assert all("date=2023-06-01" in str(path) for path in paths)
    assert any("asset=BTC/instrument_type=option" in str(path) for path in paths)
    restored = list(read_partitioned_csv(root))
    auto_restored = list(read_partitioned(root))
    assert len(restored) == len(snapshots)
    assert len(auto_restored) == len(snapshots)
    assert set(restored) == set(snapshots)
    assert set(auto_restored) == set(snapshots)
    with pytest.raises(FileExistsError):
        write_partitioned(snapshots, root)


def test_partitioned_csv_closes_files_when_utc_date_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = _multi_date_snapshots()
    active = 0
    peak = 0
    original_open = Path.open

    class TrackedTextFile:
        def __init__(self, stream: Any) -> None:
            self._stream = stream
            self._closed = False

        def close(self) -> None:
            nonlocal active
            if self._closed:
                return
            try:
                self._stream.close()
            finally:
                self._closed = True
                active -= 1

        def __getattr__(self, name: str) -> Any:
            return getattr(self._stream, name)

    def tracked_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal active, peak
        stream = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if not str(mode).startswith("w"):
            return stream
        active += 1
        peak = max(peak, active)
        return TrackedTextFile(stream)

    monkeypatch.setattr(Path, "open", tracked_open)
    root = tmp_path / "normalized"
    paths = write_partitioned(snapshots, root)

    assert len(paths) == 12 * 4
    assert peak <= 4
    assert active == 0
    assert set(read_partitioned_csv(root)) == set(snapshots)


def test_partitioned_parquet_round_trip_and_auto_detection(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    snapshots = list(generate_synthetic_snapshots(periods=2))
    root = tmp_path / "normalized"

    paths = write_partitioned(snapshots, root, file_format="parquet")
    restored = list(read_partitioned_parquet(root))
    auto_restored = list(read_partitioned(root))

    assert len(paths) == 4
    assert len(restored) == len(snapshots)
    assert len(auto_restored) == len(snapshots)
    assert set(restored) == set(snapshots)
    assert set(auto_restored) == set(snapshots)


def test_partitioned_parquet_closes_writers_when_utc_date_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    snapshots = _multi_date_snapshots()
    active = 0
    peak = 0
    original_writer = pq.ParquetWriter

    class TrackedParquetWriter:
        def __init__(self, writer: Any) -> None:
            self._writer = writer
            self._closed = False

        def close(self) -> None:
            nonlocal active
            if self._closed:
                return
            try:
                self._writer.close()
            finally:
                self._closed = True
                active -= 1

        def __getattr__(self, name: str) -> Any:
            return getattr(self._writer, name)

    def tracked_writer(*args: Any, **kwargs: Any) -> TrackedParquetWriter:
        nonlocal active, peak
        writer = original_writer(*args, **kwargs)
        active += 1
        peak = max(peak, active)
        return TrackedParquetWriter(writer)

    monkeypatch.setattr(pq, "ParquetWriter", tracked_writer)
    root = tmp_path / "normalized"
    paths = write_partitioned(snapshots, root, file_format="parquet")

    assert len(paths) == 12 * 4
    assert peak <= 4
    assert active == 0
    assert set(read_partitioned_parquet(root)) == set(snapshots)


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_partitioned_writer_rejects_backwards_utc_dates(
    tmp_path: Path, file_format: str
) -> None:
    if file_format == "parquet":
        pytest.importorskip("pyarrow")
    earlier = _multi_date_snapshots(days=1)[0]
    later = _multi_date_snapshots(days=2)[-1]

    with pytest.raises(ValueError, match="non-decreasing UTC partition date"):
        write_partitioned(
            [later, earlier],
            tmp_path / file_format,
            file_format=file_format,  # type: ignore[arg-type]
        )


def test_partitioned_auto_reader_rejects_mixed_and_empty_roots(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    snapshots = list(generate_synthetic_snapshots(periods=1, assets=("BTC",)))
    mixed_root = tmp_path / "mixed"
    write_partitioned(snapshots, mixed_root, file_format="csv")
    write_partitioned(snapshots, mixed_root, file_format="parquet")

    with pytest.raises(DataPipelineError, match="mixes CSV and Parquet"):
        list(read_partitioned(mixed_root))
    with pytest.raises(DataPipelineError, match="no CSV or Parquet partitions"):
        list(read_partitioned(tmp_path / "empty"))


def test_synthetic_chain_is_deterministic_and_static_arbitrage_free() -> None:
    options = {
        "periods": 1,
        "assets": ("BTC",),
        "expiries_days": (7, 30, 90),
        "moneyness": (0.8, 0.9, 1.0, 1.1, 1.2),
        "seed": 42,
    }
    first = list(generate_synthetic_snapshots(**options))
    second = list(generate_synthetic_snapshots(**options))
    assert first == second

    chain = [row for row in first if row.instrument_type == "option"]
    expirations = sorted({row.expiration for row in chain})
    for expiration in expirations:
        cross_section = [row for row in chain if row.expiration == expiration]
        calls = sorted(
            (row for row in cross_section if row.option_type == "call"),
            key=lambda row: row.strike_price or 0.0,
        )
        puts = sorted(
            (row for row in cross_section if row.option_type == "put"),
            key=lambda row: row.strike_price or 0.0,
        )
        call_marks = [row.mark_price or 0.0 for row in calls]
        put_marks = [row.mark_price or 0.0 for row in puts]
        assert all(left >= right for left, right in pairwise(call_marks))
        assert all(left <= right for left, right in pairwise(put_marks))

        # Equal strike spacing makes non-negative second differences a direct
        # discrete butterfly-arbitrage check.
        assert all(
            call_marks[index - 1] - 2.0 * call_marks[index] + call_marks[index + 1]
            >= -1e-12
            for index in range(1, len(call_marks) - 1)
        )
        for call, put in zip(calls, puts, strict=True):
            assert call.strike_price == put.strike_price
            spot = call.underlying_price or 0.0
            parity = (spot - (call.strike_price or 0.0)) / spot
            assert (call.mark_price or 0.0) - (put.mark_price or 0.0) == pytest.approx(
                parity, abs=1e-10
            )
