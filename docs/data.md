# Data contract and reproducibility

## Sources

The default source is the Tardis.dev Deribit dataset:

- `options_chain / OPTIONS` for the active option universe, including executable quotes, IVs, open
  interest, `underlying_index`, `underlying_price` and Greeks; normalization keeps inverse BTC/ETH
  contracts and excludes linear USDC options by default;
- `quotes / BTC-PERPETUAL` and `quotes / ETH-PERPETUAL` for executable hedge prices;
- the first calendar day of every month from March 2020 through August 2026.

These dates are free without an API key according to the
[provider documentation](https://docs.tardis.dev/downloadable-csv-files). The client enforces the
free-date rule and fails explicitly on unavailable files; users must review the linked current
terms because provider availability and policy can change.

## Canonical snapshot

Every normalized option observation has:

| Field | Meaning |
|---|---|
| `timestamp` / `local_timestamp` | UTC exchange-event time / UTC collector-arrival time |
| `snapshot_time` | backward-looking UTC grid point assigned by resampling |
| `symbol` | immutable exchange symbol |
| `asset` | `BTC` or `ETH` |
| `instrument_type` | inverse `option` or USD-quoted, coin-settled inverse `perpetual` hedge |
| `expiration` | UTC contract expiry |
| `strike_price` / `option_type` | contract definition |
| `bid_price`, `ask_price`, `bid_amount`, `ask_amount` | executable top of book in native quote units |
| `mark_price`, `underlying_index`, `underlying_price` | reference values at or before the snapshot |
| `bid_iv`, `ask_iv`, `mark_iv` | provider IVs converted from percent to decimal units |
| `delta`, `gamma`, `vega`, `theta`, `rho` | provider Greeks; vega converted to USD per unit volatility |
| `open_interest` | point-in-time open interest when available |
| `source` | provenance label (`tardis`, synthetic generator or injected demo edge) |

Inverse option premiums remain in BTC or ETH. Perpetual quotes and `underlying_price` are in USD;
Deribit inverse-perpetual order amounts are USD notionals, while their P&L and fees settle in BTC or
ETH. The engine explicitly converts between the option delta in base coin, signed perpetual USD
notional, contract count ($10 per BTC contract, $1 per ETH contract), and settlement-coin P&L.

Model prices, implied volatilities and calibration vegas are recomputed from executable prices.
The execution engine currently uses the point-in-time provider delta and vega fields for position
sizing and the inverse-option hedge transform; the synthetic suite checks their dimensions. A
production study should reconcile those fields against model Greeks and report discrepancies.

## Temporal rules

1. Each raw stream is ordered by `local_timestamp`; multiple streams are merged by this arrival
   time, then exchange and symbol as deterministic tie-breakers.
2. A snapshot contains only the latest observation whose `local_timestamp <= snapshot_time`.
3. Feed lag and quote age are measured against exchange, arrival and grid times before acceptance.
4. Signal time, entry time and exit time are distinct.
5. Missing entry or exit quotes cause a skipped trade, not forward filling.
6. Entry and exit each require a genuine two-sided perpetual quote for the option portfolio's base
   asset. Its maximum age is the configured resampling interval; neither `underlying_price` nor an
   index field is used as an executable-price fallback.

## Storage rules

`download` streams to a randomized temporary `.csv.gz`, computes SHA-256, verifies gzip integrity
and records an immutable manifest with URL, date, dataset type, symbol, byte count, HTTP
length/ETag, attempt count and explicit terms acceptance. The raw payload is removed at context
exit unless `--keep-raw` is set.

`prepare` is deliberately separate. It reads the exact retained paths, applies the filters from
`configs/research.toml`, resamples in bounded streams, writes partitioned Parquet/CSV, and persists
a transformation manifest containing source and partition checksums, counts, filter thresholds,
interval, format and package version. This split prevents download metadata from inventing parsed
row counts or schemas it has not observed. Resampling state is cleared when the UTC date advances,
which prevents monthly sample gaps from being filled and bounds the active symbol universe to one
research day.

Before a confirmatory run, the runner requires the manifest schema, strict-mode flag, complete
configuration and filters to match. It recomputes every partition checksum and row count, requires
all partitions to be non-empty, and rejects both missing and unmanifested partition files.

All real-data locations are ignored by Git. The repository contains deterministic synthetic
fixtures that exercise the same schema and are suitable for unit tests and the public demo.

## Known sampling bias

Only first-of-month dates are free. Intraday observations from one date are not independent.
Inference is clustered by date and all claims are explicitly conditional on this sample. The code
does not infer a full calendar history or interpolate missing days. The empirical runner reports
which expected monthly BTC/ETH option and perpetual partitions are absent and makes confirmatory
cohorts ineligible for claims when the local corpus is incomplete. Provider incident review
remains a human research step.

## Licensing boundary

Provider terms may change. The project does not rely on a 15-minute last-observation snapshot being
sufficiently aggregated or calculated for redistribution. All real raw files, prepared partitions
and generated empirical reports are ignored by Git and remain private by default; only synthetic
fixtures and synthetic demo artifacts are publishable without a separate rights review. This is a
conservative engineering rule, not legal advice.
