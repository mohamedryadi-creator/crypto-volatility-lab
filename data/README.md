# Local data directory

The repository deliberately contains no licensed market data. `crypto-vol-lab download` streams
and verifies one compressed file at a time under `data/raw/` and records an immutable provenance
manifest under `data/manifests/`. It deletes the raw payload at context exit unless `--keep-raw`
is explicitly set. `crypto-vol-lab prepare` is a separate command: it reads retained option and
perpetual files, creates backward-looking snapshots under `data/processed/`, and persists a
transformation manifest with input/output checksums and active filters.

Tardis.dev makes the first day of each month available without an API key. Downloading requires
explicit acceptance of the provider terms through the CLI. Never commit raw, filtered,
point-in-time or otherwise row-level real market data; the 15-minute resampling is not treated as
a legal safe harbor for redistribution.
