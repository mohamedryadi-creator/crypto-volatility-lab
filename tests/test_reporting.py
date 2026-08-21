from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from crypto_vol_lab.reporting import (
    save_density_plot,
    save_equity_plot,
    save_smile_plot,
    write_json_report,
)


def test_reporting_helpers_create_artifacts(tmp_path: Path) -> None:
    k = np.linspace(-0.4, 0.4, 9)
    smile = 0.6 + 0.1 * k * k
    assert save_smile_plot(k, smile, k, smile, tmp_path / "smile.png", title="Smile").exists()

    strikes = np.linspace(50.0, 150.0, 51)
    density = np.exp(-0.5 * ((strikes - 100.0) / 15.0) ** 2)
    assert save_density_plot(
        strikes,
        density,
        tmp_path / "density.png",
        forward=100.0,
        title="Density",
    ).exists()

    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert save_equity_plot(
        {"1h": [(now, 1.0)]}, tmp_path / "equity.png", title="Equity"
    ).exists()
    report = write_json_report({"finite": 1.0}, tmp_path / "summary.json")
    assert report.read_text(encoding="utf-8").endswith("\n")


def test_reporting_rejects_nonfinite_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        save_smile_plot([0.0], [np.nan], [0.0], [0.2], tmp_path / "x.png", title="x")
