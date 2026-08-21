from pathlib import Path

import pytest

from crypto_vol_lab.config import DataConfig, SurfaceConfig, load_config


def test_default_research_config_loads() -> None:
    config = load_config(Path("configs/research.toml"))
    assert config.data.assets == ("BTC", "ETH")
    assert config.data.interval_minutes == 15
    assert config.backtest.holding_periods_minutes == (60, 240)


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[typo]\nvalue = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config(path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"interval_minutes": 5}, "either 15 or 60"),
        ({"start": "2026-01-01", "end": "2025-01-01"}, "cannot precede"),
        ({"assets": ("SOL",)}, "unsupported assets"),
    ],
)
def test_invalid_data_configuration_is_rejected(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DataConfig(**kwargs)


def test_invalid_surface_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported robust_loss"):
        SurfaceConfig(robust_loss="ordinary_magic")
