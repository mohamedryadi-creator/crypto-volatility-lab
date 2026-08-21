"""Small, deterministic reporting helpers for research artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import ArrayLike

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _target(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _array(values: ArrayLike, name: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return result


def save_smile_plot(
    log_moneyness: ArrayLike,
    market_iv: ArrayLike,
    grid_log_moneyness: ArrayLike,
    fitted_iv: ArrayLike,
    path: str | Path,
    *,
    title: str,
) -> Path:
    """Save an observed-versus-fitted volatility smile."""

    k_market = _array(log_moneyness, "log_moneyness")
    iv_market = _array(market_iv, "market_iv")
    k_grid = _array(grid_log_moneyness, "grid_log_moneyness")
    iv_grid = _array(fitted_iv, "fitted_iv")
    if k_market.shape != iv_market.shape or k_grid.shape != iv_grid.shape:
        raise ValueError("x and y arrays must have matching shapes")

    target = _target(path)
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    axis.scatter(k_market, 100.0 * iv_market, color="#0B7285", s=28, label="Synthetic quotes")
    axis.plot(k_grid, 100.0 * iv_grid, color="#E8590C", linewidth=2.2, label="SSVI fit")
    axis.axvline(0.0, color="#868E96", linewidth=0.8, linestyle="--")
    axis.set(
        title=title,
        xlabel="log-forward moneyness  k = log(K/F)",
        ylabel="Implied volatility (%)",
    )
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    figure.savefig(target, dpi=170)
    plt.close(figure)
    return target


def save_density_plot(
    strikes: ArrayLike,
    density: ArrayLike,
    path: str | Path,
    *,
    forward: float,
    title: str,
) -> Path:
    """Save a risk-neutral density with its forward marked."""

    strike_values = _array(strikes, "strikes")
    density_values = _array(density, "density")
    if strike_values.shape != density_values.shape:
        raise ValueError("strikes and density must have matching shapes")
    target = _target(path)
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    axis.fill_between(strike_values, density_values, color="#4C6EF5", alpha=0.24)
    axis.plot(strike_values, density_values, color="#364FC7", linewidth=2.0)
    axis.axvline(forward, color="#E03131", linewidth=1.2, linestyle="--", label="Forward")
    axis.set(title=title, xlabel="Strike", ylabel="Risk-neutral density")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    figure.savefig(target, dpi=170)
    plt.close(figure)
    return target


def save_equity_plot(
    curves: Mapping[str, Sequence[tuple[datetime, float]]],
    path: str | Path,
    *,
    title: str,
) -> Path:
    """Save one or more cumulative-P&L curves."""

    if not curves:
        raise ValueError("at least one equity curve is required")
    target = _target(path)
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    nonempty = 0
    for label, curve in curves.items():
        if not curve:
            continue
        times, values = zip(*curve, strict=True)
        axis.plot(times, values, linewidth=1.8, label=label)
        nonempty += 1
    if nonempty == 0:
        plt.close(figure)
        raise ValueError("at least one equity curve must be non-empty")
    axis.axhline(0.0, color="#868E96", linewidth=0.8)
    axis.set(title=title, xlabel="Exit time (UTC)", ylabel="Cumulative P&L (USD)")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    figure.autofmt_xdate()
    figure.savefig(target, dpi=170)
    plt.close(figure)
    return target


def write_json_report(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a stable, human-readable JSON report."""

    target = _target(path)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str, allow_nan=False)
        handle.write("\n")
    temporary.replace(target)
    return target


__all__ = [
    "save_density_plot",
    "save_equity_plot",
    "save_smile_plot",
    "write_json_report",
]
