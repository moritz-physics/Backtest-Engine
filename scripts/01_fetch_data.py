"""Fetch SPY, QQQ, IWM and produce a 2x2 data overview figure.

Usage
-----
    uv run python scripts/01_fetch_data.py

Output
------
    results/01_data_overview.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from backtester.data.loader import load_prices, to_returns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Matplotlib style — set once
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

COLORS = {"SPY": "#1f77b4", "QQQ": "#ff7f0e", "IWM": "#2ca02c"}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tickers = ["SPY", "QQQ", "IWM"]
    start, end = "2015-01-01", "2024-12-31"

    prices = load_prices(tickers, start, end)
    returns = to_returns(prices, method="simple")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Normalized price levels, log y-axis
    ax = axes[0, 0]
    normalized = prices / prices.iloc[0] * 100
    for ticker in tickers:
        ax.plot(normalized.index, normalized[ticker],
                label=ticker, color=COLORS[ticker])
    ax.set_yscale("log")
    ax.set_ylabel("Normalized Price (start = 100)")
    ax.set_title("Normalized Price Levels")
    ax.legend()

    # (b) Daily returns overlay
    ax = axes[0, 1]
    for ticker in tickers:
        ax.plot(returns.index, returns[ticker],
                label=ticker, color=COLORS[ticker], alpha=0.4, linewidth=0.5)
    ax.set_ylabel("Daily Return")
    ax.set_title("Daily Returns")
    ax.legend()

    # (c) Rolling 60-day annualized volatility
    ax = axes[1, 0]
    rolling_vol = returns.rolling(60).std() * np.sqrt(252)
    for ticker in tickers:
        ax.plot(rolling_vol.index, rolling_vol[ticker],
                label=ticker, color=COLORS[ticker])
    ax.set_ylabel("Annualized Volatility")
    ax.set_title("Rolling 60-Day Volatility (Annualized)")
    ax.legend()

    # (d) Correlation heatmap of daily returns
    ax = axes[1, 1]
    corr = returns.dropna().corr()
    im = ax.imshow(corr.values, vmin=0.5, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers)
    # Annotate cells
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if corr.values[i, j] > 0.8 else "black")
    ax.set_title("Return Correlations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = Path("results/01_data_overview.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)


if __name__ == "__main__":
    main()
