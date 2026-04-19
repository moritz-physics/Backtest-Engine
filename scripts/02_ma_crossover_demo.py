#!/usr/bin/env python
"""Moving-average crossover demo on SPY (2010-2024).

Strategy: go long SPY when the 20-day SMA is above the 100-day SMA,
otherwise stay flat (cash).  Uses signal_lag=1 and 5 bps linear cost.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from backtester.backtest.engine import run_backtest
from backtester.costs.linear import LinearCost
from backtester.data.loader import load_prices, to_returns
from backtester.metrics.performance import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    cagr,
    drawdown_series,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)

# ── Parameters ────────────────────────────────────────────────────────
TICKER = "SPY"
START = "2010-01-01"
END = "2024-12-31"
FAST_WINDOW = 20
SLOW_WINDOW = 100
COST_BPS = 5.0
SIGNAL_LAG = 1
OUTPUT_PATH = "results/02_ma_crossover.png"


def main() -> None:
    # ── 1. Load data ──────────────────────────────────────────────────
    prices = load_prices(TICKER, START, END)
    returns = to_returns(prices, method="simple")

    # Drop row 0 (NaN from to_returns)
    returns = returns.iloc[1:]
    prices = prices.loc[returns.index]

    # ── 2. Compute MAs and signal ─────────────────────────────────────
    ma_fast = prices[TICKER].rolling(FAST_WINDOW).mean()
    ma_slow = prices[TICKER].rolling(SLOW_WINDOW).mean()
    raw_signal = (ma_fast > ma_slow).astype(float)

    # Wrap in DataFrame with matching column
    signal = raw_signal.to_frame(name=TICKER)

    # ── 3. Trim to valid signal period (post MA warmup) ───────────────
    first_valid = signal.first_valid_index()
    # The slow MA needs SLOW_WINDOW days; find first non-NaN
    valid_mask = signal[TICKER].notna() & ma_slow.notna()
    first_valid = signal.index[valid_mask][0]

    signal = signal.loc[first_valid:]
    returns_bt = returns.loc[first_valid:]

    # ── 4. Run backtest ───────────────────────────────────────────────
    cost_model = LinearCost(bps_per_trade=COST_BPS)
    result = run_backtest(
        signal=signal,
        returns=returns_bt,
        cost_model=cost_model,
        signal_lag=SIGNAL_LAG,
    )

    # ── 5. Compute metrics ────────────────────────────────────────────
    strat_net = result.portfolio_net_returns
    strat_gross = result.portfolio_gross_returns
    bench = returns_bt[TICKER]

    metrics_table = _compute_metrics_table(strat_gross, strat_net, bench,
                                           result.positions)
    _print_metrics(metrics_table, first_valid, returns_bt.index[-1])

    # ── 6. Plot ───────────────────────────────────────────────────────
    _plot(strat_gross, strat_net, bench, first_valid, returns_bt.index[-1])

    print(f"\nFigure saved to {OUTPUT_PATH}")


def _compute_metrics_table(
    gross: pd.Series,
    net: pd.Series,
    bench: pd.Series,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """Build a comparison table: strategy gross, strategy net, benchmark."""
    rows = {}
    for label, r in [("Strategy (gross)", gross),
                     ("Strategy (net)", net),
                     ("Buy & Hold SPY", bench)]:
        rows[label] = {
            "Ann. Return": annualized_return(r),
            "CAGR": cagr(r),
            "Ann. Volatility": annualized_volatility(r),
            "Sharpe": sharpe_ratio(r),
            "Sortino": sortino_ratio(r),
            "Max Drawdown": max_drawdown(r),
            "Calmar": calmar_ratio(r),
            "Hit Rate": hit_rate(r),
        }

    # Add turnover only for strategy rows
    t = turnover(positions)
    rows["Strategy (gross)"]["Turnover (ann.)"] = t
    rows["Strategy (net)"]["Turnover (ann.)"] = t
    rows["Buy & Hold SPY"]["Turnover (ann.)"] = 0.0

    return pd.DataFrame(rows).T


def _print_metrics(
    table: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    """Print a formatted metrics comparison table."""
    print("\n" + "=" * 72)
    print(f"  MA Crossover ({FAST_WINDOW}/{SLOW_WINDOW}) on {TICKER}")
    print(f"  Backtest period: {start_date.date()} to {end_date.date()}")
    print(f"  Cost model: {COST_BPS} bps linear, signal lag = {SIGNAL_LAG}")
    print("=" * 72)

    # Format the table for display
    fmt = table.copy()
    pct_cols = ["Ann. Return", "CAGR", "Ann. Volatility",
                "Max Drawdown", "Hit Rate"]
    for col in pct_cols:
        fmt[col] = fmt[col].map(lambda x: f"{x:+.2%}" if col != "Hit Rate"
                                else f"{x:.1%}")

    ratio_cols = ["Sharpe", "Sortino", "Calmar"]
    for col in ratio_cols:
        fmt[col] = fmt[col].map(lambda x: f"{x:.2f}")

    fmt["Turnover (ann.)"] = fmt["Turnover (ann.)"].map(lambda x: f"{x:.1f}")

    print(fmt.to_string())
    print()


def _plot(
    gross: pd.Series,
    net: pd.Series,
    bench: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    """Equity curve + drawdown figure."""
    import pathlib
    pathlib.Path("results").mkdir(exist_ok=True)

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Equity curves
    eq_gross = (1 + gross).cumprod()
    eq_net = (1 + net).cumprod()
    eq_bench = (1 + bench).cumprod()

    ax_eq.plot(eq_gross.index, eq_gross.values, label="Strategy (gross)",
               linewidth=1.2, color="steelblue")
    ax_eq.plot(eq_net.index, eq_net.values, label="Strategy (net)",
               linewidth=1.2, color="darkblue")
    ax_eq.plot(eq_bench.index, eq_bench.values, label="Buy & Hold SPY",
               linewidth=1.2, color="gray", alpha=0.7)

    ax_eq.set_yscale("log")
    ax_eq.set_ylabel("Equity (log scale)")
    ax_eq.set_title(
        f"MA Crossover ({FAST_WINDOW}/{SLOW_WINDOW}) on {TICKER}  |  "
        f"{start_date.date()} to {end_date.date()}"
    )
    ax_eq.legend(loc="upper left", framealpha=0.9)
    ax_eq.grid(True, alpha=0.3)

    # Drawdown
    dd = drawdown_series(net)
    ax_dd.fill_between(dd.index, dd.values, 0, color="darkblue", alpha=0.3)
    ax_dd.plot(dd.index, dd.values, color="darkblue", linewidth=0.8)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
