#!/usr/bin/env python
"""RSI(14) mean-reversion strategy on SPY (2010-2024).

Strategy
--------
- Go long  (+1) when RSI(14) < 30  (oversold, expect upward reversion)
- Go flat  ( 0) when 30 <= RSI <= 70
- Go short (-1) when RSI(14) > 70  (overbought, expect downward reversion)

Compared against: buy-and-hold SPY and the MA(20/100) crossover from
demo 02, both evaluated over the same date range.

Background
----------
RSI = Relative Strength Index (Wilder, 1978).  Measures the ratio of
average gains to average losses over a lookback window, scaled to 0–100.
Classical thresholds: <30 oversold, >70 overbought.  The premise is that
extremes mean-revert — works in some asset classes and regimes, fails in
others.

This script uses Wilder's exponential smoothing (ewm with alpha=1/14),
which is the original convention.  The engine's signal_lag=1 ensures no
look-ahead bias: the signal computed from data up to day t is first
acted upon on day t+1.

Prior expectation: RSI mean-reversion on SPY daily is historically
mediocre-to-bad over 2010-2024 (a persistent uptrend with few
mean-reverting episodes).  If Sharpe > 1 appears, suspect a bug.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester.backtest.engine import run_backtest
from backtester.costs.linear import LinearCost
from backtester.data.loader import load_prices, to_returns
from backtester.data.rates import load_risk_free_rate, to_daily_rate
from backtester.features.indicators import rsi
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
TICKERS = ["SPY"]  # e.g. ["SPY"], ["SPY", "QQQ"], ["AAPL", "MSFT", "TSLA"]
START = "2010-01-01"
END = "2024-12-31"
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MA_FAST = 20
MA_SLOW = 100
COST_BPS = 5.0
SIGNAL_LAG = 1
OUTPUT_PATH = "results/05_rsi_mean_reversion.png"


def main() -> None:
    ticker_label = ", ".join(TICKERS)

    # ── 1. Load data ──────────────────────────────────────────────────
    prices = load_prices(TICKERS, START, END)
    returns = to_returns(prices, method="simple")
    returns = returns.iloc[1:]  # drop first NaN row
    prices = prices.loc[returns.index]

    # Risk-free rate
    rf_annual = load_risk_free_rate(START, END, series="DTB3")
    rf_daily = to_daily_rate(rf_annual)

    # ── 2. Compute RSI signal ────────────────────────────────────────
    rsi_values = rsi(prices, period=RSI_PERIOD)

    # No look-ahead: RSI at time t uses only prices up to t.
    # signal_lag=1 in run_backtest shifts positions by 1 day,
    # so the position on day t+1 is based on RSI computed from
    # data up to day t.  Both mechanisms together prevent look-ahead.
    rsi_signal = pd.DataFrame(
        np.where(rsi_values < RSI_OVERSOLD, 1.0,
        np.where(rsi_values > RSI_OVERBOUGHT, -1.0, 0.0)),
        index=prices.index,
        columns=prices.columns,
    )

    # ── 3. Compute MA crossover signal (inline, same as demo 02) ─────
    ma_fast = prices.rolling(MA_FAST).mean()
    ma_slow = prices.rolling(MA_SLOW).mean()
    ma_signal = (ma_fast > ma_slow).astype(float)

    # ── 4. Trim to common valid period ────────────────────────────────
    # MA crossover needs SLOW_WINDOW days warmup; RSI needs RSI_PERIOD.
    # Use the later of the two so both strategies are evaluated over
    # identical dates for a fair comparison.
    # For DataFrames, first_valid_index returns the first row where
    # *any* column is non-NaN; we need *all* columns valid.
    rsi_first = rsi_values.dropna().index[0]
    ma_first = ma_slow.dropna().index[0]
    first_valid = max(rsi_first, ma_first)

    rsi_signal = rsi_signal.loc[first_valid:]
    ma_signal = ma_signal.loc[first_valid:]
    returns_bt = returns.loc[first_valid:]
    rsi_bt = rsi_values.loc[first_valid:]

    # ── 5. Run backtests ──────────────────────────────────────────────
    cost_model = LinearCost(bps_per_trade=COST_BPS)

    result_rsi = run_backtest(
        signal=rsi_signal,
        returns=returns_bt,
        cost_model=cost_model,
        signal_lag=SIGNAL_LAG,
        cash_rate=rf_daily,
    )

    result_ma = run_backtest(
        signal=ma_signal,
        returns=returns_bt,
        cost_model=cost_model,
        signal_lag=SIGNAL_LAG,
        cash_rate=rf_daily,
    )

    # ── 6. Compute metrics ────────────────────────────────────────────
    rf_daily_bt = rf_daily.reindex(returns_bt.index)
    # Benchmark: equal-weight buy-and-hold across all tickers
    bench = returns_bt.mean(axis=1)
    bench.name = "benchmark"

    table = _build_metrics_table(result_rsi, result_ma, bench, rf_daily_bt,
                                 ticker_label)
    _print_metrics(table, first_valid, returns_bt.index[-1], ticker_label)

    # ── 7. Print interpretation ───────────────────────────────────────
    _print_interpretation(result_rsi, result_ma, rsi_signal, ticker_label)

    # ── 8. Plot ───────────────────────────────────────────────────────
    _plot(result_rsi, result_ma, bench, rsi_bt,
          first_valid, returns_bt.index[-1], ticker_label)

    print(f"\nFigure saved to {OUTPUT_PATH}")


def _build_metrics_table(
    result_rsi: object,
    result_ma: object,
    bench: pd.Series,
    rf_daily: pd.Series,
    ticker_label: str,
) -> pd.DataFrame:
    """Build a 3-row comparison table: RSI net, MA net, B&H."""
    rows = {}
    configs = [
        ("RSI(14) Mean-Rev (net)", result_rsi.portfolio_net_returns, rf_daily),
        ("MA(20/100) Trend (net)", result_ma.portfolio_net_returns, rf_daily),
        (f"Buy & Hold {ticker_label}", bench, rf_daily),
    ]

    for label, r, rf in configs:
        rows[label] = {
            "Ann. Return": annualized_return(r),
            "CAGR": cagr(r),
            "Ann. Volatility": annualized_volatility(r),
            "Sharpe": sharpe_ratio(r, rf=rf),
            "Sortino": sortino_ratio(r, rf=rf),
            "Max Drawdown": max_drawdown(r),
            "Calmar": calmar_ratio(r),
            "Hit Rate": hit_rate(r),
        }

    rows["RSI(14) Mean-Rev (net)"]["Turnover (ann.)"] = turnover(
        result_rsi.positions)
    rows["MA(20/100) Trend (net)"]["Turnover (ann.)"] = turnover(
        result_ma.positions)
    rows[f"Buy & Hold {ticker_label}"]["Turnover (ann.)"] = 0.0

    return pd.DataFrame(rows).T


def _print_metrics(
    table: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    ticker_label: str,
) -> None:
    print("\n" + "=" * 80)
    print(f"  RSI(14) Mean-Reversion vs MA(20/100) Trend vs B&H  |  "
          f"{ticker_label}")
    print(f"  Backtest period: {start_date.date()} to {end_date.date()}")
    print(f"  Cost model: {COST_BPS} bps linear, signal lag = {SIGNAL_LAG}")
    print(f"  Cash rate: DTB3 (3-Month T-Bill)")
    print("=" * 80)

    fmt = table.copy()
    pct_cols = ["Ann. Return", "CAGR", "Ann. Volatility",
                "Max Drawdown", "Hit Rate"]
    for col in pct_cols:
        fmt[col] = fmt[col].map(
            lambda x, c=col: f"{x:+.2%}" if c != "Hit Rate"
            else f"{x:.1%}"
        )

    ratio_cols = ["Sharpe", "Sortino", "Calmar"]
    for col in ratio_cols:
        fmt[col] = fmt[col].map(lambda x: f"{x:.2f}")

    fmt["Turnover (ann.)"] = fmt["Turnover (ann.)"].map(lambda x: f"{x:.1f}")

    print(fmt.to_string())
    print()


def _print_interpretation(
    result_rsi: object,
    result_ma: object,
    rsi_signal: pd.DataFrame,
    ticker_label: str,
) -> None:
    """Print an honest interpretation of results."""
    # State distribution for RSI strategy (pre-lag signal), averaged
    # across all tickers (each ticker contributes one position per day).
    n_total = rsi_signal.size  # rows * columns
    pct_long = (rsi_signal == 1.0).sum().sum() / n_total * 100
    pct_flat = (rsi_signal == 0.0).sum().sum() / n_total * 100
    pct_short = (rsi_signal == -1.0).sum().sum() / n_total * 100

    rsi_turn = turnover(result_rsi.positions)
    ma_turn = turnover(result_ma.positions)

    print("─" * 80)
    print("  Interpretation")
    print("─" * 80)
    print(f"  RSI signal state distribution (across {ticker_label}):")
    print(f"    Long  (RSI < 30): {pct_long:5.1f}% of asset-days")
    print(f"    Flat  (30-70):    {pct_flat:5.1f}% of asset-days")
    print(f"    Short (RSI > 70): {pct_short:5.1f}% of asset-days")
    print()
    print(f"  Annualized turnover: RSI = {rsi_turn:.1f},  MA = {ma_turn:.1f}")
    print(f"  RSI turnover is {rsi_turn / ma_turn:.1f}x that of MA crossover."
          if ma_turn > 0 else "")
    print()
    print("  Conclusion:")
    print(f"  Over {START} to {END}, RSI mean-reversion suffers in trending")
    print("  markets: it goes short at overbought levels that persist in")
    print("  bull runs, and spends most time flat (missing the trend).  The")
    print("  MA crossover, being a trend-following strategy, captures")
    print("  uptrends far more effectively.  The higher turnover of RSI")
    print("  further erodes returns via costs.  RSI may add value in")
    print("  range-bound regimes or as a filter within a larger system.")
    print()


def _plot(
    result_rsi: object,
    result_ma: object,
    bench: pd.Series,
    rsi_values: pd.DataFrame | pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    ticker_label: str,
) -> None:
    pathlib.Path("results").mkdir(exist_ok=True)

    fig, (ax_eq, ax_rsi, ax_dd) = plt.subplots(
        3, 1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [4, 2.5, 3.5]},
        sharex=True,
    )

    # ── Top: equity curves (log scale) ────────────────────────────────
    rsi_net = result_rsi.portfolio_net_returns
    ma_net = result_ma.portfolio_net_returns

    eq_rsi = (1 + rsi_net).cumprod()
    eq_ma = (1 + ma_net).cumprod()
    eq_bench = (1 + bench).cumprod()

    ax_eq.plot(eq_bench.index, eq_bench.values,
               label=f"Buy & Hold {ticker_label}", linewidth=1.2,
               color="gray", alpha=0.7)
    ax_eq.plot(eq_ma.index, eq_ma.values,
               label="MA(20/100) Trend", linewidth=1.2,
               color="steelblue")
    ax_eq.plot(eq_rsi.index, eq_rsi.values,
               label="RSI(14) Mean-Rev", linewidth=1.2,
               color="darkorange")

    ax_eq.set_yscale("log")
    ax_eq.set_ylabel("Equity (log scale)")
    ax_eq.set_title(
        f"RSI(14) Mean-Reversion vs MA(20/100) Trend vs B&H  |  "
        f"{ticker_label} {start_date.date()} to {end_date.date()}"
    )
    ax_eq.legend(loc="upper left", framealpha=0.9)
    ax_eq.grid(True, alpha=0.3)

    # ── Middle: RSI time series with thresholds ───────────────────────
    # Plot each ticker's RSI as a separate line
    rsi_df = rsi_values if isinstance(rsi_values, pd.DataFrame) else rsi_values.to_frame()
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(rsi_df.columns), 3)))
    for i, col in enumerate(rsi_df.columns):
        ax_rsi.plot(rsi_df.index, rsi_df[col].values,
                    linewidth=0.6, color=colors[i], alpha=0.7,
                    label=f"RSI {col}")
    ax_rsi.axhline(RSI_OVERSOLD, color="green", linestyle="--",
                   linewidth=0.8, alpha=0.7, label=f"RSI = {RSI_OVERSOLD}")
    ax_rsi.axhline(RSI_OVERBOUGHT, color="red", linestyle="--",
                   linewidth=0.8, alpha=0.7, label=f"RSI = {RSI_OVERBOUGHT}")
    ax_rsi.set_ylabel("RSI(14)")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax_rsi.grid(True, alpha=0.3)

    # ── Bottom: drawdowns for both strategies ─────────────────────────
    dd_rsi = drawdown_series(rsi_net)
    dd_ma = drawdown_series(ma_net)

    ax_dd.fill_between(dd_rsi.index, dd_rsi.values, 0,
                       color="darkorange", alpha=0.2)
    ax_dd.plot(dd_rsi.index, dd_rsi.values, color="darkorange",
               linewidth=0.8, label="RSI(14) Mean-Rev")
    ax_dd.fill_between(dd_ma.index, dd_ma.values, 0,
                       color="steelblue", alpha=0.2)
    ax_dd.plot(dd_ma.index, dd_ma.values, color="steelblue",
               linewidth=0.8, label="MA(20/100) Trend")

    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.legend(loc="lower left", framealpha=0.9, fontsize=8)
    ax_dd.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
