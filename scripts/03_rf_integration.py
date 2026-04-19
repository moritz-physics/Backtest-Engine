#!/usr/bin/env python
"""Risk-free rate integration demo.

Re-runs the MA crossover strategy from script 02, but with two
improvements:
  1. Uninvested capital earns the risk-free rate (DTB3 T-bill rate).
  2. Sharpe and Sortino use the actual time-varying risk-free rate
     instead of rf=0.

Prints a before/after comparison table and saves a figure comparing
the equity curves with and without cash earnings.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from backtester.backtest.engine import run_backtest
from backtester.costs.linear import LinearCost
from backtester.data.loader import load_prices, to_returns
from backtester.data.rates import load_risk_free_rate, to_daily_rate
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
OUTPUT_PATH = "results/03_rf_integration.png"


def main() -> None:
    # ── 1. Load data ──────────────────────────────────────────────────
    prices = load_prices(TICKER, START, END)
    returns = to_returns(prices, method="simple")
    returns = returns.iloc[1:]
    prices = prices.loc[returns.index]

    # Load risk-free rate
    rf_annual = load_risk_free_rate(START, END, series="DTB3")
    rf_daily = to_daily_rate(rf_annual)

    # ── 2. Compute MAs and signal ─────────────────────────────────────
    ma_fast = prices[TICKER].rolling(FAST_WINDOW).mean()
    ma_slow = prices[TICKER].rolling(SLOW_WINDOW).mean()
    raw_signal = (ma_fast > ma_slow).astype(float)
    signal = raw_signal.to_frame(name=TICKER)

    # ── 3. Trim to valid signal period ────────────────────────────────
    valid_mask = signal[TICKER].notna() & ma_slow.notna()
    first_valid = signal.index[valid_mask][0]
    signal = signal.loc[first_valid:]
    returns_bt = returns.loc[first_valid:]

    # ── 4. Run backtests: without and with cash rate ──────────────────
    cost_model = LinearCost(bps_per_trade=COST_BPS)

    result_no_cash = run_backtest(
        signal=signal,
        returns=returns_bt,
        cost_model=cost_model,
        signal_lag=SIGNAL_LAG,
    )

    result_with_cash = run_backtest(
        signal=signal,
        returns=returns_bt,
        cost_model=cost_model,
        signal_lag=SIGNAL_LAG,
        cash_rate=rf_daily,
    )

    # ── 5. Compute metrics ────────────────────────────────────────────
    bench = returns_bt[TICKER]
    rf_daily_bt = rf_daily.reindex(returns_bt.index)

    table = _build_comparison_table(
        result_no_cash, result_with_cash, bench, rf_daily_bt,
    )
    _print_table(table, first_valid, returns_bt.index[-1])

    # ── 6. Plot ───────────────────────────────────────────────────────
    _plot(result_no_cash, result_with_cash, bench,
          first_valid, returns_bt.index[-1])

    print(f"\nFigure saved to {OUTPUT_PATH}")


def _build_comparison_table(
    no_cash: object,
    with_cash: object,
    bench: pd.Series,
    rf_daily: pd.Series,
) -> pd.DataFrame:
    """Build a 4-row comparison table."""
    rows = {}

    configs = [
        ("Strategy (rf=0, no cash)", no_cash.portfolio_net_returns, 0.0),
        ("Strategy (rf=DTB3, +cash)", with_cash.portfolio_net_returns,
         rf_daily),
        ("Buy & Hold (rf=0)", bench, 0.0),
        ("Buy & Hold (rf=DTB3)", bench, rf_daily),
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

    # Add turnover
    t = turnover(with_cash.positions)
    rows["Strategy (rf=0, no cash)"]["Turnover (ann.)"] = t
    rows["Strategy (rf=DTB3, +cash)"]["Turnover (ann.)"] = t
    rows["Buy & Hold (rf=0)"]["Turnover (ann.)"] = 0.0
    rows["Buy & Hold (rf=DTB3)"]["Turnover (ann.)"] = 0.0

    return pd.DataFrame(rows).T


def _print_table(
    table: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    print("\n" + "=" * 80)
    print(f"  MA Crossover ({FAST_WINDOW}/{SLOW_WINDOW}) on {TICKER}"
          "  —  Risk-Free Rate Integration")
    print(f"  Backtest period: {start_date.date()} to {end_date.date()}")
    print(f"  Cost model: {COST_BPS} bps linear, signal lag = {SIGNAL_LAG}")
    print("  Risk-free rate: DTB3 (3-Month T-Bill)")
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


def _plot(
    no_cash: object,
    with_cash: object,
    bench: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    import pathlib
    pathlib.Path("results").mkdir(exist_ok=True)

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    eq_no_cash = (1 + no_cash.portfolio_net_returns).cumprod()
    eq_with_cash = (1 + with_cash.portfolio_net_returns).cumprod()
    eq_bench = (1 + bench).cumprod()

    ax_eq.plot(eq_no_cash.index, eq_no_cash.values,
               label="Strategy (no cash earnings)",
               linewidth=1.2, color="steelblue", linestyle="--")
    ax_eq.plot(eq_with_cash.index, eq_with_cash.values,
               label="Strategy (+ T-bill cash)",
               linewidth=1.2, color="darkblue")
    ax_eq.plot(eq_bench.index, eq_bench.values,
               label="Buy & Hold SPY",
               linewidth=1.2, color="gray", alpha=0.7)

    ax_eq.set_yscale("log")
    ax_eq.set_ylabel("Equity (log scale)")
    ax_eq.set_title(
        f"MA Crossover ({FAST_WINDOW}/{SLOW_WINDOW}) on {TICKER}  |  "
        f"Cash = DTB3 T-Bill  |  "
        f"{start_date.date()} to {end_date.date()}"
    )
    ax_eq.legend(loc="upper left", framealpha=0.9)
    ax_eq.grid(True, alpha=0.3)

    # Drawdown for the cash-earning strategy
    dd = drawdown_series(with_cash.portfolio_net_returns)
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
