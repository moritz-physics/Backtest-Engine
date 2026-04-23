# Backtest Engine

A small, vectorized backtesting framework for systematic equity strategies,
written in Python from first principles as a learning project in
quantitative finance. It is designed for a single reader — me, or someone
like me — who wants to understand how a backtest actually works rather than
drive one through a black box. Correctness comes before cleverness:
look-ahead bias is prevented by construction, transaction costs are always
charged, uninvested capital can optionally earn a real risk-free rate, and
every claim in the `metrics` module has a unit test behind it. Gross and
net performance are always reported side by side.

![SPY 20/100 moving-average crossover equity curve with drawdown panel, benchmarked against buy-and-hold SPY over 2010–2024.](results/02_ma_crossover.png)

## What's inside

- **`backtester.data`** — `load_prices`, `load_ohlcv`, `to_returns` for
  adjusted-close equity data from Yahoo Finance with a local parquet cache;
  `load_risk_free_rate`, `to_daily_rate` for FRED T-bill/Fed-Funds series.
- **`backtester.backtest`** — `run_backtest`, the vectorized engine. Takes a
  target-position DataFrame and a simple-returns DataFrame, shifts by
  `signal_lag` to kill look-ahead, applies costs, and returns a frozen
  `BacktestResult` dataclass.
- **`backtester.costs`** — `LinearCost`, a proportional cost model charging
  a per-unit spread in basis points on every position change.
- **`backtester.features`** — indicator primitives for signal construction
  (currently Wilder-smoothed RSI).
- **`backtester.metrics`** — Sharpe, Sortino, Calmar, CAGR vs. arithmetic
  annualized return, drawdown series, turnover, hit rate. Formulas are
  documented in every docstring.
- **`scripts/`** — runnable demos, numbered in the order they were built.
- **`tests/`** — pytest suite. PnL arithmetic, cost accounting, cash
  accounting, and metric identities are checked against hand-computed
  values.

Deeper notes on each area:

- [Data loader and risk-free rates](docs/01_data_loader.md)
- [Backtest engine and metrics](docs/02_backtest_engine.md)
- [Demo findings](docs/03_demo_findings.md)

## Quick start

Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Run a demo:

```bash
uv run python scripts/02_ma_crossover_demo.py
```

Minimal usage inside your own script:

```python
from backtester.data.loader import load_prices, to_returns
from backtester.backtest.engine import run_backtest
from backtester.costs.linear import LinearCost
from backtester.metrics.performance import sharpe_ratio, max_drawdown

prices = load_prices("SPY", "2010-01-01", "2024-12-31")
returns = to_returns(prices, method="simple").iloc[1:]

fast = prices["SPY"].rolling(20).mean()
slow = prices["SPY"].rolling(100).mean()
signal = (fast > slow).astype(float).to_frame("SPY").loc[slow.first_valid_index():]
returns = returns.loc[signal.index]

result = run_backtest(signal, returns, cost_model=LinearCost(bps_per_trade=5.0))

print(f"Sharpe (net): {sharpe_ratio(result.portfolio_net_returns):.2f}")
print(f"Max DD (net): {max_drawdown(result.portfolio_net_returns):.1%}")
```

Figures produced by the demos land in `results/`:

```bash
ls results/
# 01_data_overview.png  02_ma_crossover.png  03_rf_integration.png  05_rsi_mean_reversion.png
```

## Design principles

- **No look-ahead bias.** `signal_lag >= 1` is enforced in the engine; the
  position on day *t* is always derived from data available strictly before
  *t*.
- **Explicit NaN handling.** Loaders either drop via inner-join or preserve
  NaN via outer-join, never silently fill. `to_returns` raises if
  mid-series NaN appears.
- **Costs are always on.** A cost model is a mandatory input to every
  demo; 5 bps is the default assumption for liquid US large-cap names.
- **Gross and net are always reported.** Both portfolio series are
  computed and printed, so cost impact is never hidden.
- **Vectorized first.** Every transformation is a pandas/numpy
  expression; the engine has no per-day loop.
- **Tested to known answers.** PnL, turnover, Sharpe, drawdown, and the
  `signal_lag` invariant are all pinned to hand-computed expected values
  in `tests/`.

## What I learned

- **MA(20/100) on SPY loses to buy-and-hold but roughly halves the
  drawdown.** Over 2010–2024 the crossover delivers 8.5% CAGR vs. 14.5%
  for B&H, with max drawdown 22% vs. 34%. On Sharpe (0.73 vs. 0.88) B&H
  still wins. The lesson is not that trend-following is broken, but that
  a bull-market sample makes any long-flat strategy look worse than
  always-long.
- **Using a real risk-free rate makes the Sharpe lower, not higher.**
  When uninvested capital earns DTB3 and excess return is computed
  against the same series, strategy Sharpe drops from 0.73 to 0.65 and
  B&H Sharpe drops from 0.88 to 0.81. The rf=0 convention silently
  inflates Sharpe ratios across the literature.
- **RSI(14) mean-reversion on SPY is a negative finding, cleanly.** 3.4%
  CAGR, Sharpe 0.37, turnover 7× higher than the MA crossover. The
  strategy spends most of the sample flat, shorts overbought conditions
  that persist through bull runs, and bleeds to costs. Expected, but
  worth showing honestly rather than tuning into a false positive.
- **Honest cost and cash accounting narrow — but do not eliminate — the
  gap between strategies and benchmarks.** Most of the visible alpha in
  naive backtests lives in rf=0 and zero-cost assumptions.
- **The hardest bugs are in data alignment, not in strategy logic.** Most
  of the defects fixed during construction were index mismatches,
  warmup-period trimming, and NaN leaks, not formula errors.

## Next steps

- **Pairs / cointegration trading.** Engle–Granger and Johansen tests,
  z-score entries on a stationary spread.
- **Walk-forward validation.** Implement the `splits/` module so that
  in-sample parameter choices are validated on genuinely unseen data.
- **Deep hedging.** Replace a rule-based signal with a small neural
  policy trained end-to-end on hedging P&L, as an exploration of where
  learned strategies actually help.

## References

- Marcos López de Prado, *Advances in Financial Machine Learning*, 2018.
- Ernest P. Chan, *Quantitative Trading* and *Algorithmic Trading*.
- Gatev, Goetzmann & Rouwenhorst, *Pairs Trading: Performance of a
  Relative-Value Arbitrage Rule*, RFS 2006 — for the pairs work ahead.
- J. Welles Wilder, *New Concepts in Technical Trading Systems*, 1978 —
  for the original RSI and Wilder smoothing conventions.

## License and author

MIT. Moritz Heidtmann, 2026.
