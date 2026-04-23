# Backtest engine and metrics

The engine lives in `backtester.backtest.engine` and exposes a single
function, `run_backtest`, which returns a frozen `BacktestResult`
dataclass containing positions, gross returns, net returns, costs,
turnover, and the two portfolio-level series.

## From signal to PnL

The engine takes two aligned DataFrames (same DatetimeIndex, same
columns) and a few scalars. The pipeline is entirely vectorized:

```
  signal (target weights, indexed by date, one col per asset)
        │
        ▼  shift forward by signal_lag (>= 1), fill leading rows with 0
  positions
        │
        ├── × returns        ──► gross_returns (per-asset)
        │   + (1 − |pos|) × cash_rate  (if cash_rate provided)
        │
        └── |Δ position|     ──► turnover_series
                              ──► cost_series = turnover × bps / 10_000

  net_returns           = gross_returns − cost_series
  portfolio_gross (Σ)   = gross_returns.sum(axis=1)
  portfolio_net   (Σ)   = net_returns.sum(axis=1)
```

Every step is a pandas/numpy expression. There is no per-day loop.

## The `signal_lag=1` invariant

`signal_lag` is validated to be `>= 1`; `run_backtest` raises
`ValueError` otherwise. This is the single most important rule in the
engine, because every form of look-ahead bias eventually reduces to
"the position used on day *t* was informed by data from day *t* or
later." A lag of 1 means: today's position was determined by yesterday's
signal, which was computed from data up to and including yesterday's
close. The first `signal_lag` rows of the backtest are flat (position
zero), which is a minor loss of sample but the correct default.

The invariant is also pinned by a unit test in `tests/test_engine.py`
that constructs a signal perfectly anticipating the next day's return
and verifies that PnL is exactly as lagged as expected. If someone
later changes the shift direction or drops the validation, the test
fails immediately.

## Cost model

`LinearCost(bps_per_trade=5.0)` charges a proportional cost on every
unit of absolute position change:

```
cost_t = |position_t − position_{t−1}| × bps / 10_000
```

Row 0 assumes a prior position of zero, so entering an initial 100%
long position at 5 bps costs 5 bps of NAV on day 0. The default of 5
bps is appropriate for liquid US large-cap ETFs; it is too tight for
single-name equities and wildly too tight for small-caps or
international names, which should be noted when interpreting results
on those universes.

## Cash accounting

When a `cash_rate` pd.Series (already in daily units) is passed,
uninvested capital earns it:

```
effective_return_t = position_t × return_t
                   + (1 − |position_t|) × cash_rate_t
```

For a fully invested strategy (|position| = 1) the cash term is zero.
For a flat position (|position| = 0) the full cash rate is earned. For
a leveraged position (|position| > 1) the formula yields a negative
cash term, which is the correct sign for a borrowing cost — though the
magnitude uses the lending rate as a stand-in for borrow, so leveraged
backtests will be *optimistic* about financing.

The Sharpe and Sortino functions accept the same `cash_rate` series as
`rf`, so excess returns are computed consistently.

## Metric definitions

All metrics are in `backtester.metrics.performance` and take a daily
simple-returns `pd.Series`. *N* = `periods_per_year`, default 252.

- **`annualized_return(r)`** — `(1 + mean(r))^N − 1`. Arithmetic mean
  compounded. Overstates realised return when volatility is high.
- **`cagr(r)`** — `prod(1 + r)^(N/T) − 1`. True realised geometric
  return; the honest counterpart to `annualized_return`.
- **`annualized_volatility(r)`** — `std(r, ddof=1) × sqrt(N)`. Sample
  standard deviation (the pandas default), annualized by √N.
- **`sharpe_ratio(r, rf=0)`** — `mean(r − rf) / std(r − rf) × sqrt(N)`.
  `rf` is a scalar annualized rate or a series of daily rates.
- **`sortino_ratio(r, rf=0)`** — same as Sharpe but the denominator is
  downside deviation `sqrt(mean(min(r − rf, 0)^2))`.
- **`max_drawdown(r)`** — minimum of `(W − cummax(W)) / cummax(W)` on
  the wealth series `W = cumprod(1 + r)`, with the running peak
  clipped at 1.0 so a day-one loss registers as a drawdown.
- **`calmar_ratio(r)`** — `annualized_return(r) / |max_drawdown(r)|`.
- **`turnover(positions)`** — `mean(|Δ position|.sum(axis=1)) × N`;
  annualized one-way turnover summed across assets.
- **`hit_rate(r)`** — `(r > 0).mean()`, fraction of positive days.

Zero-denominator edge cases (constant returns, no drawdown) return
`inf`, `-inf`, or `nan` depending on the sign of the numerator, and
are covered by unit tests.
