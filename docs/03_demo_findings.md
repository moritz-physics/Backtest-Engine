# Demo findings

Four demo scripts live in `scripts/` and write their figures to
`results/`. All are run on SPY over 2010-01-01 to 2024-12-31 (backtest
period starts 2010-05-27 after MA warmup) with 5 bps linear costs and
`signal_lag=1`. Numbers below are the values printed by the current
code on the cached data.

## 01 — Data overview

`scripts/01_fetch_data.py` fetches SPY, QQQ, IWM, plots adjusted-close,
cumulative returns, drawdowns, and rolling annualized volatility in a
2×2 grid. Output: `results/01_data_overview.png`. Its purpose is to
exercise the loader and sanity-check the data before any strategy is
built; no backtest is run.

## 02 — MA(20/100) crossover on SPY

`scripts/02_ma_crossover_demo.py`. Go long when the 20-day SMA is above
the 100-day SMA, otherwise flat. Figure: `results/02_ma_crossover.png`.

|                    | Strategy (net) | Buy & Hold |
|--------------------|---------------:|-----------:|
| CAGR               |          8.49% |     14.51% |
| Ann. volatility    |         12.17% |     17.01% |
| Sharpe (rf = 0)    |           0.73 |       0.88 |
| Max drawdown       |        −22.01% |    −33.72% |
| Turnover (ann.)    |            2.4 |        0.0 |

Honest interpretation: the strategy loses on total return and on Sharpe
to buy-and-hold, but cuts the worst drawdown by roughly a third. This
is the textbook trade-off of a simple trend filter in a secular bull
market — you pay return for drawdown control. It is not a positive
finding, but it is not a broken strategy either; it is a correctly
calibrated baseline against which more interesting strategies can be
compared.

## 03 — Risk-free rate integration

`scripts/03_rf_integration.py` re-runs the MA(20/100) strategy with two
changes: uninvested capital earns DTB3 (3-month T-bill), and Sharpe /
Sortino use the same DTB3 series as `rf`. Figure:
`results/03_rf_integration.png`.

|                                  |  CAGR | Sharpe | Max DD |
|----------------------------------|------:|-------:|-------:|
| Strategy (rf = 0, no cash)       | 8.49% |   0.73 | −22.0% |
| Strategy (rf = DTB3, cash earns) | 8.74% |   0.65 | −22.0% |
| Buy & hold (rf = 0)              | 14.5% |   0.88 | −33.7% |
| Buy & hold (rf = DTB3)           | 14.5% |   0.81 | −33.7% |

Honest interpretation: enabling cash earnings slightly *raises* the
strategy's CAGR (now 8.74%), because a meaningful fraction of the
sample is spent flat and earning rf. But Sharpe *falls*, from 0.73 to
0.65, because the excess-return numerator now subtracts a non-trivial
rf. The same deflation hits buy-and-hold (0.88 → 0.81). The honest
picture is that strategies and benchmarks alike look better under
rf = 0 than they do against a real cash rate, and the rf = 0 default in
casual backtesting is a systematic source of inflated Sharpe numbers.

## 04 — SPY/GLD rotation

Not built. The original plan was a 2-asset momentum rotation, but the
findings from demos 02 and 03 (directional signals on a single risk
asset) already cover the same ideas more cleanly, and the engine is
ready for a proper pairs / cointegration strategy next. A rotation
demo would add implementation time without teaching anything new.

## 05 — RSI(14) mean-reversion on SPY

`scripts/05_rsi_mean_reversion.py`. Long at RSI < 30, short at RSI > 70,
flat in between. Figure: `results/05_rsi_mean_reversion.png`.

|                        |  CAGR | Sharpe | Max DD | Turnover |
|------------------------|------:|-------:|-------:|---------:|
| RSI(14) mean-rev (net) | 3.39% |   0.37 |  −9.3% |     17.7 |
| MA(20/100) trend (net) | 8.74% |   0.65 | −22.0% |      2.4 |
| Buy & hold SPY         | 14.5% |   0.81 | −33.7% |      0.0 |

Signal distribution over the sample: 1.4% of days long, 89.2% flat,
9.4% short.

Honest interpretation: a clean negative finding. The strategy
underperforms B&H by ~11 percentage points of CAGR, gets a worse
Sharpe than either other strategy, and turns over 7× as much as the
MA crossover. The mechanism is visible in the distribution — the
strategy spends most of the window flat (earning at best rf) and
enters shorts on overbought readings that persist through sustained
uptrends. The shallow max drawdown (−9%) is not evidence of skill; it
is a consequence of rarely being invested. I am reporting this as-is
rather than tuning thresholds until the number looks better, because
doing otherwise is how in-sample overfitting produces plausible-looking
but spurious strategies.
