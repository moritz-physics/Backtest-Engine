# Data loader and risk-free rates

The `backtester.data` package provides two loaders: equity OHLCV from Yahoo
Finance, and risk-free rates from FRED. Both are thin, single-purpose
wrappers designed so that the returned objects can be handed directly to
the backtest engine without further alignment work.

## API summary

From `backtester.data.loader`:

- **`load_ohlcv(tickers, start, end, *, cache=True, alignment="inner",
  drop_warn_threshold=0.05) -> pd.DataFrame`** — full OHLCV with
  MultiIndex columns `(field, ticker)`. Prices are adjusted close
  (`auto_adjust=True`), which folds splits and dividends into the price
  series and implicitly assumes dividend reinvestment at the close.
- **`load_prices(...) -> pd.DataFrame`** — thin wrapper that returns only
  the `Close` column, one price series per ticker.
- **`to_returns(prices, method="log" | "simple") -> pd.DataFrame`** —
  converts a price panel to returns. Row 0 is NaN; any mid-series NaN
  raises.

From `backtester.data.rates`:

- **`load_risk_free_rate(start, end, series="DTB3" | "DFF", *,
  cache=True) -> pd.Series`** — fetches an annualized-rate series from
  FRED's public CSV endpoint, converts percent to decimal, and
  forward-fills weekends and holidays.
- **`to_daily_rate(annual_rate, periods_per_year=252,
  method="simple" | "compounded") -> pd.Series`** — converts the
  annualized series to a per-period rate usable as the engine's
  `cash_rate` argument.

## Caching

Every loader writes a parquet file to `data/cache/` keyed on
`(ticker, start, end)` or `(series, start, end)`. The cache is a plain
filesystem cache with no invalidation; date-range mismatches miss and
trigger a fresh download. This is intentional — the cache is meant to
speed up repeated runs over a fixed backtest window, not to serve as a
long-lived historical store.

## NaN and alignment policy

Equity data can legitimately be missing for a ticker on a given trading
day (halts, delistings, cross-listed instruments that follow a different
calendar). The loader exposes two alignment modes:

- **`alignment="inner"` (default).** Multi-ticker loads are aligned via
  inner join on the DatetimeIndex: only dates present in *every* ticker
  are kept. No prices are fabricated. If the inner join drops more than
  `drop_warn_threshold` of the longest series, a warning is logged that
  identifies the ticker with the most missing data and the dropped date
  range. The default threshold is 5%.
- **`alignment="outer"`.** All dates from every ticker are kept and NaN
  is preserved where data is missing. The caller is then responsible
  for handling NaN downstream.

`to_returns` is strict: NaN is permitted only in row 0 of the output
(the first return is undefined); any mid-series NaN raises a
`ValueError` with a message pointing at the gap. This stops silent
corruption from propagating into the engine.

## Data-source caveats

These caveats matter and should shape how you interpret any result from
the framework.

- **Survivorship bias in Yahoo Finance.** The ticker universe on Yahoo
  is the surviving one. Delisted names and merged shares often do not
  appear, or appear with truncated history. Any result over a long
  window on a universe picked from today's tickers is upward-biased by
  some unknown amount. This framework makes no attempt to correct for
  it; backtests are run on index ETFs (SPY, QQQ, IWM) where the
  survivorship effect is small but not zero.
- **DTB3 as a proxy for the cash rate.** The 3-month T-bill secondary
  market rate is the simplest defensible proxy for the return on
  uninvested capital. It is not the overnight rate, and it ignores the
  actual funding cost a retail account would face (broker sweep rates
  are usually lower). `DFF` (Federal Funds Effective Rate) is offered
  as an alternative and is a tighter match for overnight cash.
- **Forward-fill on rates.** Weekends and FRED holidays are
  forward-filled, because an administered or quoted rate that was in
  effect on Friday is the best estimate for Saturday and Sunday. This
  is *not* the same reasoning that would apply to equity prices, which
  are never forward-filled here — a missing equity print is a gap to
  be surfaced, while a missing rate print is a calendar artefact.
