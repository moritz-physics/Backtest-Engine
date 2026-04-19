# CLAUDE.md — Project Conventions

This file gives Claude Code context about this repository. 
Keep it updated as conventions evolve.

## Project

A vectorized backtesting framework for systematic trading strategies
in Python. Built for learning quantitative finance best practices.
Owner: a physics/AI master's student, not a professional quant — 
prioritize clarity, correctness, and teaching value over cleverness.

## Core Domain Constraints (NON-NEGOTIABLE)

These are the rules that define whether the framework is correct. 
Violating them silently corrupts all downstream results.

1. **No look-ahead bias.** Every feature, signal, or decision at time t 
   must be computable using only data available at or before t. 
   In practice this means: after computing a signal from data up to t, 
   always shift the signal by 1 before multiplying by returns. Review 
   every new feature for this.

2. **Point-in-time data.** Never use today's universe membership, 
   today's fundamentals, or today's adjusted prices as if they were 
   known historically, unless the adjustment is explicitly accounted for.

3. **Costs are not optional.** Every backtest reports gross AND net 
   (after cost) performance. Default assumption: 5 bps per trade.

4. **Reproducibility.** All randomness must be seeded. All runs must 
   be reproducible from a single script entry point.

## Stack

- Python 3.11+
- Package management: uv (preferred) or conda
- Core: numpy, pandas, scipy
- Data: yfinance (primary), pandas_datareader, pandas_market_calendars
- Plotting: matplotlib (no seaborn unless necessary)
- Testing: pytest
- Linting: ruff
- Type checking: (optional) mypy, lightly

Avoid: TA-Lib (adds a C dependency), proprietary data APIs, 
libraries that obscure logic (e.g., heavy backtesting frameworks —
we are building our own on purpose).

## Code Style

- Vectorized pandas/numpy first. Loops only when vectorization is 
  genuinely impossible, and noted in a comment explaining why.
- Type hints on all public functions.
- Docstrings in NumPy style on public functions, explaining especially 
  any financial/domain assumptions.
- Small, composable functions. Prefer pure functions over stateful 
  objects where possible.
- No silent NaN handling. Decide explicitly: drop, fillna with 
  documented rule, or raise.

## Repo Layout
src/backtester/      # importable package — the real logic
data/              # loaders, calendars, universe handling
features/          # signal/feature engineering
backtest/          # core engine, position-to-PnL logic
metrics/           # Sharpe, drawdown, turnover, etc.
costs/             # transaction cost models
splits/            # walk-forward and other time-series splitters
tests/               # pytest; meaningful coverage of PnL logic especially
notebooks/           # exploration only — do not put source-of-truth logic here
scripts/             # runnable entry points
results/             # regenerable figures/tables, .gitignored except READMEs
data/                # .gitignored; raw cache populated by scripts/fetch_data.py