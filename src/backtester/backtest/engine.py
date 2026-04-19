"""Vectorized backtest engine.

Takes a signal DataFrame (target positions) and an asset-returns DataFrame,
applies a configurable look-ahead lag, computes strategy returns gross and
net of transaction costs, and returns everything in a frozen dataclass.

Look-ahead bias prevention
--------------------------
The position used on day *t* comes from the signal at day *t - signal_lag*.
``signal_lag`` must be >= 1 (enforced).  This means:

* Signal computed from data up to and including day *t* is first acted
  upon on day *t + signal_lag*.
* The first ``signal_lag`` days of the backtest are flat (zero position).

Returns convention
------------------
The ``returns`` input **must be simple (arithmetic) returns**, not log
returns.  The strategy return on day *t* is::

    gross_t = position_t * return_t + (1 - |position_t|) * cash_rate_t
    portfolio_gross_t = sum across assets

When ``cash_rate`` is None, the second term is zero and the formula
simplifies to ``position_t * return_t``.

This is only correct for simple returns; log returns are not additive
across assets weighted by position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtester.costs.linear import LinearCost


@dataclass(frozen=True)
class BacktestResult:
    """Container for all outputs of :func:`run_backtest`.

    Attributes
    ----------
    positions : pd.DataFrame
        Post-shift positions (what the strategy actually holds each day).
    gross_returns : pd.DataFrame
        Per-asset gross strategy returns (position * asset return).
    net_returns : pd.DataFrame
        Per-asset strategy returns after subtracting transaction costs.
    cost_series : pd.DataFrame
        Daily transaction costs per asset.
    turnover_series : pd.DataFrame
        Daily absolute position change per asset.
    portfolio_gross_returns : pd.Series
        Row-sum of *gross_returns* across assets.
    portfolio_net_returns : pd.Series
        Row-sum of *net_returns* across assets.
    summary : dict[str, float]
        Initially empty; the caller may populate with metrics.
    """

    positions: pd.DataFrame
    gross_returns: pd.DataFrame
    net_returns: pd.DataFrame
    cost_series: pd.DataFrame
    turnover_series: pd.DataFrame
    portfolio_gross_returns: pd.Series
    portfolio_net_returns: pd.Series
    summary: dict[str, float] = field(default_factory=dict)


def run_backtest(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    cost_model: LinearCost | None = None,
    signal_lag: int = 1,
    cash_rate: pd.Series | None = None,
) -> BacktestResult:
    """Run a vectorized backtest.

    Parameters
    ----------
    signal : pd.DataFrame
        Target positions in [-1, 1] (or arbitrary weights).  Index is
        dates, columns are asset tickers.  Must share the exact same
        index and columns as *returns*.
    returns : pd.DataFrame
        **Simple** (arithmetic) daily returns for each asset.
    cost_model : LinearCost or None, default None
        Transaction cost model.  If None, gross and net returns are
        identical (zero cost).
    signal_lag : int, default 1
        Number of days to shift the signal forward.  Must be >= 1 to
        prevent look-ahead bias.  A lag of 1 means today's position
        was determined by yesterday's signal.
    cash_rate : pd.Series or None, default None
        Daily risk-free rate (already converted from annualized to
        daily via :func:`~backtester.data.rates.to_daily_rate`).
        When provided, uninvested capital earns this rate::

            effective_return = position * asset_return
                             + (1 - |position|) * cash_rate

        For fully invested positions (|position| = 1), cash
        contribution is zero.  For leveraged positions (|position| > 1),
        the formula naturally yields zero or negative cash contribution
        (borrowing cost).  The index of *cash_rate* must contain every
        date in *returns*.index.

    Returns
    -------
    BacktestResult

    Raises
    ------
    ValueError
        If ``signal_lag < 1``, indices/columns don't match, cash_rate
        index doesn't cover returns index, or NaN appears in positions
        after the shift-and-fill.
    """
    # --- Validate signal_lag ---
    if signal_lag < 1:
        raise ValueError(
            f"signal_lag must be >= 1 to prevent look-ahead bias, "
            f"got {signal_lag}"
        )

    # --- Validate cash_rate alignment ---
    if cash_rate is not None:
        missing = returns.index.difference(cash_rate.index)
        if len(missing) > 0:
            raise ValueError(
                "cash_rate index must contain every date in returns.index. "
                f"Missing {len(missing)} dates, first: {missing[0]}."
            )

    # --- Validate index and column alignment ---
    if not signal.index.equals(returns.index):
        raise ValueError(
            "signal and returns must have identical DatetimeIndex. "
            f"signal has {len(signal.index)} dates, "
            f"returns has {len(returns.index)} dates. "
            "Align them before calling run_backtest."
        )
    if not signal.columns.equals(returns.columns):
        raise ValueError(
            "signal and returns must have identical columns. "
            f"signal columns: {list(signal.columns)}, "
            f"returns columns: {list(returns.columns)}."
        )

    # --- Shift signal to form positions ---
    # Position on day t comes from signal at day t - signal_lag.
    # The first signal_lag days are filled with 0 (flat).
    positions = signal.shift(signal_lag).fillna(0.0)

    # NaN audit: no NaN should remain after fillna
    if positions.isna().any().any():
        raise ValueError(
            "NaN detected in positions after shift and fillna(0). "
            "This indicates NaN in the input signal beyond the "
            f"first {signal_lag} rows."
        )

    # --- Gross strategy returns ---
    # For each asset: position * asset_return + (1 - |position|) * cash_rate
    # When cash_rate is None, the cash term is zero.
    gross_returns = positions * returns
    if cash_rate is not None:
        cash_aligned = cash_rate.reindex(returns.index)
        unused_capital = 1.0 - positions.abs()
        # Broadcast the daily cash rate across all asset columns
        cash_contribution = unused_capital.multiply(cash_aligned, axis=0)
        gross_returns = gross_returns + cash_contribution

    # --- Transaction costs ---
    prev_positions = positions.shift(1).fillna(0.0)
    turnover_series = (positions - prev_positions).abs()

    if cost_model is not None:
        cost_series = cost_model.cost(positions)
    else:
        cost_series = pd.DataFrame(0.0, index=positions.index,
                                   columns=positions.columns)

    # --- Net returns ---
    net_returns = gross_returns - cost_series

    # --- Portfolio-level (sum across assets) ---
    portfolio_gross = gross_returns.sum(axis=1)
    portfolio_gross.name = "portfolio_gross"
    portfolio_net = net_returns.sum(axis=1)
    portfolio_net.name = "portfolio_net"

    return BacktestResult(
        positions=positions,
        gross_returns=gross_returns,
        net_returns=net_returns,
        cost_series=cost_series,
        turnover_series=turnover_series,
        portfolio_gross_returns=portfolio_gross,
        portfolio_net_returns=portfolio_net,
    )
