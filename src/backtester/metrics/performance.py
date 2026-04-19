"""Performance metrics for daily-return series.

All functions accept a **daily simple returns** ``pd.Series`` (not log
returns) unless noted otherwise.  Annualization assumes 252 trading days
per year by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(
    r: pd.Series, periods_per_year: int = 252
) -> float:
    """Annualized return via geometric compounding of the arithmetic mean.

    Formula::

        (1 + mean(r))^N - 1

    where *N* = ``periods_per_year``.

    This answers: "If the strategy earned its average daily return every
    day for a year, what would the annual return be?"  It is **not** the
    realised CAGR — see :func:`cagr` for that.  The two coincide when
    daily returns are constant (zero volatility) and diverge as
    volatility increases (volatility drag).

    Parameters
    ----------
    r : pd.Series
        Daily simple returns.
    periods_per_year : int, default 252
        Trading days per year.

    Returns
    -------
    float
    """
    mean_r = r.mean()
    return (1.0 + mean_r) ** periods_per_year - 1.0


def cagr(r: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate (realised geometric return).

    Formula::

        prod(1 + r)^(N / T) - 1

    where *T* = ``len(r)`` and *N* = ``periods_per_year``.

    This is the true annualized return the strategy delivered over the
    sample period, accounting for compounding and volatility drag.
    Compare with :func:`annualized_return`, which compounds the
    arithmetic mean and will be higher for volatile strategies.

    Parameters
    ----------
    r : pd.Series
        Daily simple returns.
    periods_per_year : int, default 252
        Trading days per year.

    Returns
    -------
    float
    """
    total_return = (1.0 + r).prod()
    n_periods = len(r)
    return total_return ** (periods_per_year / n_periods) - 1.0


def annualized_volatility(
    r: pd.Series, periods_per_year: int = 252
) -> float:
    """Annualized volatility (sample standard deviation, scaled).

    Formula::

        std(r, ddof=1) * sqrt(N)

    Uses sample standard deviation (ddof=1), which is the pandas
    default and the standard convention in practice.

    Parameters
    ----------
    r : pd.Series
        Daily simple returns.
    periods_per_year : int, default 252
        Trading days per year.

    Returns
    -------
    float
    """
    return r.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(
    r: pd.Series,
    rf: float | pd.Series = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio.

    Formula::

        mean(r - rf_daily) / std(r - rf_daily, ddof=1) * sqrt(N)

    The risk-free rate *rf* can be either:

    - A **scalar** (annualized rate, e.g. 0.05 for 5%), which is
      converted to a constant daily rate via ``rf / N``.
    - A **pd.Series** of daily rates aligned to *r*'s index, used
      directly as the daily risk-free rate (no further conversion).

    Parameters
    ----------
    r : pd.Series
        Daily simple returns.
    rf : float or pd.Series, default 0.0
        Risk-free rate.  Scalar = annualized; Series = daily rates.
    periods_per_year : int, default 252
        Trading days per year.

    Returns
    -------
    float
        ``np.inf`` if mean excess return > 0 and std == 0,
        ``-np.inf`` if mean excess return < 0 and std == 0,
        ``np.nan`` if both are zero.
    """
    if isinstance(rf, pd.Series):
        rf_daily = rf.reindex(r.index)
    else:
        rf_daily = rf / periods_per_year
    excess = r - rf_daily
    mu = excess.mean()
    sigma = excess.std(ddof=1)

    if np.isclose(sigma, 0.0, atol=1e-14):
        if mu > 0.0:
            return np.inf
        if mu < 0.0:
            return -np.inf
        return np.nan

    return (mu / sigma) * np.sqrt(periods_per_year)


def sortino_ratio(
    r: pd.Series,
    rf: float | pd.Series = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio.

    Like the Sharpe ratio but penalizes only downside volatility.

    Formula::

        mean(r - rf_daily) / downside_deviation * sqrt(N)

    Downside deviation::

        sqrt(mean(min(r - rf_daily, 0)^2))

    This uses all observations: days with non-negative excess return
    contribute zero to the sum (not excluded).  This is the more
    common convention (Sortino & van der Meer, 1991).

    The risk-free rate *rf* can be either a scalar (annualized) or a
    pd.Series of daily rates — same convention as :func:`sharpe_ratio`.

    Parameters
    ----------
    r : pd.Series
        Daily simple returns.
    rf : float or pd.Series, default 0.0
        Risk-free rate.  Scalar = annualized; Series = daily rates.
    periods_per_year : int, default 252
        Trading days per year.

    Returns
    -------
    float
        ``np.inf`` / ``-np.inf`` / ``np.nan`` when downside deviation
        is zero (same convention as :func:`sharpe_ratio`).
    """
    if isinstance(rf, pd.Series):
        rf_daily = rf.reindex(r.index)
    else:
        rf_daily = rf / periods_per_year
    excess = r - rf_daily
    mu = excess.mean()
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt((downside**2).mean())

    if np.isclose(downside_dev, 0.0, atol=1e-14):
        if mu > 0.0:
            return np.inf
        if mu < 0.0:
            return -np.inf
        return np.nan

    return (mu / downside_dev) * np.sqrt(periods_per_year)


def max_drawdown(r: pd.Series) -> float:
    """Maximum drawdown (peak-to-trough decline).

    Computed from the cumulative wealth index
    ``W = cumprod(1 + r)``.

    Returns
    -------
    float
        A non-positive number.  Zero if the equity curve is
        monotonically increasing.
    """
    wealth = (1.0 + r).cumprod()
    # The peak must start at 1.0 (pre-trade equity) so that a loss
    # on the very first day registers as a drawdown.
    running_max = wealth.cummax().clip(lower=1.0)
    dd = (wealth - running_max) / running_max
    return dd.min()


def drawdown_series(r: pd.Series) -> pd.Series:
    """Full drawdown time series for plotting.

    Parameters
    ----------
    r : pd.Series
        Daily simple returns.

    Returns
    -------
    pd.Series
        Each value is the fractional drawdown from the running peak
        at that point in time (non-positive).
    """
    wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax().clip(lower=1.0)
    return (wealth - running_max) / running_max


def calmar_ratio(
    r: pd.Series, periods_per_year: int = 252
) -> float:
    """Calmar ratio: annualized return divided by maximum drawdown.

    Formula::

        annualized_return(r) / abs(max_drawdown(r))

    Parameters
    ----------
    r : pd.Series
        Daily simple returns.
    periods_per_year : int, default 252
        Trading days per year.

    Returns
    -------
    float
        ``np.inf`` if max drawdown is zero (monotonically increasing
        equity).
    """
    mdd = max_drawdown(r)
    if mdd == 0.0:
        return np.inf
    ann_ret = annualized_return(r, periods_per_year)
    return ann_ret / abs(mdd)


def turnover(
    positions: pd.DataFrame, periods_per_year: int = 252
) -> float:
    """Annualized portfolio turnover (one-way).

    Formula::

        mean(|position_t - position_{t-1}|) * N

    The first day assumes prior position of zero.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily position weights, one column per asset.
    periods_per_year : int, default 252
        Trading days per year.

    Returns
    -------
    float
        Annualized one-way turnover.
    """
    prev = positions.shift(1).fillna(0.0)
    daily_turnover = (positions - prev).abs().sum(axis=1)
    return daily_turnover.mean() * periods_per_year


def hit_rate(r: pd.Series) -> float:
    """Fraction of days with positive returns.

    Formula::

        (r > 0).mean()

    Parameters
    ----------
    r : pd.Series
        Daily simple returns.

    Returns
    -------
    float
        Value in [0, 1].
    """
    return (r > 0).mean()
