"""Technical indicators for signal generation.

All indicators are computed from historical data only (no look-ahead).
Each function documents its warmup period — the number of initial rows
that will be NaN due to insufficient data.
"""

from __future__ import annotations

import pandas as pd


def rsi(
    prices: pd.DataFrame | pd.Series,
    period: int = 14,
) -> pd.DataFrame | pd.Series:
    """Relative Strength Index (Wilder, 1978).

    Measures the ratio of average gains to average losses over a lookback
    window, scaled to 0–100.  Classical interpretation: values below 30
    indicate oversold conditions (potential mean-reversion upward), values
    above 70 indicate overbought conditions (potential mean-reversion
    downward).

    Uses **Wilder's smoothing** (exponentially weighted moving average with
    ``alpha = 1 / period``), which is the original and most widely used
    convention.  Some implementations use a simple rolling mean instead —
    this function does not.

    Parameters
    ----------
    prices : pd.DataFrame or pd.Series
        Price series (e.g. adjusted close).  For a DataFrame, RSI is
        computed independently for each column.
    period : int, default 14
        Lookback window for the exponential average of gains and losses.

    Returns
    -------
    pd.DataFrame or pd.Series
        RSI values in [0, 100].  The first ``period`` rows are NaN
        (insufficient data for the warmup).  When both average gain and
        average loss are zero (e.g. constant prices), the result is NaN.
        When average loss is zero but average gain is positive (monotonically
        increasing prices), RSI is 100.
    """
    delta = prices.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    result = 100.0 - 100.0 / (1.0 + rs)

    return result
