"""Transaction cost models for backtesting.

The base model is a simple linear (proportional) cost that charges a
fixed number of basis points on every unit of position change.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class LinearCost:
    """Proportional transaction cost model.

    Cost on each day equals the absolute change in position multiplied
    by a per-unit spread expressed in basis points::

        cost_t = |position_t - position_{t-1}| * bps_per_trade / 10_000

    The first day's cost assumes the portfolio starts flat (position = 0),
    so entering an initial position of 1.0 at 5 bps costs 0.0005.

    Parameters
    ----------
    bps_per_trade : float, default 5.0
        Half-spread cost in basis points charged on each unit of
        absolute position change.  5 bps is a reasonable default for
        liquid US large-cap equities.
    """

    bps_per_trade: float = 5.0              #transaction cost variable

    def cost(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Compute daily transaction costs from a position series.

        Parameters
        ----------
        positions : pd.DataFrame
            Position weights indexed by date, one column per asset.

        Returns
        -------
        pd.DataFrame
            Same shape as *positions*.  Each cell holds the cost
            incurred on that day for that asset.
        """
        # Previous position; row 0 assumes flat (0) prior position.
        prev = positions.shift(1).fillna(0.0)
        delta = (positions - prev).abs()
        return delta * self.bps_per_trade / 10_000
