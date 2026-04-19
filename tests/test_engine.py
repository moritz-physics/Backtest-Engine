"""Tests for the backtest engine."""

import numpy as np
import pandas as pd
import pytest

from backtester.backtest.engine import run_backtest
from backtester.costs.linear import LinearCost


def _make_index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n, freq="B")


class TestFlatPosition:
    def test_zero_signal_gives_zero_returns(self):
        n = 10
        idx = _make_index(n)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame({"A": rng.normal(0.001, 0.01, n)}, index=idx)
        signal = pd.DataFrame({"A": 0.0}, index=idx)

        result = run_backtest(signal, returns, signal_lag=1)

        np.testing.assert_array_equal(result.gross_returns.values, 0.0)
        np.testing.assert_array_equal(result.net_returns.values, 0.0)


class TestConstantPosition:
    def test_buy_and_hold_matches(self):
        n = 6
        idx = _make_index(n)
        daily_ret = 0.01
        returns = pd.DataFrame({"A": daily_ret}, index=idx)
        signal = pd.DataFrame({"A": 1.0}, index=idx)

        result = run_backtest(signal, returns, signal_lag=1)

        # After lag=1 shift: positions = [0, 1, 1, 1, 1, 1]
        expected_pos = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        np.testing.assert_array_almost_equal(
            result.positions["A"].values, expected_pos
        )

        # Gross returns: position * return
        expected_gross = [0.0, daily_ret, daily_ret, daily_ret,
                          daily_ret, daily_ret]
        np.testing.assert_array_almost_equal(
            result.gross_returns["A"].values, expected_gross
        )


class TestLookAheadBias:
    def test_lagged_signal_cannot_peek_at_future(self):
        """A perfect-foresight signal with lag=1 must NOT achieve the
        cheating PnL (signal_t * return_t with no shift)."""
        n = 100
        idx = _make_index(n)
        rng = np.random.default_rng(123)
        rets = rng.normal(0.0005, 0.01, n)
        returns = pd.DataFrame({"A": rets}, index=idx)

        # Perfect foresight: signal_t = sign(return_{t+1})
        # This signal "knows" tomorrow's return.
        future_sign = np.sign(np.roll(rets, -1))
        future_sign[-1] = 0.0  # last day has no future
        signal = pd.DataFrame({"A": future_sign}, index=idx)

        result = run_backtest(signal, returns, signal_lag=1)

        # With lag=1, position_t = signal_{t-1} = sign(return_t).
        # So gross_return_t = sign(return_t) * return_t = |return_t|
        # for all t >= 1.
        # This is the "lagged" PnL — NOT the cheating PnL.
        lagged_pnl = result.portfolio_gross_returns.sum()

        # The cheating PnL would be: signal_t * return_t (no shift).
        # signal_t = sign(return_{t+1}), so cheating = sign(r_{t+1}) * r_t
        cheating_pnl = (future_sign * rets).sum()

        # They must differ (the engine correctly applies the lag).
        assert not np.isclose(lagged_pnl, cheating_pnl, atol=1e-6), (
            "Lagged PnL equals cheating PnL — look-ahead bias detected!"
        )

        # Sanity: lagged PnL should be positive (we get |r_t| for t>=1)
        assert lagged_pnl > 0


class TestCostFlip:
    def test_flip_cost(self):
        """Flipping from +1 to -1 at 5 bps costs 2 * 5/10000 = 0.001."""
        n = 5
        idx = _make_index(n)
        # Signal: +1 for days 0-1, then -1 for days 2-4.
        # With lag=1: positions = [0, +1, +1, -1, -1]
        sig = [1.0, 1.0, -1.0, -1.0, -1.0]
        signal = pd.DataFrame({"A": sig}, index=idx)
        returns = pd.DataFrame({"A": 0.01}, index=idx)

        cost_model = LinearCost(bps_per_trade=5.0)
        result = run_backtest(signal, returns, cost_model=cost_model,
                              signal_lag=1)

        costs = result.cost_series["A"].values
        # Day 0: |0 - 0| = 0
        assert costs[0] == pytest.approx(0.0)
        # Day 1: |1 - 0| * 5/10000 = 0.0005
        assert costs[1] == pytest.approx(0.0005)
        # Day 2: |1 - 1| = 0
        assert costs[2] == pytest.approx(0.0)
        # Day 3: |-1 - 1| * 5/10000 = 0.001
        assert costs[3] == pytest.approx(0.001)
        # Day 4: |-1 - (-1)| = 0
        assert costs[4] == pytest.approx(0.0)


class TestCostOnInitialEntry:
    def test_entry_from_flat_incurs_cost(self):
        """Entering from flat (0 -> 1) must incur cost = 0.0005 at 5 bps."""
        n = 5
        idx = _make_index(n)
        signal = pd.DataFrame({"A": 1.0}, index=idx)
        returns = pd.DataFrame({"A": 0.01}, index=idx)

        cost_model = LinearCost(bps_per_trade=5.0)
        result = run_backtest(signal, returns, cost_model=cost_model,
                              signal_lag=1)

        # Positions after shift: [0, 1, 1, 1, 1]
        costs = result.cost_series["A"].values
        # Day 0: flat -> flat = 0
        assert costs[0] == pytest.approx(0.0)
        # Day 1: flat -> 1 = 0.0005
        assert costs[1] == pytest.approx(0.0005)
        # Days 2-4: no change = 0
        for i in range(2, n):
            assert costs[i] == pytest.approx(0.0)


class TestMismatchedIndex:
    def test_different_dates_raises(self):
        idx1 = _make_index(5)
        idx2 = _make_index(6)
        signal = pd.DataFrame({"A": 1.0}, index=idx1)
        returns = pd.DataFrame({"A": 0.01}, index=idx2)

        with pytest.raises(ValueError, match="identical DatetimeIndex"):
            run_backtest(signal, returns)

    def test_different_columns_raises(self):
        idx = _make_index(5)
        signal = pd.DataFrame({"A": 1.0}, index=idx)
        returns = pd.DataFrame({"B": 0.01}, index=idx)

        with pytest.raises(ValueError, match="identical columns"):
            run_backtest(signal, returns)


class TestCashRateFlatPosition:
    def test_flat_earns_cash(self):
        """When position=0, strategy return = cash_rate each day."""
        n = 10
        idx = _make_index(n)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame({"A": rng.normal(0.001, 0.01, n)}, index=idx)
        signal = pd.DataFrame({"A": 0.0}, index=idx)

        # 5% annual, simple daily = 0.05/252
        daily_rf = 0.05 / 252
        cash_rate = pd.Series(daily_rf, index=idx)

        result = run_backtest(signal, returns, signal_lag=1,
                              cash_rate=cash_rate)

        # position=0 everywhere after shift, so:
        # gross = 0 * return + (1 - 0) * cash_rate = cash_rate
        expected = np.full(n, daily_rf)
        np.testing.assert_array_almost_equal(
            result.portfolio_gross_returns.values, expected
        )


class TestCashRateFullPosition:
    def test_fully_invested_no_cash(self):
        """When position=1, strategy return = asset return (no cash)."""
        n = 6
        idx = _make_index(n)
        daily_ret = 0.01
        returns = pd.DataFrame({"A": daily_ret}, index=idx)
        signal = pd.DataFrame({"A": 1.0}, index=idx)

        daily_rf = 0.05 / 252
        cash_rate = pd.Series(daily_rf, index=idx)

        result = run_backtest(signal, returns, signal_lag=1,
                              cash_rate=cash_rate)

        # After shift: positions = [0, 1, 1, 1, 1, 1]
        # Day 0: pos=0 -> gross = 0*ret + 1*cash = cash
        # Days 1-5: pos=1 -> gross = 1*ret + 0*cash = ret
        assert result.portfolio_gross_returns.iloc[0] == pytest.approx(
            daily_rf
        )
        for i in range(1, n):
            assert result.portfolio_gross_returns.iloc[i] == pytest.approx(
                daily_ret
            )


class TestCashRatePartialPosition:
    def test_half_invested(self):
        """position=0.5: return = 0.5*asset + 0.5*cash."""
        n = 6
        idx = _make_index(n)
        daily_ret = 0.01
        returns = pd.DataFrame({"A": daily_ret}, index=idx)
        signal = pd.DataFrame({"A": 0.5}, index=idx)

        daily_rf = 0.05 / 252
        cash_rate = pd.Series(daily_rf, index=idx)

        result = run_backtest(signal, returns, signal_lag=1,
                              cash_rate=cash_rate)

        # After shift: positions = [0, 0.5, 0.5, 0.5, 0.5, 0.5]
        # Day 0: pos=0 -> 1.0 * cash
        assert result.portfolio_gross_returns.iloc[0] == pytest.approx(
            daily_rf
        )
        # Days 1-5: 0.5*ret + 0.5*cash
        expected = 0.5 * daily_ret + 0.5 * daily_rf
        for i in range(1, n):
            assert result.portfolio_gross_returns.iloc[i] == pytest.approx(
                expected
            )


class TestCashRateIndexMismatch:
    def test_missing_dates_raises(self):
        n = 10
        idx = _make_index(n)
        returns = pd.DataFrame({"A": 0.01}, index=idx)
        signal = pd.DataFrame({"A": 1.0}, index=idx)

        # Cash rate covers only first 5 days
        cash_rate = pd.Series(0.0001, index=idx[:5])

        with pytest.raises(ValueError, match="cash_rate index must contain"):
            run_backtest(signal, returns, signal_lag=1, cash_rate=cash_rate)


class TestSignalLagValidation:
    def test_lag_zero_raises(self):
        idx = _make_index(5)
        signal = pd.DataFrame({"A": 1.0}, index=idx)
        returns = pd.DataFrame({"A": 0.01}, index=idx)

        with pytest.raises(ValueError, match="signal_lag must be >= 1"):
            run_backtest(signal, returns, signal_lag=0)

    def test_negative_lag_raises(self):
        idx = _make_index(5)
        signal = pd.DataFrame({"A": 1.0}, index=idx)
        returns = pd.DataFrame({"A": 0.01}, index=idx)

        with pytest.raises(ValueError, match="signal_lag must be >= 1"):
            run_backtest(signal, returns, signal_lag=-1)
