"""Tests for backtester.features.indicators."""

import numpy as np
import pandas as pd
import pytest

from backtester.features.indicators import rsi


def _make_index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n, freq="B")


class TestRSIRange:
    def test_rsi_values_in_0_100(self):
        """After warmup, all RSI values must be in [0, 100]."""
        rng = np.random.default_rng(42)
        prices = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 200))),
            index=_make_index(200),
        )
        result = rsi(prices, period=14)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0


class TestRSIWarmup:
    def test_first_period_rows_nan_rest_finite(self):
        """First `period` rows are NaN, the rest are finite."""
        rng = np.random.default_rng(7)
        n = 50
        period = 14
        prices = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))),
            index=_make_index(n),
        )
        result = rsi(prices, period=period)
        # First row is NaN from diff(); combined with ewm min_periods,
        # the first `period` values should be NaN.
        assert result.iloc[:period].isna().all()
        assert result.iloc[period:].notna().all()


class TestRSIKnownValue:
    def test_hand_computed_rsi(self):
        """Verify RSI against a hand-computed value on a short sequence.

        Prices (15 values): 44, 44.34, 44.09, 43.61, 44.33, 44.83,
        45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28

        These are the classic Wilder textbook example prices (adapted).
        We verify the final RSI value using manual calculation with
        Wilder smoothing (ewm alpha=1/14).
        """
        prices_list = [
            44.00, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10,
            45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
        ]
        prices = pd.Series(prices_list, index=_make_index(15))
        result = rsi(prices, period=14)

        # --- Hand computation (must match Wilder smoothing with clip) ---
        deltas = pd.Series(prices_list).diff()
        gains = deltas.clip(lower=0.0)
        losses = (-deltas).clip(lower=0.0)
        avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
        expected_rsi = 100.0 - 100.0 / (1.0 + rs)

        np.testing.assert_almost_equal(result.iloc[-1], expected_rsi, decimal=4)


class TestRSIConstantPrices:
    def test_constant_prices_give_nan(self):
        """Constant prices → avg_gain=0 and avg_loss=0 → RSI is NaN."""
        prices = pd.Series([100.0] * 30, index=_make_index(30))
        result = rsi(prices, period=14)
        # After warmup, all values should be NaN (0/0)
        assert result.iloc[14:].isna().all()

    def test_monotonic_increase_gives_100(self):
        """Monotonically increasing prices → avg_loss=0 → RSI=100."""
        prices = pd.Series(
            np.arange(1.0, 31.0), index=_make_index(30),
        )
        result = rsi(prices, period=14)
        valid = result.dropna()
        np.testing.assert_array_almost_equal(valid.values, 100.0, decimal=10)


class TestRSINoLookahead:
    def test_truncating_future_does_not_change_past(self):
        """RSI at index t computed on full data must equal RSI computed
        on prices[:t+1].  This is the anti-look-ahead test.
        """
        rng = np.random.default_rng(99)
        n = 60
        period = 14
        prices = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))),
            index=_make_index(n),
        )
        full_rsi = rsi(prices, period=period)

        # Check at several points after warmup
        for t in [period, period + 5, period + 20, n - 1]:
            truncated_rsi = rsi(prices.iloc[: t + 1], period=period)
            np.testing.assert_almost_equal(
                full_rsi.iloc[t], truncated_rsi.iloc[-1], decimal=10,
                err_msg=f"Look-ahead detected at index {t}",
            )


class TestRSIDataFrame:
    def test_dataframe_columnwise(self):
        """DataFrame input → same shape, each column independent."""
        rng = np.random.default_rng(123)
        n = 50
        idx = _make_index(n)
        df = pd.DataFrame(
            {
                "A": 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))),
                "B": 50.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n))),
            },
            index=idx,
        )
        result = rsi(df, period=14)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape
        assert list(result.columns) == ["A", "B"]

        # Each column should match the Series-based computation
        for col in df.columns:
            expected = rsi(df[col], period=14)
            pd.testing.assert_series_equal(
                result[col], expected, check_names=False,
            )
