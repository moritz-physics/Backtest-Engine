"""Tests for performance metrics."""

import numpy as np
import pandas as pd
import pytest

from backtester.metrics.performance import (
    annualized_return,
    calmar_ratio,
    cagr,
    drawdown_series,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)


def _make_series(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


class TestSharpeZeroVariance:
    def test_constant_positive_returns_inf(self):
        r = _make_series([0.01] * 100)
        result = sharpe_ratio(r)
        assert result == np.inf

    def test_constant_negative_returns_neg_inf(self):
        r = _make_series([-0.01] * 100)
        result = sharpe_ratio(r)
        assert result == -np.inf

    def test_constant_zero_returns_nan(self):
        r = _make_series([0.0] * 100)
        result = sharpe_ratio(r)
        assert np.isnan(result)


class TestSortinoZeroDownside:
    def test_all_positive_returns_inf(self):
        r = _make_series([0.01] * 100)
        result = sortino_ratio(r)
        assert result == np.inf


class TestMaxDrawdown:
    def test_monotonically_increasing_zero(self):
        # Constant positive return -> monotonically increasing equity
        r = _make_series([0.01] * 100)
        assert max_drawdown(r) == pytest.approx(0.0)

    def test_50pct_drawdown(self):
        # Equity: 1.0 -> 0.5 -> 1.0
        # return day 1: (0.5 - 1.0) / 1.0 = -0.5
        # return day 2: (1.0 - 0.5) / 0.5 = +1.0
        r = _make_series([-0.5, 1.0])
        assert max_drawdown(r) == pytest.approx(-0.5)

    def test_drawdown_series_shape(self):
        r = _make_series([-0.5, 1.0])
        dd = drawdown_series(r)
        assert len(dd) == 2
        # After day 1: wealth=0.5, peak=1.0 -> dd = -0.5
        assert dd.iloc[0] == pytest.approx(-0.5)
        # After day 2: wealth=1.0, peak=1.0 -> dd = 0.0
        assert dd.iloc[1] == pytest.approx(0.0)


class TestAnnualizedReturn:
    def test_one_percent_daily(self):
        r = _make_series([0.01] * 252)
        result = annualized_return(r, periods_per_year=252)
        expected = (1.01) ** 252 - 1
        assert result == pytest.approx(expected, rel=1e-10)


class TestCAGR:
    def test_constant_return_matches_annualized(self):
        """For constant daily returns (zero vol), CAGR and
        annualized_return should give the same result."""
        r = _make_series([0.01] * 252)
        ann = annualized_return(r, periods_per_year=252)
        c = cagr(r, periods_per_year=252)
        expected = (1.01) ** 252 - 1
        assert ann == pytest.approx(expected, rel=1e-10)
        assert c == pytest.approx(expected, rel=1e-10)

    def test_volatile_returns_cagr_less_than_annualized(self):
        """With volatility, CAGR < annualized_return due to vol drag."""
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.02, 252)
        r = _make_series(list(rets))
        ann = annualized_return(r, periods_per_year=252)
        c = cagr(r, periods_per_year=252)
        # Volatility drag makes geometric < arithmetic
        assert c < ann


class TestTurnover:
    def test_constant_position_zero_turnover(self):
        """No position changes after initial entry -> only initial entry
        contributes to mean, so turnover = 1 * 252 / n."""
        n = 252
        idx = pd.bdate_range("2020-01-01", periods=n, freq="B")
        positions = pd.DataFrame({"A": 1.0}, index=idx)
        # Only day 0 has turnover (0 -> 1), rest are 0.
        # mean daily turnover = 1/252, annualized = 1.0
        result = turnover(positions, periods_per_year=252)
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_truly_flat_position(self):
        """Position is always zero -> zero turnover."""
        n = 100
        idx = pd.bdate_range("2020-01-01", periods=n, freq="B")
        positions = pd.DataFrame({"A": 0.0}, index=idx)
        assert turnover(positions) == pytest.approx(0.0)


class TestHitRate:
    def test_alternating(self):
        r = _make_series([0.01, -0.005, 0.01, -0.005])
        assert hit_rate(r) == pytest.approx(0.5)

    def test_all_positive(self):
        r = _make_series([0.01, 0.02, 0.03])
        assert hit_rate(r) == pytest.approx(1.0)

    def test_all_negative(self):
        r = _make_series([-0.01, -0.02, -0.03])
        assert hit_rate(r) == pytest.approx(0.0)


class TestSharpeWithSeriesRf:
    def test_series_rf_matches_scalar_when_constant(self):
        """A constant Series rf should give the same Sharpe as the
        equivalent scalar (annualized) rf."""
        rng = np.random.default_rng(99)
        r = _make_series(list(rng.normal(0.001, 0.01, 100)))

        scalar_rf = 0.05  # annualized
        daily_rf = 0.05 / 252
        series_rf = pd.Series(daily_rf, index=r.index)

        sharpe_scalar = sharpe_ratio(r, rf=scalar_rf)
        sharpe_series = sharpe_ratio(r, rf=series_rf)

        assert sharpe_scalar == pytest.approx(sharpe_series, rel=1e-10)


class TestCalmarRatio:
    def test_no_drawdown_returns_inf(self):
        r = _make_series([0.01] * 100)
        assert calmar_ratio(r) == np.inf
