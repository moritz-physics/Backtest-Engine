"""Tests for the risk-free rate loader."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtester.data.rates import load_risk_free_rate, to_daily_rate


def _make_fred_csv(start: str, end: str, *, gaps: bool = False) -> str:
    """Build a fake FRED CSV response (percent values)."""
    idx = pd.bdate_range(start, end, freq="B")
    lines = ["DATE,DTB3"]
    for i, date in enumerate(idx):
        if gaps and i in (2, 3):
            lines.append(f"{date.strftime('%Y-%m-%d')},.")
        else:
            lines.append(f"{date.strftime('%Y-%m-%d')},5.25")
    return "\n".join(lines)


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.raise_for_status = MagicMock()
    return resp


class TestLoadRiskFreeRate:
    @patch("backtester.data.rates.requests.get")
    def test_shape_and_type(self, mock_get):
        mock_get.return_value = _mock_response(
            _make_fred_csv("2023-01-01", "2023-01-31")
        )
        result = load_risk_free_rate("2023-01-01", "2023-01-31", cache=False)

        assert isinstance(result, pd.Series)
        assert result.name == "DTB3"
        assert len(result) > 0

    @patch("backtester.data.rates.requests.get")
    def test_percent_to_decimal_conversion(self, mock_get):
        mock_get.return_value = _mock_response(
            _make_fred_csv("2023-01-01", "2023-01-31")
        )
        result = load_risk_free_rate("2023-01-01", "2023-01-31", cache=False)

        # FRED returns 5.25 (percent); we expect 0.0525 (decimal)
        assert result.iloc[0] == pytest.approx(0.0525)

    @patch("backtester.data.rates.requests.get")
    def test_forward_fill_gaps(self, mock_get):
        mock_get.return_value = _mock_response(
            _make_fred_csv("2023-01-01", "2023-01-31", gaps=True)
        )
        result = load_risk_free_rate("2023-01-01", "2023-01-31", cache=False)

        # No NaN should remain after forward-fill
        assert not result.isna().any()
        # The gap values should be forward-filled from the prior value
        assert result.iloc[2] == pytest.approx(0.0525)
        assert result.iloc[3] == pytest.approx(0.0525)

    @patch("backtester.data.rates.requests.get")
    def test_cache_roundtrip(self, mock_get, tmp_path, monkeypatch):
        mock_get.return_value = _mock_response(
            _make_fred_csv("2023-01-01", "2023-01-10")
        )
        monkeypatch.setattr("backtester.data.rates._CACHE_DIR", tmp_path)

        # First call: fetches from FRED and caches
        r1 = load_risk_free_rate("2023-01-01", "2023-01-10", cache=True)
        assert mock_get.call_count == 1

        # Second call: reads from cache
        r2 = load_risk_free_rate("2023-01-01", "2023-01-10", cache=True)
        assert mock_get.call_count == 1  # no additional fetch
        pd.testing.assert_series_equal(r1, r2)

    @patch("backtester.data.rates.requests.get")
    def test_empty_raises(self, mock_get):
        mock_get.return_value = _mock_response("DATE,DTB3\n")
        with pytest.raises(ValueError, match="no data"):
            load_risk_free_rate("2099-01-01", "2099-01-31", cache=False)


class TestToDailyRate:
    def test_simple_method(self):
        annual = pd.Series([0.05, 0.10])
        daily = to_daily_rate(annual, periods_per_year=252, method="simple")
        expected = pd.Series([0.05 / 252, 0.10 / 252])
        pd.testing.assert_series_equal(daily, expected)

    def test_compounded_method(self):
        annual = pd.Series([0.05])
        daily = to_daily_rate(
            annual, periods_per_year=252, method="compounded"
        )
        expected = (1.05) ** (1 / 252) - 1
        assert daily.iloc[0] == pytest.approx(expected)

    def test_simple_vs_compounded_close(self):
        """For small rates, simple and compounded should be very close."""
        annual = pd.Series([0.05])
        simple = to_daily_rate(annual, method="simple")
        comp = to_daily_rate(annual, method="compounded")
        # Difference should be negligible (< 1 bps = 1e-4)
        assert abs(simple.iloc[0] - comp.iloc[0]) < 1e-4

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            to_daily_rate(pd.Series([0.05]), method="invalid")
