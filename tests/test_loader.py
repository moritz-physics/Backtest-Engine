"""Tests for backtester.data.loader.

All tests use mocked yfinance — no network calls.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backtester.data.loader import load_ohlcv, load_prices, to_returns


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(
    dates: pd.DatetimeIndex,
    close: list[float] | np.ndarray,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame matching yfinance's output format."""
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.full(len(close), 1_000_000),
        },
        index=dates,
    )


# 10 trading days
DATES_FULL = pd.bdate_range("2023-01-02", periods=10)
# Same but missing day 5 (index 4)
DATES_GAP = DATES_FULL.delete(4)

PRICES_A = np.linspace(100, 110, 10)
PRICES_B = np.linspace(200, 220, 10)
PRICES_C = np.linspace(50, 55, len(DATES_GAP))  # 9 days


def _mock_download_single(tickers, *, start, end, auto_adjust, progress):
    """Mock for single-ticker download."""
    ticker = tickers if isinstance(tickers, str) else tickers[0]
    lookup = {
        "AAA": _make_ohlcv(DATES_FULL, PRICES_A),
        "BBB": _make_ohlcv(DATES_FULL, PRICES_B),
        "CCC": _make_ohlcv(DATES_GAP, PRICES_C),
    }
    return lookup[ticker]


def _mock_download_multi(tickers, *, start, end, auto_adjust, progress):
    """Mock for multi-ticker download — returns MultiIndex columns."""
    if isinstance(tickers, str):
        tickers = [tickers]
    frames = {
        "AAA": _make_ohlcv(DATES_FULL, PRICES_A),
        "BBB": _make_ohlcv(DATES_FULL, PRICES_B),
        "CCC": _make_ohlcv(DATES_GAP, PRICES_C),
    }
    pieces = {}
    for ticker in tickers:
        df = frames[ticker]
        for col in df.columns:
            pieces[(col, ticker)] = df[col]
    result = pd.DataFrame(pieces)
    result.columns = pd.MultiIndex.from_tuples(
        result.columns, names=["Price", "Ticker"]
    )
    return result


def _mock_download_auto(tickers, *, start, end, auto_adjust, progress):
    """Route to single or multi mock based on ticker count."""
    if isinstance(tickers, str):
        tickers = [tickers]
    if len(tickers) == 1:
        return _mock_download_single(tickers, start=start, end=end,
                                     auto_adjust=auto_adjust, progress=progress)
    return _mock_download_multi(tickers, start=start, end=end,
                                auto_adjust=auto_adjust, progress=progress)


# ---------------------------------------------------------------------------
# Tests: load_prices
# ---------------------------------------------------------------------------


@patch("backtester.data.loader.yf.download", side_effect=_mock_download_auto)
def test_single_ticker_shape(mock_dl, tmp_path, monkeypatch):
    monkeypatch.setattr("backtester.data.loader._CACHE_DIR", tmp_path / "cache")
    result = load_prices("AAA", "2023-01-02", "2023-01-16")

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["AAA"]
    assert len(result) == 10
    # Sorted ascending
    assert (result.index == result.index.sort_values()).all()


@patch("backtester.data.loader.yf.download", side_effect=_mock_download_auto)
def test_multi_ticker_alignment(mock_dl, tmp_path, monkeypatch):
    monkeypatch.setattr("backtester.data.loader._CACHE_DIR", tmp_path / "cache")
    result = load_prices(["AAA", "BBB", "CCC"], "2023-01-02", "2023-01-16")

    # CCC is missing 1 day → inner join should have 9 rows
    assert len(result) == 9
    assert list(result.columns) == ["AAA", "BBB", "CCC"]
    assert not result.isna().any().any()


@patch("backtester.data.loader.yf.download", side_effect=_mock_download_auto)
def test_cache_roundtrip(mock_dl, tmp_path, monkeypatch):
    monkeypatch.setattr("backtester.data.loader._CACHE_DIR", tmp_path / "cache")

    first = load_prices("AAA", "2023-01-02", "2023-01-16", cache=True)

    # On second call, make yfinance raise — must come from cache
    with patch(
        "backtester.data.loader.yf.download",
        side_effect=RuntimeError("should not be called"),
    ):
        second = load_prices("AAA", "2023-01-02", "2023-01-16", cache=True)

    pd.testing.assert_frame_equal(first, second, check_freq=False)


@patch("backtester.data.loader.yf.download", side_effect=_mock_download_auto)
def test_load_prices_cache_false_always_fetches(mock_dl, tmp_path, monkeypatch):
    monkeypatch.setattr("backtester.data.loader._CACHE_DIR", tmp_path / "cache")

    # First call with cache=True to populate cache file
    load_prices("AAA", "2023-01-02", "2023-01-16", cache=True)
    assert mock_dl.call_count == 1

    # Second call with cache=False — should fetch again
    load_prices("AAA", "2023-01-02", "2023-01-16", cache=False)
    assert mock_dl.call_count == 2


# ---------------------------------------------------------------------------
# Tests: load_ohlcv
# ---------------------------------------------------------------------------


@patch("backtester.data.loader.yf.download", side_effect=_mock_download_auto)
def test_load_ohlcv_returns_all_fields(mock_dl, tmp_path, monkeypatch):
    monkeypatch.setattr("backtester.data.loader._CACHE_DIR", tmp_path / "cache")
    result = load_ohlcv(["AAA", "BBB"], "2023-01-02", "2023-01-16")

    assert isinstance(result.columns, pd.MultiIndex)
    fields = result.columns.get_level_values("Field").unique().tolist()
    assert sorted(fields) == ["Close", "High", "Low", "Open", "Volume"]
    tickers = result.columns.get_level_values("Ticker").unique().tolist()
    assert sorted(tickers) == ["AAA", "BBB"]
    assert len(result) == 10


# ---------------------------------------------------------------------------
# Tests: alignment
# ---------------------------------------------------------------------------


@patch("backtester.data.loader.yf.download", side_effect=_mock_download_auto)
def test_alignment_outer_preserves_nan(mock_dl, tmp_path, monkeypatch):
    monkeypatch.setattr("backtester.data.loader._CACHE_DIR", tmp_path / "cache")
    result = load_prices(
        ["AAA", "BBB", "CCC"],
        "2023-01-02", "2023-01-16",
        alignment="outer",
    )

    # Outer: all 10 dates kept, CCC has 1 NaN
    assert len(result) == 10
    assert result["CCC"].isna().sum() == 1
    assert not result["AAA"].isna().any()


@patch("backtester.data.loader.yf.download", side_effect=_mock_download_auto)
def test_alignment_inner_warns_on_heavy_drop(mock_dl, tmp_path, monkeypatch, caplog):
    """When one ticker covers only ~20% of the period, WARNING is logged."""
    # Create a ticker that only has the last 2 of 10 days
    short_dates = DATES_FULL[-2:]
    short_prices = [100.0, 101.0]

    def mock_with_short(tickers, *, start, end, auto_adjust, progress):
        if isinstance(tickers, str):
            tickers = [tickers]
        frames = {
            "AAA": _make_ohlcv(DATES_FULL, PRICES_A),
            "SHORT": _make_ohlcv(short_dates, short_prices),
        }
        if len(tickers) == 1:
            return frames[tickers[0]]
        pieces = {}
        for ticker in tickers:
            df = frames[ticker]
            for col in df.columns:
                pieces[(col, ticker)] = df[col]
        result = pd.DataFrame(pieces)
        result.columns = pd.MultiIndex.from_tuples(
            result.columns, names=["Price", "Ticker"]
        )
        return result

    monkeypatch.setattr("backtester.data.loader._CACHE_DIR", tmp_path / "cache")
    with patch("backtester.data.loader.yf.download", side_effect=mock_with_short):
        with caplog.at_level(logging.WARNING, logger="backtester.data.loader"):
            result = load_prices(
                ["AAA", "SHORT"],
                "2023-01-02", "2023-01-16",
            )

    # Only 2 rows survive
    assert len(result) == 2
    # WARNING should mention dropped dates and the worst ticker
    assert any("alignment dropped" in r.message.lower() for r in caplog.records)
    assert any("SHORT" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests: to_returns
# ---------------------------------------------------------------------------


def test_to_returns_constant():
    prices = pd.DataFrame(
        {"X": [100.0] * 5},
        index=pd.bdate_range("2023-01-02", periods=5),
    )
    ret = to_returns(prices, method="log")
    assert ret.iloc[0].isna().all()
    assert (ret.iloc[1:] == 0.0).all().all()

    ret_simple = to_returns(prices, method="simple")
    assert ret_simple.iloc[0].isna().all()
    assert (ret_simple.iloc[1:] == 0.0).all().all()


def test_to_returns_nan_only_first_row():
    prices = pd.DataFrame(
        {"X": [100.0, 102.0, 101.0, 105.0, 103.0]},
        index=pd.bdate_range("2023-01-02", periods=5),
    )
    ret = to_returns(prices, method="log")
    assert ret.iloc[0].isna().all()
    assert not ret.iloc[1:].isna().any().any()


def test_to_returns_log_vs_simple():
    prices = pd.Series(
        [100.0, 110.0],
        index=pd.bdate_range("2023-01-02", periods=2),
    )

    log_ret = to_returns(prices, method="log")
    expected_log = np.log(110.0 / 100.0)
    assert np.isclose(log_ret.iloc[1], expected_log)

    simple_ret = to_returns(prices, method="simple")
    expected_simple = (110.0 - 100.0) / 100.0
    assert np.isclose(simple_ret.iloc[1], expected_simple)


def test_nan_mid_series_raises():
    prices = pd.Series(
        [100.0, np.nan, 102.0],
        index=pd.bdate_range("2023-01-02", periods=3),
    )
    with pytest.raises(ValueError, match="mid-series NaN"):
        to_returns(prices)
