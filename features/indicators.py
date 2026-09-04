"""Causal indicators.

Every function here is **causal**: the value at index `i` depends only on data at
indices <= `i`. That property is what makes precomputing a whole series safe, and
it is not optional. A single centred rolling window or a `shift(-1)` anywhere in
this file would produce a backtest that looks superb and cannot be traded.

`tests/test_backtest.py::test_no_lookahead_in_signals` verifies the property
empirically rather than trusting this docstring: it truncates the data and checks
that earlier signals are bit-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average. `adjust=False` gives the recursive form, which
    is what a live incremental implementation would produce bar by bar."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """max(high-low, |high-prev_close|, |low-prev_close|)."""
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR — an EMA of true range with alpha = 1/period."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rolling_return(series: pd.Series, period: int) -> pd.Series:
    """Simple return over `period` bars: close[i] / close[i-period] - 1."""
    return series / series.shift(period) - 1.0


def rolling_percentile(series: pd.Series, period: int) -> pd.Series:
    """Where the current value sits in its own trailing distribution, in [0, 1].

    Used for volatility filters: an ATR percentile above ~0.95 means the market
    just did something unusual, which is normally the worst moment to open a new
    trend position.
    """
    return series.rolling(period, min_periods=period).rank(pct=True)


def realized_vol(series: pd.Series, period: int, periods_per_year: int = 252) -> pd.Series:
    """Annualised standard deviation of log returns."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(period, min_periods=period).std() * np.sqrt(periods_per_year)


def donchian(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series]:
    """Highest high and lowest low over the previous `period` bars, EXCLUDING now.

    The exclusion matters. Including the current bar makes a breakout condition
    trivially true at the moment it is tested, which is the single most common
    look-ahead bug in breakout backtests.
    """
    upper = df["high"].rolling(period, min_periods=period).max().shift(1)
    lower = df["low"].rolling(period, min_periods=period).min().shift(1)
    return upper, lower
