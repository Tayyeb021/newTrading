"""Order-flow features from trade prints.

The signal class that bars cannot express. A bar says price went from A to B on
N contracts. Trades say *who crossed the spread to make it happen*: buyers
lifting the offer or sellers hitting the bid. That difference -- delta -- and
where it concentrates is the raw material of every order-flow method.

Input is a trades frame with columns `ts`, `price`, `size`, and `side`, where
side is +1 for a buyer-initiated print (aggressor bought at the ask) and -1 for
seller-initiated. Databento's MBP-1 and trades schemas carry this directly; on
data that lacks it, `infer_side` applies the tick rule, which is the standard
approximation and is labelled one.

Everything here is causal: a feature at bar t reads prints up to t only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def infer_side(trades: pd.DataFrame) -> pd.Series:
    """Tick rule: an uptick is a buy, a downtick a sell, unchanged inherits."""
    d = np.sign(trades["price"].diff())
    side = d.replace(0, np.nan).ffill().fillna(0)
    return side.astype(int)


def delta_bars(trades: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Per-bar buy volume, sell volume, delta, and the bar's price range.

    `delta_pct` is delta over total volume: +1.0 means every contract was
    bought at the offer. `absorption` is volume per unit of range, high when
    size traded and price did not move -- a passive party absorbed it.
    """
    t = trades.copy()
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    if "side" not in t.columns:
        t["side"] = infer_side(t)
    t["buy"] = np.where(t["side"] > 0, t["size"], 0.0)
    t["sell"] = np.where(t["side"] < 0, t["size"], 0.0)
    g = t.set_index("ts").resample(freq, label="left", closed="left")
    out = pd.DataFrame({
        "open": g["price"].first(),
        "high": g["price"].max(),
        "low": g["price"].min(),
        "close": g["price"].last(),
        "volume": g["size"].sum(),
        "buy_vol": g["buy"].sum(),
        "sell_vol": g["sell"].sum(),
        "prints": g["size"].count(),
    }).dropna(subset=["open"])
    out["delta"] = out["buy_vol"] - out["sell_vol"]
    out["delta_pct"] = out["delta"] / out["volume"].replace(0, np.nan)
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["absorption"] = out["volume"] / rng
    out["cum_delta"] = out["delta"].cumsum()
    return out.reset_index().rename(columns={"ts": "ts"})


def imbalance(bars: pd.DataFrame, window: int = 12) -> pd.Series:
    """Rolling delta as a share of rolling volume. Causal."""
    d = bars["delta"].rolling(window, min_periods=window).sum()
    v = bars["volume"].rolling(window, min_periods=window).sum().replace(0, np.nan)
    return d / v


def delta_divergence(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """Price making a new high while cumulative delta does not: buyers are not
    behind the move. Returns +1 (bearish divergence), -1 (bullish), 0."""
    px_hi = bars["close"] >= bars["close"].rolling(window, min_periods=window).max()
    px_lo = bars["close"] <= bars["close"].rolling(window, min_periods=window).min()
    cd_hi = bars["cum_delta"] >= bars["cum_delta"].rolling(window, min_periods=window).max()
    cd_lo = bars["cum_delta"] <= bars["cum_delta"].rolling(window, min_periods=window).min()
    out = pd.Series(0, index=bars.index, dtype=int)
    out[px_hi & ~cd_hi] = 1
    out[px_lo & ~cd_lo] = -1
    return out


def open_imbalance(bars: pd.DataFrame, open_hour: int, open_minute: int, minutes: int = 30) -> pd.DataFrame:
    """Delta in the first `minutes` after a session open, one row per session.

    The hypothesis it feeds: does the opening imbalance predict the session's
    direction? The screen decides; this only measures.
    """
    b = bars.copy()
    b["ts"] = pd.to_datetime(b["ts"], utc=True)
    b["day"] = b["ts"].dt.date
    start = b["ts"].dt.hour * 60 + b["ts"].dt.minute
    open_min = open_hour * 60 + open_minute
    window = b[(start >= open_min) & (start < open_min + minutes)]
    g = window.groupby("day")
    out = pd.DataFrame({
        "open_delta": g["delta"].sum(),
        "open_volume": g["volume"].sum(),
        "open_close": g["close"].last(),
        "open_first": g["open"].first(),
        "available_at": g["ts"].max() + pd.Timedelta(minutes=5),  # after the last bar in the window closes
    })
    out["open_delta_pct"] = out["open_delta"] / out["open_volume"].replace(0, np.nan)
    return out.reset_index()
