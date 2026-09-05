"""Is the rollover-hour drift a tradeable edge or a data artifact?

The screen's only survivors were "long during 20:00-02:00 UTC" on EURUSD and
gold. 21:00 UTC is the FX daily rollover and server midnight. Three things can
make a bar-level drift appear there without any money existing:

A. **Spread.** The edge is ~0.04 ATR = ~4 points. If the spread at that hour is
   10-30 points, the effect is real and worthless. Measured from ticks.

B. **Bid-based bars.** Bars close on the bid. If the spread blows out into
   rollover and normalises after, the close dips and 'recovers' with no mid-price
   move at all. Signature: hour_20 negative and hour_21 positive by similar
   amounts. EURUSD shows exactly -0.032 then +0.041.

C. **Where in the hour.** A real drift is spread across the hour. An artifact is
   all in the first five minutes after the break. Decomposed on M5.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.store import BarStore  # noqa: E402
from execution.brokertime import utc_to_server_naive  # noqa: E402
from features.indicators import atr  # noqa: E402


def check_a_spread_at_rollover(symbols=("EURUSD", "XAUUSD"), days_back=(1, 2, 3, 4)):
    import MetaTrader5 as mt5
    from datetime import timezone

    mt5.initialize()
    print("A. SPREAD BY MINUTE AROUND 21:00 UTC (from ticks)")
    now = datetime.now(timezone.utc)
    for sym in symbols:
        mt5.symbol_select(sym, True)
        info = mt5.symbol_info(sym)
        rows = []
        for d in days_back:
            day = (now - timedelta(days=d)).date()
            if day.weekday() >= 5:
                continue
            start_utc = datetime(day.year, day.month, day.day, 20, 50, tzinfo=timezone.utc)
            start_srv = utc_to_server_naive(start_utc)
            ticks = mt5.copy_ticks_from(sym, start_srv, 40000, mt5.COPY_TICKS_INFO)
            if ticks is None or len(ticks) == 0:
                continue
            t = pd.DataFrame(ticks)
            t = t[(t["bid"] > 0) & (t["ask"] > 0)]
            t["srv"] = pd.to_datetime(t["time"], unit="s")
            t["utc_min"] = ((t["srv"] - start_srv).dt.total_seconds() // 60).astype(int) - 10
            t["spread_pts"] = (t["ask"] - t["bid"]) / info.point
            rows.append(t[["utc_min", "spread_pts"]])
        if not rows:
            print(f"  {sym}: no ticks available")
            continue
        all_t = pd.concat(rows)
        by_min = all_t.groupby("utc_min")["spread_pts"].median()
        print(f"  {sym}  (median spread in points, minutes relative to 21:00 UTC; normal daytime spread ~{by_min.loc[-10:-6].median():.0f})")
        line = "   "
        for m in range(-10, 70, 5):
            v = by_min.get(m, np.nan)
            line += f" {m:+3d}m:{v:5.0f}"
        print(line)
    mt5.shutdown()


def check_b_symmetry_and_gap(store):
    print("\nB. BID-BAR ARTIFACT SIGNATURE  (hour_20 vs hour_21, and gap vs intrabar)")
    for sym in ("EURUSD", "XAUUSD"):
        df = store.read(sym, "H1"); df = df[df["ts"].dt.year >= 2010].reset_index(drop=True)
        a = atr(df, 14).to_numpy(); c = df["close"].to_numpy(); o = df["open"].to_numpy()
        h = df["ts"].dt.hour.to_numpy()
        fwd = np.full(len(df), np.nan); fwd[:-1] = (c[1:] - c[:-1]) / a[:-1]
        drift = np.nanmean(fwd)
        # the 'forward return' of the bar stamped hour h runs from its close to the next close;
        # split that into the GAP (next open - this close) and INTRABAR (next close - next open)
        gap = np.full(len(df), np.nan); gap[:-1] = (o[1:] - c[:-1]) / a[:-1]
        intra = np.full(len(df), np.nan); intra[:-1] = (c[1:] - o[1:]) / a[:-1]
        print(f"  {sym}")
        print(f"    {'bar':>6}{'fwd':>9}{'gap':>9}{'intrabar':>10}{'n':>7}   (demeaned, ATR units)")
        for hh in (19, 20, 21, 22, 0, 1):
            m = h == hh
            print(f"    {hh:02d}:00{np.nanmean(fwd[m]) - drift:>+9.3f}{np.nanmean(gap[m]) - np.nanmean(gap):>+9.3f}"
                  f"{np.nanmean(intra[m]) - np.nanmean(intra):>+10.3f}{m.sum():>7}")


def check_c_where_in_the_hour(store):
    print("\nC. WHERE INSIDE 21:00-23:00 UTC THE RETURN OCCURS  (M5, demeaned, ATR of the H1)")
    for sym in ("EURUSD", "XAUUSD"):
        df = store.read(sym, "M5"); df = df[df["ts"].dt.year >= 2016].reset_index(drop=True)
        h1atr = atr(df, 14 * 12).to_numpy()  # ~H1-scale ATR from M5
        c = df["close"].to_numpy()
        r = np.full(len(df), np.nan); r[:-1] = (c[1:] - c[:-1]) / h1atr[:-1]
        r -= np.nanmean(r)
        slot = df["ts"].dt.hour.to_numpy() * 60 + df["ts"].dt.minute.to_numpy()
        print(f"  {sym}   5-min slot -> mean demeaned return (x1000), n")
        line, total = "   ", 0.0
        for hh in (21, 22):
            for mm in range(0, 60, 5):
                m = slot == hh * 60 + mm
                v = np.nanmean(r[m]) * 1000 if m.sum() else np.nan
                total += 0 if np.isnan(v) else v
                line += f" {hh:02d}:{mm:02d}={v:+5.1f}"
            print(line); line = "   "
        print(f"    sum over the two hours: {total:+.1f}  (an artifact concentrates in the first slot)")


if __name__ == "__main__":
    store = BarStore("data/bars")
    check_a_spread_at_rollover()
    check_b_symmetry_and_gap(store)
    check_c_where_in_the_hour(store)
