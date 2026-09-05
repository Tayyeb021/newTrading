"""Positioning as a signal: CFTC COT against forward returns.

The first hypothesis class this project could never ask about, because it is
not a function of price. Every row is the weekly futures positioning of large
speculators and commercial hedgers, joined to daily bars strictly by its
PUBLICATION time (Friday evening), never its as-of date.

The evidence is honestly mixed, so each construction states what it is
testing and whose finding it would confirm:

- **Speculator extremes fade** -- Wang (2001) finds non-commercial positioning
  is a continuation indicator but that extremes revert; a "COT index" above
  0.9 is a crowded long.
- **Speculator change follows** -- the 4-week change in net speculative
  position as a continuation signal.
- **Commercials lead** -- Wang finds commercial sentiment a contrary indicator:
  when hedgers are unusually long, go long.
- **Spec minus commercial** -- the two disagree most at turns.

Sanders et al. (2004) found none of this in energy and warned that one
market's result says nothing about another's. So each market is scored
separately and nothing is pooled.

Price data here is the IC Markets daily CFD for the same underlying, used as a
proxy for the futures return. Spot gold and COMEX gold move together to well
within one day's ATR, and the screen demeans returns, so the proxy is sound
for a *signal-level* test. Costs are not modelled at this stage by design.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cot import COTStore, features, join_to_bars  # noqa: E402
from data.store import BarStore  # noqa: E402
from research.screen import Hypothesis, score  # noqa: E402

#: COT market -> daily price series that tracks the same underlying.
PROXY = {"GC": "XAUUSD", "ES": "US500", "6E": "EURUSD"}


def sig_spec_extreme_fade(df):
    idx = df["cot_spec_net_index"].to_numpy(dtype=float)
    return np.where(idx > 0.9, -1.0, np.where(idx < 0.1, 1.0, 0.0))


def sig_spec_change_follow(df):
    return np.sign(np.nan_to_num(df["cot_spec_net_chg4"].to_numpy(dtype=float)))


def sig_commercial_lead(df):
    idx = df["cot_comm_net_index"].to_numpy(dtype=float)
    return np.where(idx > 0.8, 1.0, np.where(idx < 0.2, -1.0, 0.0))


def sig_spec_minus_comm(df):
    d = df["cot_spec_net_index"].to_numpy(dtype=float) - df["cot_comm_net_index"].to_numpy(dtype=float)
    return np.where(d > 0.6, -1.0, np.where(d < -0.6, 1.0, 0.0))


CONSTRUCTIONS = [
    ("spec_extreme_fade", "crowded speculative longs revert (Wang 2001)", sig_spec_extreme_fade),
    ("spec_change_follow", "4-week change in spec positioning continues (Wang 2003)", sig_spec_change_follow),
    ("commercial_lead", "hedgers are the contrary indicator: long when they are", sig_commercial_lead),
    ("spec_minus_comm", "fade when speculators and hedgers disagree most", sig_spec_minus_comm),
]
HORIZONS = (5, 20)  # trading days


def main() -> int:
    bars = BarStore("data/bars")
    cot = COTStore("data/cot")
    trials = len(PROXY) * len(CONSTRUCTIONS) * len(HORIZONS)
    bar = stats.norm.ppf(1 - 0.025 / trials)

    print(f"COT POSITIONING SCREEN  -  {trials} trials, bar |t| > {bar:.2f}, positive in >= 70% of years")
    print("signal joined at PUBLICATION time (Fri 21:00 UTC), demeaned forward return in ATR units\n")
    print(f"  {'construction':<22}{'market':<8}{'h':>3}{'n':>7}{'excess':>9}{'t':>7}{'hit':>6}{'years+':>9}  verdict")
    print("  " + "-" * 84)

    scores = []
    for code, proxy in PROXY.items():
        raw = cot.read(code)
        if raw.empty:
            print(f"  {code}: no COT data - run scripts/download_cot.py"); continue
        px = bars.read(proxy, "D1")
        px = px[px["ts"].dt.year >= 2016].reset_index(drop=True)
        if px.empty:
            print(f"  {proxy}: no bars"); continue
        joined = join_to_bars(px, features(raw))
        for name, why, fn in CONSTRUCTIONS:
            for h in HORIZONS:
                hyp = Hypothesis(f"{name}", why, [proxy], "D1", h, fn)
                s = score(hyp, joined, proxy, trials)
                scores.append((code, h, s))
                print(f"  {name:<22}{code + '/' + proxy:<8}{h:>3}{s.n:>7,}{s.excess_atr:>+9.3f}{s.t:>7.2f}"
                      f"{s.hit:>6.0%}{s.years_pos:>5}/{s.years:<3}  {s.verdict}")

    survivors = [(c, h, s) for c, h, s in scores if s.verdict == "SURVIVES"]
    print(f"\n  survivors: {len(survivors)}")
    for c, h, s in survivors:
        print(f"    {s.hypothesis} on {c} at {h}d: {s.excess_atr:+.3f} ATR, t={s.t:.2f}, {s.years_pos}/{s.years} years")
    best = max(scores, key=lambda x: abs(x[2].t)) if scores else None
    if best and not survivors:
        c, h, s = best
        print(f"  nearest: {s.hypothesis} on {c} at {h}d, t={s.t:.2f} ({s.years_pos}/{s.years} years)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
