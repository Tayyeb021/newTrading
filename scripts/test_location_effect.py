"""Does the pullback effect survive outside the gold bull market?

The decomposition found one component with evidence: entering after a pullback
beat chasing an extended move by ~0.42 ATR on XAUUSD M15, drift-controlled. But
that sample was 773 days during which gold rose 86%. An effect measured entirely
inside one regime is a hypothesis, not a finding.

This is the test that decides it. Three things change:

- **The trigger is removed.** The decomposition showed it destroyed two thirds of
  the edge and 70% of the sample.
- **Longer timeframes, longer samples.** M30 reaches back 4 years and H1 reaches
  back 8, against M15's 2. Those windows contain periods when gold fell.
- **Split by year.** A real effect shows up in most periods. One that lives in
  2024-2026 and nowhere else was the regime all along.

The statistic is a **Welch t-test on the difference** between the pulled-back and
extended arms. Both arms share whatever drift the period had, so the difference
isolates the location effect from the trend. That is the whole point of testing a
differential rather than a level.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.store import BarStore  # noqa: E402
from strategies.mtf_pullback import MTFPullback, load_bias_frames  # noqa: E402

BIAS_FOR = {"M15": ("H4", "H1"), "M30": ("H4", "H1"), "H1": ("D1", "H4"), "H4": ("D1",)}
HORIZON_FOR = {"M15": 24, "M30": 12, "H1": 6, "H4": 3}  # ~6 hours of exposure each


def arms(df: pd.DataFrame, strategy: MTFPullback, horizon: int):
    """Forward returns for the pulled-back and extended arms, in ATR units.

    No trigger, no stops, no costs -- this measures the location effect alone.
    """
    p = strategy.prepare(df.copy()).reset_index(drop=True)
    close = p["close"].to_numpy(dtype=float)
    a = p["atr"].to_numpy(dtype=float)
    bias = p["bias"].to_numpy(dtype=int)
    ext = np.abs(p["ext_atr"].to_numpy(dtype=float))
    n = len(p)

    fwd = np.full(n, np.nan)
    fwd[: n - horizon] = (close[horizon:] - close[: n - horizon]) / a[: n - horizon]
    signed = fwd * bias

    # Non-overlapping windows only.
    keep = np.zeros(n, dtype=bool)
    keep[strategy.warmup : n - horizon : horizon] = True
    usable = keep & (bias != 0) & np.isfinite(signed) & np.isfinite(ext)

    limit = strategy.max_extension_atr
    return p["ts"].to_numpy(), signed, usable & (ext <= limit), usable & (ext > limit)


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Difference in means, t-statistic, p-value. Unequal variances assumed."""
    if a.size < 5 or b.size < 5:
        return float("nan"), 0.0, 1.0
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(t), float(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+",
                    default=["XAUUSD:M30", "XAUUSD:H1", "XAUUSD:H4",
                             "EURUSD:H1", "GBPUSD:H1"])
    args = ap.parse_args()

    store = BarStore("data/bars")

    print("\nLOCATION EFFECT - pulled back vs extended, trigger removed")
    print("Differential isolates location from drift: both arms share the period's trend.\n")
    print(f"  {'symbol':<9}{'tf':<5}{'span':>7}{'value n':>9}{'value':>9}"
          f"{'ext n':>8}{'ext':>9}{'diff':>9}{'t':>7}{'p':>8}  verdict")
    print("  " + "-" * 96)

    per_symbol: dict[str, list] = {}
    for pair in args.pairs:
        symbol, tf = pair.split(":")
        df = store.read(symbol, tf)
        bias_tfs = BIAS_FOR[tf]
        bias = load_bias_frames(store, symbol, bias_tfs)
        if df.empty or not bias:
            print(f"  {symbol:<9}{tf:<5} no data")
            continue

        strategy = MTFPullback(execution_timeframe=tf, bias_frames=bias,
                               bias_timeframes=bias_tfs, stop_timeframe=None)
        ts, signed, at_value, extended = arms(df, strategy, HORIZON_FOR[tf])
        v, e = signed[at_value], signed[extended]
        diff, t, p = welch(v, e)
        days = (df["ts"].iloc[-1] - df["ts"].iloc[0]).days

        verdict = ("SURVIVES" if abs(t) >= 2 and diff > 0 else
                   "inverted" if abs(t) >= 2 else "no effect")
        print(f"  {symbol:<9}{tf:<5}{days:>6}d{v.size:>9}{v.mean():>+9.3f}"
              f"{e.size:>8}{e.mean():>+9.3f}{diff:>+9.3f}{t:>7.2f}{p:>8.3f}  {verdict}")
        per_symbol[pair] = (ts, signed, at_value, extended, df)

    # ---------------------------------------------------------------- by year
    print("\n\nBY YEAR - does it hold when the market was not rising?\n")
    for pair in ("XAUUSD:H1", "XAUUSD:M30"):
        if pair not in per_symbol:
            continue
        ts, signed, at_value, extended, df = per_symbol[pair]
        years = pd.DatetimeIndex(ts).year
        print(f"  {pair}")
        print(f"    {'year':<6}{'price':>9}{'value n':>9}{'value':>9}{'ext n':>8}"
              f"{'ext':>9}{'diff':>9}{'t':>7}")
        print("    " + "-" * 68)

        wins = total = 0
        for y in sorted(set(years)):
            mask = years == y
            v, e = signed[at_value & mask], signed[extended & mask]
            if v.size < 5 or e.size < 5:
                continue
            diff, t, _ = welch(v, e)
            sub = df[pd.DatetimeIndex(df["ts"]).year == y]
            chg = sub["close"].iloc[-1] / sub["close"].iloc[0] - 1 if len(sub) > 1 else float("nan")
            total += 1
            wins += diff > 0
            flag = "  <-- market fell" if chg < 0 else ""
            print(f"    {y:<6}{chg:>+9.1%}{v.size:>9}{v.mean():>+9.3f}{e.size:>8}"
                  f"{e.mean():>+9.3f}{diff:>+9.3f}{t:>7.2f}{flag}")
        if total:
            print(f"    -> positive in {wins}/{total} years\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
