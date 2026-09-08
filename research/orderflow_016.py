"""016: does order flow predict anything, and does it survive costs?

    python research/orderflow_016.py --root ES

Three practitioner claims, pre-registered in RESEARCH_LOG.md before the data was
bought, tested with entry 003's machinery rather than a new one: demeaned
forward returns in ATR units, non-overlapping windows, Bonferroni across the
declared family of nine, the sign required consistent across calendar months,
and - the threshold that kills intraday ideas - the edge compared against the
real round-trip friction of one MES contract at the *measured* spread.

The prints are real: 30.3 million ES trades over three months, each carrying
Databento's aggressor flag, so the side is KNOWN rather than inferred by the
tick rule. That distinction matters here more than anywhere else in this
repository, because the tick rule misclassifies a large minority of prints and
every quantity below is built from the side.

Two contracts cover the window and only one is front month at a time. Bars are
built per contract and pooled afterwards, so no rolling window and no forward
return ever spans the June roll: a seam in a back-adjusted series would show up
as a delta signal that is really just basis.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.contracts import ALL_ROOTS  # noqa: E402
from features.indicators import atr  # noqa: E402
from features.orderflow import delta_bars, delta_divergence  # noqa: E402

TICKS = ROOT / "data" / "ticks"
US_OPEN_UTC = (13, 30)   # 09:30 New York, the cash open
US_CLOSE_UTC = (20, 0)   # 16:00 New York
BURN_IN = 50             # the longest lookback here is 20 bars; 003 used 300 for its 250-day features


def front_windows(root, files: list[Path]) -> dict[str, tuple[date, date]]:
    """Which calendar days each contract was actually the front month.

    Databento returns the full requested range for every symbol asked for, so
    both contracts cover June. Trading the back month is not what any of these
    hypotheses claim, and its thin book would flatter the delta measures.
    """
    codes = {f.stem for f in files}
    spans: dict[str, list[date]] = {}
    d = min(pd.read_parquet(f, columns=["ts"])["ts"].min() for f in files).date()
    last = max(pd.read_parquet(f, columns=["ts"])["ts"].max() for f in files).date()
    while d <= last:
        c = root.code(*root.front(d))
        if c in codes:
            spans.setdefault(c, []).append(d)
        d += timedelta(days=1)
    return {c: (days[0], days[-1]) for c, days in spans.items()}


def measured_spread(prints: pd.DataFrame, sample: int = 2_000_000) -> float:
    """The spread the data itself shows, not one assumed.

    Consecutive prints on opposite sides of the book, within a second of each
    other, are a buyer lifting the offer and a seller hitting the bid: the gap
    between them is the spread being crossed. The median of that is the number
    a taker actually pays.
    """
    p = prints.iloc[len(prints) // 2: len(prints) // 2 + sample]
    dt = p["ts"].diff().dt.total_seconds().to_numpy()
    dside = np.abs(np.diff(p["side"].to_numpy(), prepend=np.nan))
    dpx = np.abs(p["price"].diff().to_numpy())
    flip = (dside == 2) & (dt <= 1.0) & np.isfinite(dpx) & (dpx > 0)
    return float(np.median(dpx[flip])) if flip.sum() > 1000 else float("nan")


def score(name: str, obs: pd.DataFrame, trials: int, cost_atr: float) -> dict:
    """`obs` has columns sig, fwd, ts - already pooled, already non-overlapping.

    Demeaning is against the WHOLE sample's forward return, not just the
    signalled bars, so a rule that happens to fire in an up-trending window is
    not credited with the trend.
    """
    drift = float(np.nanmean(obs["fwd"]))
    fired = obs[obs["sig"] != 0]
    if len(fired) < 30:
        return {"name": name, "n": int(len(fired)), "verdict": "too few"}

    signed = ((fired["fwd"] - drift) * fired["sig"]).to_numpy(dtype=float)
    t, p = stats.ttest_1samp(signed, 0.0)
    stamps = pd.DatetimeIndex(fired["ts"])
    if stamps.tz is not None:
        stamps = stamps.tz_convert(None)  # every timestamp here is already UTC
    by_month = pd.Series(signed).groupby(stamps.to_period("M").values).mean()

    edge = float(signed.mean())
    bonf = bool(p < 0.05 / trials)
    consistent = bool(len(by_month) >= 3 and (by_month > 0).sum() / len(by_month) >= 0.70)
    pays = bool(edge > 2 * cost_atr)
    return {
        "name": name, "n": int(len(fired)), "edge_atr": edge, "t": float(t), "p": float(p),
        "hit": float((signed > 0).mean()), "drift_atr": drift,
        "months_pos": int((by_month > 0).sum()), "months": int(len(by_month)),
        "by_month": {str(k): round(float(v), 5) for k, v in by_month.items()},
        "bonferroni": bonf, "consistent": consistent, "beats_2x_cost": pays,
        "cost_atr": cost_atr, "needed_atr": 2 * cost_atr,
        "verdict": "SURVIVES" if (bonf and consistent and pays) else
                   ("real but does not pay" if (bonf and consistent) else
                    ("significant, inconsistent" if bonf else "no signal")),
    }


def bar_observations(bars: pd.DataFrame, sig: np.ndarray) -> pd.DataFrame:
    """One row per bar: the signal known at its close, and the next bar's move.

    The last bar of a contract gets no forward return, so nothing is ever
    measured across the roll.
    """
    a = atr(bars, 14).to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    n = len(bars)
    fwd = np.full(n, np.nan)
    fwd[: n - 1] = (close[1:] - close[: n - 1]) / a[: n - 1]
    out = pd.DataFrame({"ts": bars["ts"].to_numpy(), "sig": np.nan_to_num(sig), "fwd": fwd})
    out.iloc[:BURN_IN, out.columns.get_loc("sig")] = 0.0
    return out[np.isfinite(out["fwd"])]


def session_observations(bars5: pd.DataFrame) -> pd.DataFrame:
    """Hypothesis 3: signed volume in the first 30 minutes of the cash session,
    against the move from 14:00 UTC to the 16:00 New York close.

    Measured by clock, not by bar count: a bar-count horizon would jump the
    overnight halt on any day with a missing bar and silently score the wrong
    window.
    """
    b = bars5.copy()
    ts = pd.DatetimeIndex(b["ts"])
    b["day"] = ts.date
    minute = ts.hour * 60 + ts.minute
    open_min = US_OPEN_UTC[0] * 60 + US_OPEN_UTC[1]
    close_min = US_CLOSE_UTC[0] * 60 + US_CLOSE_UTC[1]
    a = atr(b, 14)

    rows = []
    for day, g in b.groupby("day", sort=True):
        m = minute[b["day"].to_numpy() == day]
        window = g[(m >= open_min) & (m < open_min + 30)]
        entry = g[m == open_min + 30]
        exit_ = g[m == close_min]
        if window.empty or entry.empty or exit_.empty:
            continue
        atr_at_entry = float(a.loc[entry.index[0]])
        if not np.isfinite(atr_at_entry) or atr_at_entry <= 0:
            continue
        rows.append({
            "ts": entry["ts"].iloc[0],
            "sig": float(np.sign(window["delta"].sum())),
            "fwd": (float(exit_["close"].iloc[0]) - float(entry["close"].iloc[0])) / atr_at_entry,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="ES")
    ap.add_argument("--cost-root", default="MES", help="the contract whose friction the edge must beat")
    ap.add_argument("--trials", type=int, default=9, help="declared family size for Bonferroni")
    args = ap.parse_args()

    root, cost_root = ALL_ROOTS[args.root], ALL_ROOTS[args.cost_root]
    files = sorted((TICKS / args.root).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no prints under {TICKS / args.root}; run scripts/download_ticks.py --root {args.root}")

    windows = front_windows(root, files)
    print(f"\nORDER FLOW - entry 016, {args.root}")
    print("=" * 78)

    spreads, per_contract = [], {}
    for f in files:
        code = f.stem
        if code not in windows:
            print(f"  {code}: never front month in the sample, skipped")
            continue
        lo, hi = windows[code]
        p = pd.read_parquet(f)
        p = p[(p["ts"].dt.date >= lo) & (p["ts"].dt.date <= hi)].reset_index(drop=True)
        if p.empty:
            continue
        spreads.append((code, measured_spread(p), len(p)))
        per_contract[code] = p
        print(f"  {code}: front {lo} -> {hi}, {len(p):,} prints, "
              f"measured spread {spreads[-1][1]:.4f} pts")

    spread = float(np.average([s for _, s, _ in spreads], weights=[n for _, _, n in spreads]))
    comm_pts = cost_root.commission_per_side / cost_root.multiplier
    round_trip = spread + 2 * comm_pts
    print(f"\n  friction of one {args.cost_root}: spread {spread:.4f} + commission "
          f"{2 * comm_pts:.4f} = {round_trip:.4f} {args.root} points per round trip")
    print(f"  threshold 3 requires an edge above 2x that, i.e. {2 * round_trip:.4f} points")

    results, costs = [], {}
    bars_by_freq: dict[int, list[pd.DataFrame]] = {5: [], 15: [], 60: []}
    for code, p in per_contract.items():
        for minutes in bars_by_freq:
            bars_by_freq[minutes].append(delta_bars(p, f"{minutes}min"))

    for minutes, frames in bars_by_freq.items():
        a_all = pd.concat([atr(b, 14) for b in frames])
        median_atr = float(a_all.median())
        cost_atr = round_trip / median_atr
        costs[minutes] = (median_atr, cost_atr)
        print(f"\n  {minutes:>2}min bars: {sum(len(b) for b in frames):,}   "
              f"median ATR {median_atr:.2f} pts   round trip {cost_atr:.4f} ATR")

        cont, div = [], []
        for b in frames:
            # 1. continuation: the sign of this bar's signed volume, held one bar
            cont.append(bar_observations(b, np.sign(b["delta"].to_numpy())))
            # 2. divergence: the library returns +1 for a BEARISH divergence, so
            #    the position it implies is the opposite sign.
            div.append(bar_observations(b, -delta_divergence(b, 20).to_numpy(dtype=float)))
        results.append(score(f"delta_continuation_{minutes}m", pd.concat(cont), args.trials, cost_atr))
        results.append(score(f"delta_divergence_{minutes}m", pd.concat(div), args.trials, cost_atr))

    sess = pd.concat([session_observations(b) for b in bars_by_freq[5]], ignore_index=True)
    results.append(score("opening_imbalance", sess, args.trials, costs[5][1]))
    print(f"\n  opening imbalance: {len(sess)} complete cash sessions "
          f"(14:00 -> 20:00 UTC, {costs[5][1]:.4f} ATR round trip)")

    print(f"\n  {'hypothesis':<26}{'n':>7}{'edge ATR':>10}{'needs':>9}{'t':>8}{'hit':>7}"
          f"{'months':>8}   verdict")
    print("  " + "-" * 92)
    for r in results:
        if r["verdict"] == "too few":
            print(f"  {r['name']:<26}{r['n']:>7}   too few observations to judge")
            continue
        months = f"{r['months_pos']}/{r['months']}"
        print(f"  {r['name']:<26}{r['n']:>7}{r['edge_atr']:>10.4f}{r['needed_atr']:>9.4f}"
              f"{r['t']:>8.2f}{r['hit']:>7.1%}{months:>8}   {r['verdict']}")

    survivors = [r for r in results if r["verdict"] == "SURVIVES"]
    real = [r for r in results if r["verdict"] == "real but does not pay"]
    signif = [r for r in results if r.get("bonferroni")]
    print(f"\nVERDICT, entry 016")
    print("=" * 78)
    print(f"  Bonferroni bar for {args.trials} trials: p < {0.05 / args.trials:.5f}")
    print(f"  {len(signif)} of {len(results)} clear threshold 1 (the effect exists before costs)")
    print(f"  {len(survivors)} of {len(results)} clear all three")
    for r in real:
        print(f"    {r['name']:<26} real, edge {r['edge_atr']:.4f} ATR against "
              f"{r['needed_atr']:.4f} needed - short by {r['needed_atr'] - r['edge_atr']:.4f}")
    print(f"\n  {'PASSED - extend to twelve months' if survivors else 'FAILED'}\n")

    (ROOT / "state").mkdir(exist_ok=True)
    (ROOT / "state" / "gauntlet_016.json").write_text(json.dumps({
        "root": args.root, "prints": int(sum(len(p) for p in per_contract.values())),
        "spread_pts": spread, "round_trip_pts": round_trip, "cost_root": args.cost_root,
        "trials": args.trials, "results": results, "survivors": len(survivors),
        "passed": bool(survivors)}, indent=2, default=str), encoding="utf-8")
    return 0 if survivors else 1


if __name__ == "__main__":
    raise SystemExit(main())
