"""015a: cross-sectional value on futures, as the evidence was built.

    python research/value_015.py
    python research/value_015.py --with-book     # beside momentum and carry

Asness, Moskowitz and Pedersen (2013) measure futures value as a long-horizon
reversal and rank markets **against each other**. What has fallen for five years
is cheap; what has risen is dear. Testing it as a per-market timing rule would
be testing something else and calling it value, so this does not use the
Strategy interface at all - it builds a ranked, monthly-rebalanced portfolio,
which is the shape the published result has.

The reference price is the average of the log price 4.5 to 5.5 years back, so a
single stale print five years ago cannot decide a position today. Positions are
inverse-volatility weighted, long the cheapest third, short the dearest third,
held one month, and charged the same futures costs at the same 2x stress as
every other entry.

Value is the OPPOSITE of momentum at this horizon by construction. If both are
real they should be weakly or negatively correlated, which is the entire reason
to run them together.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "research"))
sys.path.append(str(ROOT / "scripts"))

from backtest.costs import CostModel  # noqa: E402
from core.contracts import FULL_UNIVERSE  # noqa: E402
from futures_gauntlet import load_universe  # noqa: E402
from ml.stats import sharpe  # noqa: E402

LOOKBACK_Y = 5.0
BAND_Y = 0.5      # average the reference over 4.5-5.5 years back
VOL_WINDOW = 63   # days, for inverse-volatility weighting
TOP_FRACTION = 1 / 3


def month_end_panel(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One column per market, one row per month end, of back-adjusted close."""
    cols = {}
    for name, df in bars.items():
        s = df.set_index(pd.to_datetime(df["ts"], utc=True))["close"].astype(float)
        cols[name] = s.resample("ME").last()
    return pd.DataFrame(cols).sort_index()


def value_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """Log reference price minus log price now. Higher means cheaper.

    A back-adjusted futures series can go negative far back in history, which
    makes a log undefined. Those markets simply have no value score for those
    dates rather than a fabricated one.
    """
    lo, hi = int((LOOKBACK_Y - BAND_Y) * 12), int((LOOKBACK_Y + BAND_Y) * 12)
    safe = panel.where(panel > 0)
    logp = np.log(safe)
    reference = sum(logp.shift(k) for k in range(lo, hi + 1)) / (hi - lo + 1)
    return reference - logp


def build(panel: pd.DataFrame, scores: pd.DataFrame, daily: dict[str, pd.DataFrame],
          costs: CostModel, specs: dict) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Monthly long-cheap / short-dear, inverse-volatility weighted, net of costs."""
    rets = panel.pct_change()
    vol = rets.rolling(VOL_WINDOW // 21, min_periods=3).std()

    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    for dt in panel.index:
        row = scores.loc[dt].dropna()
        row = row[[c for c in row.index if np.isfinite(vol.loc[dt, c]) and vol.loc[dt, c] > 0]]
        if len(row) < 9:  # too few to form thirds worth the name
            continue
        k = max(1, int(len(row) * TOP_FRACTION))
        cheap, dear = row.nlargest(k).index, row.nsmallest(k).index
        inv = 1.0 / vol.loc[dt]
        w = pd.Series(0.0, index=panel.columns)
        w[cheap] = inv[cheap] / inv[cheap].sum() * 0.5
        w[dear] = -inv[dear] / inv[dear].sum() * 0.5
        weights.loc[dt] = w

    held = weights.shift(1).fillna(0.0)          # decided at month end, held next month
    gross = (held * rets).sum(axis=1)
    turnover = (weights - weights.shift(1)).abs().sum(axis=1).fillna(0.0)

    # cost per unit of turnover: the round-trip friction of an average market,
    # expressed as a fraction of notional, at the same 2x stress as everywhere else.
    frictions = []
    for name in panel.columns:
        c = costs.for_symbol(name)
        px = float(panel[name].dropna().iloc[-1]) if panel[name].notna().any() else np.nan
        if np.isfinite(px) and px > 0:
            frictions.append((c.spread + 2 * c.slippage) / abs(px))
    unit_cost = float(np.median(frictions)) if frictions else 0.0
    net = gross - turnover * unit_cost
    return net, turnover, weights


def stats(net: pd.Series, label: str) -> dict:
    r = net.dropna()
    ann = float(sharpe(r.to_numpy()) * np.sqrt(12))
    years = r.groupby(r.index.year).sum()
    dd = float(((1 + r).cumprod().cummax() - (1 + r).cumprod()).div((1 + r).cumprod().cummax()).max())
    return {"label": label, "net_sharpe": ann, "months": int(len(r)),
            "positive_years": int((years > 0).sum()), "n_years": int(len(years)),
            "max_drawdown": dd, "annual_return": float(r.mean() * 12),
            "years": {int(y): float(v) for y, v in years.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", type=int, default=2011)
    ap.add_argument("--stress", type=float, default=2.0)
    ap.add_argument("--data", default="data/futures")
    args = ap.parse_args()

    names = list(FULL_UNIVERSE)
    bars, specs, trade = load_universe(args.since, Path(args.data), "full", names)
    panel = month_end_panel(bars)
    scores = value_scores(panel)
    costs = CostModel.for_futures(trade).stressed(args.stress)
    net, turnover, weights = build(panel, scores, bars, costs, specs)

    usable = scores.notna().sum(axis=1)
    first = usable[usable >= 9].index.min()
    net = net[net.index >= first]

    s = stats(net, "value_xs")
    print(f"\nVALUE, CROSS-SECTIONAL - {len(names)} markets, {args.since}-{date.today().year}, "
          f"costs x{args.stress:g}")
    print(f"  first month with enough history: {first:%Y-%m}")
    print(f"  net Sharpe {s['net_sharpe']:.2f}   annual {s['annual_return']:+.2%}   "
          f"max DD {s['max_drawdown']:.1%}")
    print(f"  positive years {s['positive_years']}/{s['n_years']}   "
          f"average turnover {turnover.mean():.2f} of notional a month")
    print("  years: " + ", ".join(f"{y}:{v:+.1%}" for y, v in s["years"].items()))

    rows = [
        ("1. net Sharpe >= 0.40 at 2x costs", s["net_sharpe"] >= 0.40, f"{s['net_sharpe']:.2f}"),
        ("2. positive in >= 70% of years", s["positive_years"] >= 0.7 * s["n_years"],
         f"{s['positive_years']}/{s['n_years']}"),
        ("5. last five years net Sharpe > 0",
         float(sharpe(net[net.index >= net.index.max() - pd.DateOffset(years=5)].to_numpy()) * np.sqrt(12)) > 0,
         f"{float(sharpe(net[net.index >= net.index.max() - pd.DateOffset(years=5)].to_numpy()) * np.sqrt(12)):.2f}"),
    ]
    print(f"\nVERDICT, entry 015a")
    print("=" * 70)
    for label, ok, detail in rows:
        print(f"  {'PASS' if ok else 'FAIL':<5} {label:<48} {detail}")
    passed = all(ok for _, ok, _ in rows)
    print(f"\n  {'PASSED' if passed else 'FAILED - value is dead in this form'}\n")

    Path("state").mkdir(exist_ok=True)
    (Path("state") / "gauntlet_015a.json").write_text(json.dumps(
        {**s, "turnover": float(turnover.mean()),
         "verdict": [{"test": l, "pass": bool(o), "detail": d} for l, o, d in rows],
         "passed": passed, "monthly": {str(k.date()): float(v) for k, v in net.items()}},
        indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
