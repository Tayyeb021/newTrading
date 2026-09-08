"""Which instruments does the book actually work on?

    python research/per_market.py

A sector total answers nothing. `rates +$15.0M` is compatible with five markets
each earning three, and with one market earning fifteen while four lose. The
difference decides whether the forward record is trading the edge or trading
around it.

Reads the per-market breakdown the gauntlet now keeps, and checks it against
`state/forward_manifest.json` — the 13 micros actually being traded forward.
Nothing is selected here and no threshold is applied: this is a description of
a book already declared dead against its own bar, not a search for a better
subset. Picking the winners out of this table would be exactly the post-hoc
selection the log has refused five times.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.contracts import CORE_UNIVERSE, FULL_UNIVERSE, PARENT_OF  # noqa: E402

STATE = ROOT / "state"


def load(name: str, key: str) -> dict | None:
    path = STATE / name
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return d.get(key)


def traded_parents() -> dict[str, str]:
    """Parent root -> the micro the forward record actually sends."""
    manifest = STATE / "forward_manifest.json"
    if not manifest.exists():
        return {}
    out = {}
    for m in json.loads(manifest.read_text(encoding="utf-8"))["universe"]:
        out[PARENT_OF.get(m, m)] = m
    return out


def table(title: str, markets: dict, traded: dict[str, str]) -> dict:
    rows = sorted(markets.items(), key=lambda kv: kv[1]["net_pnl"], reverse=True)
    total = sum(v["net_pnl"] for _, v in rows)
    print(f"\n{title}")
    print("=" * 86)
    print(f"  {'market':<8}{'sector':<12}{'net P&L':>12}{'share':>9}{'cumul':>9}"
          f"{'gross':>12}{'friction':>10}{'trades':>8}  forward")
    print("  " + "-" * 84)
    cumulative = 0.0
    positive = [r for r in rows if r[1]["net_pnl"] > 0]
    pos_total = sum(v["net_pnl"] for _, v in positive)
    for name, v in rows:
        cumulative += v["net_pnl"]
        share = v["net_pnl"] / total if total else 0.0
        mark = traded.get(name, "")
        print(f"  {name:<8}{v['bucket']:<12}{v['net_pnl'] / 1e6:>11.2f}M{share:>8.0%}"
              f"{cumulative / total if total else 0:>8.0%}"
              f"{v['gross_pnl'] / 1e6:>11.2f}M{v['friction'] / 1e6:>9.2f}M"
              f"{v['trades']:>8}  {mark}")

    # How concentrated is it? The share of the total carried by the best few.
    conc = {}
    for k in (1, 3, 5):
        conc[k] = sum(v["net_pnl"] for _, v in rows[:k]) / total if total else 0.0
    print(f"\n  {len(positive)} of {len(rows)} markets positive. "
          f"Top 1 = {conc[1]:.0%} of net, top 3 = {conc[3]:.0%}, top 5 = {conc[5]:.0%}.")
    print(f"  Winners made {pos_total / 1e6:+.1f}M, losers "
          f"{(total - pos_total) / 1e6:+.1f}M, net {total / 1e6:+.1f}M.")

    in_fwd = [r for r in rows if r[0] in traded]
    out_fwd = [r for r in rows if r[0] not in traded]
    fwd_pnl = sum(v["net_pnl"] for _, v in in_fwd)
    print(f"\n  The forward record trades {len(in_fwd)} of these {len(rows)}, carrying "
          f"{fwd_pnl / 1e6:+.1f}M of the {total / 1e6:+.1f}M ({fwd_pnl / total if total else 0:.0%}).")
    if out_fwd:
        worst_missed = max(out_fwd, key=lambda kv: kv[1]["net_pnl"])
        print(f"  Largest earner NOT traded forward: {worst_missed[0]} "
              f"({worst_missed[1]['net_pnl'] / 1e6:+.2f}M, {worst_missed[1]['bucket']})")
    return {"total": total, "rows": dict(rows), "concentration": conc,
            "forward_share": fwd_pnl / total if total else 0.0,
            "positive": len(positive), "n": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="_mkt")
    args = ap.parse_args()

    traded = traded_parents()
    books = [
        ("TREND (tsmom 60/120/250 ensemble)", f"gauntlet_010{args.tag}.json", "book"),
        ("CARRY", f"gauntlet_010c_with_trend{args.tag}.json", "carry"),
        ("TREND + CARRY, the book the forward record runs",
         f"gauntlet_010c_with_trend{args.tag}.json", "trend_plus_carry"),
    ]
    out = {}
    for title, fname, key in books:
        m = load(fname, key)
        if m is None or "markets" not in m:
            print(f"\n{title}: no per-market data in {fname} - "
                  f"rerun the gauntlet with the current code")
            continue
        out[key] = table(title, m["markets"], traded)

    never = [PARENT_OF.get(x, x) for x in
             json.loads((STATE / "forward_manifest.json").read_text(encoding="utf-8"))["universe"]]
    unresearched = [p for p in never if p not in CORE_UNIVERSE]
    if unresearched:
        print(f"\nTRADED FORWARD BUT NEVER RESEARCHED")
        print("=" * 86)
        for p in unresearched:
            b = FULL_UNIVERSE[p].bucket if p in FULL_UNIVERSE else "?"
            print(f"  {p} ({b}) is in the forward record and in none of entries 007-016.")

    (STATE / "per_market.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
