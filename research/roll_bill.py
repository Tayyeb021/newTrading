"""What the backtests never paid to roll a contract.

    python research/roll_bill.py

`backtest/portfolio.py` trades a *continuous* series. A continuous series has no
expiries in it, so a position held for two years costs nothing to carry - and in
the market it would have been closed and reopened eight times, crossing a spread
and paying commission each time. `data/continuous.py:154` has the function that
prices exactly this, `roll_cost_cash`; it is called by the live shadow adapter
and by one print statement in a script, and by no backtest.

This prices the missing bill from the trade record: for every fill, the rolls of
its own market that fall between entry and exit, at its own size.

Nothing is being selected here and no threshold applied. This is a measurement
of an error in the harness, and every entry from 007 onward is affected by it in
the same direction: their costs are too low, so their results are too good.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from backtest_futures import load_expiries  # noqa: E402
from core.contracts import FULL_UNIVERSE, data_root  # noqa: E402
from data.continuous import roll_cost_cash, stitch  # noqa: E402


def roll_dates(root_name: str, folder: Path, since: int) -> list[pd.Timestamp]:
    exp = load_expiries(root_name, folder)
    if not exp:
        return []
    _, rolls = stitch(data_root(root_name), exp, start=date(since, 1, 1), end=date.today())
    # Roll.on is the roll date. Reaching for a `date` or `ts` attribute that
    # does not exist returned an empty calendar and a $0 bill, which read as
    # "there is no problem" - the exact failure this file exists to catch.
    return sorted(pd.Timestamp(r.on, tz="UTC") for r in rolls)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trades", default="data/trades_010.parquet")
    ap.add_argument("--result", default="state/gauntlet_010_trades.json")
    ap.add_argument("--data", default="data/futures")
    ap.add_argument("--since", type=int, default=2011)
    ap.add_argument("--spread-ticks", type=float, default=1.0,
                    help="ticks crossed per leg of a roll; 1.0 is roll_cost_cash's own default")
    ap.add_argument("--stress", type=float, default=2.0,
                    help="the cost stress the entry declared, applied to the roll bill too")
    args = ap.parse_args()

    tpath = Path(args.trades)
    if not tpath.exists():
        raise SystemExit(f"no trade dump at {tpath}; run the gauntlet with --dump-trades {tpath}")
    trades = pd.read_parquet(tpath)
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"], utc=True)

    calendars = {s: roll_dates(s, Path(args.data), args.since) for s in sorted(trades["symbol"].unique())}

    rows = []
    for symbol, grp in trades.groupby("symbol"):
        root = FULL_UNIVERSE[symbol]
        cal = calendars.get(symbol, [])
        bill = crossings = 0.0
        for t in grp.itertuples():
            n = sum(1 for d in cal if t.entry_ts < d <= t.exit_ts)
            if not n:
                continue
            crossings += n
            bill += n * roll_cost_cash(root, abs(t.volume), args.spread_ticks) * args.stress
        rows.append({"symbol": symbol, "bucket": root.bucket, "trades": len(grp),
                     "rolls_crossed": int(crossings), "unpaid_roll_cost": bill,
                     "net_pnl": float(grp["net_pnl"].sum()),
                     "charged_costs": float(grp["costs"].sum())})

    df = pd.DataFrame(rows).sort_values("unpaid_roll_cost", ascending=False)
    total_bill = df["unpaid_roll_cost"].sum()
    total_net = df["net_pnl"].sum()
    total_charged = df["charged_costs"].sum()

    print(f"\nTHE UNPAID ROLL BILL - {len(trades):,} fills, {args.stress:g}x cost stress, "
          f"{args.spread_ticks:g} tick per leg")
    print("=" * 84)
    print(f"  {'market':<8}{'sector':<12}{'rolls':>8}{'unpaid':>13}{'charged':>13}"
          f"{'net P&L':>13}{'unpaid/net':>11}")
    print("  " + "-" * 82)
    for r in df.head(15).itertuples():
        share = r.unpaid_roll_cost / r.net_pnl if r.net_pnl else float("nan")
        print(f"  {r.symbol:<8}{r.bucket:<12}{r.rolls_crossed:>8}"
              f"{r.unpaid_roll_cost / 1e6:>12.2f}M{r.charged_costs / 1e6:>12.2f}M"
              f"{r.net_pnl / 1e6:>12.2f}M{share:>10.0%}")

    print(f"\n  unpaid roll cost   {total_bill / 1e6:>10.2f}M")
    print(f"  costs actually charged {total_charged / 1e6:>6.2f}M   "
          f"(the bill is {total_bill / total_charged:.1f}x what the run charged)")
    print(f"  book net P&L       {total_net / 1e6:>10.2f}M")
    print(f"  net after the bill {(total_net - total_bill) / 1e6:>10.2f}M   "
          f"({total_bill / total_net:.0%} of net)")

    result = Path(args.result)
    if result.exists():
        m = json.loads(result.read_text(encoding="utf-8")).get("book", {})
        if m:
            print(f"\n  reported net Sharpe {m['net_sharpe']:.3f} was computed without any of this.")

    (ROOT / "state").mkdir(exist_ok=True)
    (ROOT / "state" / "roll_bill.json").write_text(json.dumps({
        "total_unpaid": total_bill, "total_charged": total_charged, "net_pnl": total_net,
        "spread_ticks": args.spread_ticks, "stress": args.stress,
        "by_market": df.to_dict("records")}, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
