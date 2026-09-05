"""Run a book of sleeves through one equity curve and one risk engine.

    python scripts/backtest_portfolio.py
    python scripts/backtest_portfolio.py --since 2015 --symbols EURUSD XAUUSD US500

The default book is deliberately a trap: two momentum sleeves with different
lookbacks. They look like two strategies and they are one. The report should
say so -- a sleeve correlation above 0.6 and a diversification ratio near 1.0.
If it does, the layer is doing the job it exists for: telling you when the
diversification you are counting on is not there.

Specs and spreads come from config/broker.json, so costs are this account's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.portfolio import PortfolioBacktester, portfolio_report  # noqa: E402
from core.config import RiskProfile  # noqa: E402
from core.sleeve import Sleeve  # noqa: E402
from data.store import BarStore  # noqa: E402
from risk.build import build_engine  # noqa: E402
from strategies.trend import TrendFollowing  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["EURUSD", "XAUUSD", "US500"])
    ap.add_argument("--since", type=int, default=2015)
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--profile", default="challenge")
    ap.add_argument("--reset-on-halt", action="store_true", default=True)
    # The profile's concurrent-position cap was written for one strategy. A book
    # of S sleeves over K symbols can want S*K positions, and a cap of 3 starves
    # the allocator before it does anything: the first run rejected 4,603 signals
    # on max_positions alone. Size the cap for the book.
    ap.add_argument("--max-positions", type=int, default=None,
                    help="override the profile's concurrent-position cap for this book")
    args = ap.parse_args()

    sys.argv = ["x"]
    from scripts.backtest_mtf import load_broker

    specs, costs, _ = load_broker()
    store = BarStore("data/bars")
    profile = RiskProfile.load(args.profile)
    if args.max_positions is not None:
        from dataclasses import replace
        profile = replace(profile, max_concurrent_positions=args.max_positions)
    symbols = tuple(s for s in args.symbols if s in specs)

    sleeves = [
        Sleeve("trend60", lambda sym: TrendFollowing(lookback=60, ema_period=60), symbols, weight=1.0),
        Sleeve("trend250", lambda sym: TrendFollowing(lookback=250, ema_period=100), symbols, weight=1.0),
    ]

    bars = {}
    for sl in sleeves:
        for sym in sl.symbols:
            df = store.read(sym, sl.timeframe)
            df = df[df["ts"].dt.year >= args.since].reset_index(drop=True)
            if df.empty:
                print(f"no {sl.timeframe} data for {sym}")
                return 1
            bars[(sl.name, sym)] = df

    engine = build_engine(profile, args.equity, {s: specs[s] for s in symbols}, sleeves)
    print(f"\nbook: {[s.name for s in sleeves]} on {list(symbols)} since {args.since}")
    print(f"open-risk budget {profile.max_open_risk:.1%} split "
          f"{ {s.name: f'{w:.0%}' for s, w in zip(sleeves, [1/len(sleeves)]*len(sleeves))} }")
    print(f"limits: {[type(l).__name__ for l in engine.limits]}")

    result = PortfolioBacktester(sleeves, {s: specs[s] for s in symbols}, engine, costs,
                                 starting_equity=args.equity,
                                 reset_on_halt=args.reset_on_halt).run(bars)
    print(portfolio_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
