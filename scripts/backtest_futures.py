"""Backtest on futures: per-expiry data -> continuous series -> the portfolio engine.

    python scripts/backtest_futures.py --synthetic                          # proves the pipeline, no data needed
    python scripts/backtest_futures.py --universe full --equity 2000000 --size-as full   # pure signal test:
                                                                            #   granularity never binds
    python scripts/backtest_futures.py --universe full --equity 250000      # what a real book can hold
    python scripts/backtest_futures.py --universe full --profile challenge  # what the prop rules cost it
    python scripts/backtest_futures.py --roots ES GC ZN --lookbacks 20 60 120

Two questions, two runs. "Does the signal exist" is asked at an equity so large
that one contract is always within the risk limit; otherwise daily-bar stops on
bonds and grains exceed 0.5% of a small book and the risk engine refuses most of
the signals, which measures granularity, not edge. "What can I hold" is asked at
the real equity, and `scripts/capital_ladder.py` explains the gap.

Per-expiry files come from `scripts/download_databento.py` and live under
data/futures/<ROOT>/<YYYYMM>.parquet, keyed by the full-size DATA root. Each
root is stitched into a back-adjusted continuous series (`data/continuous.py`),
the rolls are logged, and the result runs through the SAME portfolio backtester
and risk engine as everything else, with a futures cost model: commission per
side, spread in ticks, and no financing.

Data and sizing are separate on purpose. History is read from the full-size
contract (longest record, same price to a tick); positions are sized and costed
with the contract a small account can actually trade -- the micro where one
exists (`--size-as micro`, the default). `--size-as full` sizes with the big
contract, which is what a large account would do.

`--lookbacks` with several values builds one sleeve per speed. Combining speeds
is the standard way to stop a trend book's result depending on one parameter.

`--synthetic` builds a plausible set of expiries with a known term structure
and runs the whole path. It proves the machinery. It proves nothing about
markets, and the report says so.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.costs import CostModel  # noqa: E402
from backtest.portfolio import PortfolioBacktester, portfolio_report  # noqa: E402
from core.config import RiskProfile  # noqa: E402
from core.contracts import ALL_ROOTS, FULL_UNIVERSE, MICRO_UNIVERSE, FuturesRoot, data_root, tradeable  # noqa: E402
from core.sleeve import Sleeve  # noqa: E402
from data.continuous import annual_roll_drag, roll_cost_cash, stitch  # noqa: E402
from risk.build import build_engine  # noqa: E402
from strategies.carry import Carry  # noqa: E402
from strategies.trend import TrendFollowing  # noqa: E402

# Price levels for the synthetic run only, in each contract's quote unit.
SYNTHETIC_LEVEL = {
    "ES": 5000.0, "NQ": 18000.0, "YM": 40000.0, "RTY": 2200.0,
    "ZT": 103.0, "ZF": 108.0, "ZN": 112.0, "ZB": 120.0, "UB": 125.0,
    "6E": 1.08, "6J": 0.0067, "6B": 1.27, "6A": 0.66, "6C": 0.73, "6S": 1.12, "6N": 0.60,
    "GC": 2400.0, "SI": 28.0, "HG": 4.2, "PL": 950.0,
    "CL": 75.0, "NG": 2.8, "RB": 2.3, "HO": 2.4,
    "ZC": 450.0, "ZS": 1050.0, "ZW": 580.0, "KE": 600.0, "ZM": 320.0, "ZL": 45.0,
    "LE": 185.0, "HE": 90.0, "GF": 260.0,
}


def synthetic_expiries(root: FuturesRoot, start: date, end: date, level: float, seed: int = 3):
    """Expiries that overlap, each priced above the last (contango), on a shared
    random walk so returns are continuous across the stitch."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    walk = level * np.exp(np.cumsum(rng.normal(0, 0.008, len(days))))
    base = pd.Series(walk, index=days)
    out = {}
    for w in root.schedule(start, end):
        lo = w.active_from - timedelta(days=120)
        hi = w.last_trade
        seg = base.loc[pd.Timestamp(lo):pd.Timestamp(hi)]
        premium = level * 0.002 * (w.month / 12)  # small, monotone term premium
        px = seg.to_numpy() + premium
        out[(w.year, w.month)] = pd.DataFrame({
            "ts": pd.to_datetime(seg.index, utc=True), "open": px, "high": px * 1.004,
            "low": px * 0.996, "close": px, "volume": 5000.0,
        })
    return out


def load_expiries(name: str, folder: Path):
    """Per-expiry frames for a root, from its own folder or its data root's."""
    for candidate in (name, data_root(name).root):
        files = sorted((folder / candidate).glob("*.parquet"))
        if files:
            return {(int(f.stem[:4]), int(f.stem[4:6])): pd.read_parquet(f) for f in files}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=None)
    ap.add_argument("--universe", choices=["micro", "full"], default=None)
    ap.add_argument("--size-as", choices=["micro", "full"], default="micro")
    ap.add_argument("--lookbacks", nargs="+", type=int, default=[60])
    ap.add_argument("--sleeves", nargs="+", choices=["trend", "carry"], default=["trend"],
                    help="trend (007/008) and/or carry (009)")
    ap.add_argument("--continuous", action="store_true",
                    help="trend sleeves resize toward a forecast-scaled target (008)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data", default="data/futures")
    ap.add_argument("--since", type=int, default=2011)
    ap.add_argument("--equity", type=float, default=25_000.0)
    ap.add_argument("--profile", default="research")
    ap.add_argument("--max-positions", type=int, default=None, help="override the profile's cap")
    args = ap.parse_args()

    if args.roots and args.universe:
        print("give --roots or --universe, not both"); return 1
    names = args.roots or list(FULL_UNIVERSE if args.universe == "full" else ["MES", "MGC"])
    unknown = [n for n in names if n not in ALL_ROOTS]
    if unknown:
        print(f"unknown roots {unknown}; known: {', '.join(sorted(ALL_ROOTS))}"); return 1

    start, end = date(args.since, 1, 1), date.today()
    research = {n: data_root(n) for n in names}                       # where the prices come from
    trade = {n: (tradeable(n) if args.size_as == "micro" else research[n]) for n in names}  # what gets sized

    bars, specs, roll_log = {}, {}, {}
    for name in names:
        droot, troot = research[name], trade[name]
        if args.synthetic:
            exp = synthetic_expiries(droot, start, end, SYNTHETIC_LEVEL[droot.root])
        else:
            exp = load_expiries(name, Path(args.data))
        if not exp:
            print(f"{name}: no expiry data under {args.data}/{droot.root}. Run scripts/download_databento.py, or use --synthetic.")
            return 1
        cont, rolls = stitch(droot, exp, start=start, end=end)
        specs[name] = troot.to_spec(name)
        bars[name] = cont
        roll_log[name] = rolls
        yrs = max((end - start).days / 365.25, 1e-9)
        print(f"{name:<4} data {droot.root:<3} sized as {troot.root:<4} {len(cont):>6,} bars, {len(exp):>3} expiries, "
              f"{len(rolls):>3} rolls, term drag {annual_roll_drag(rolls, droot, yrs):+.3f}/yr, "
              f"roll friction {roll_cost_cash(troot, 1):.2f}/contract")

    sleeves = []
    if "trend" in args.sleeves:
        prefix = "ctrend" if args.continuous else "trend"
        for lb in args.lookbacks:
            sleeves.append(Sleeve(
                f"{prefix}{lb}",
                (lambda s, lb=lb: TrendFollowing(lookback=lb, ema_period=lb, continuous=args.continuous)),
                tuple(names), timeframe="D1",
            ))
    if "carry" in args.sleeves:
        sleeves.append(Sleeve("carry", lambda s: Carry(), tuple(names), timeframe="D1"))
    costs = CostModel.for_futures(trade)
    profile = RiskProfile.load(args.profile)
    if args.max_positions is not None:
        profile = replace(profile, max_concurrent_positions=args.max_positions)
    engine = build_engine(profile, args.equity, specs, sleeves)

    print(f"\nprofile {profile.name}: {profile.risk_per_trade:.2%}/trade, {profile.max_concurrent_positions} positions, "
          f"open risk {profile.max_open_risk:.0%}; equity {args.equity:,.0f}; sleeves {[s.name for s in sleeves]}")
    result = PortfolioBacktester(sleeves, specs, engine, costs, starting_equity=args.equity,
                                 reset_on_halt=True).run({(s.name, n): bars[n] for s in sleeves for n in names})
    print(portfolio_report(result))
    if args.synthetic:
        print("\n  SYNTHETIC: the pipeline ran end to end. The numbers describe a random walk, not a market.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
