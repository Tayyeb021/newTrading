"""Backtest on futures: per-expiry data -> continuous series -> the portfolio engine.

    python scripts/backtest_futures.py --synthetic            # proves the pipeline, no data needed
    python scripts/backtest_futures.py --roots MES MGC        # from data/futures/<ROOT>/<YYYYMM>.parquet

Per-expiry files come from `scripts/download_databento.py`. Each root is
stitched into a back-adjusted continuous series (`data/continuous.py`), the
rolls are logged, and the result runs through the SAME portfolio backtester and
risk engine as everything else, with a futures cost model: commission per side,
spread in ticks, and no financing.

`--synthetic` builds a plausible set of expiries with a known term structure
and runs the whole path. It proves the machinery. It proves nothing about
markets, and the report says so.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.costs import CostModel  # noqa: E402
from backtest.portfolio import PortfolioBacktester, portfolio_report  # noqa: E402
from core.config import RiskProfile  # noqa: E402
from core.contracts import MICRO_UNIVERSE, FuturesRoot  # noqa: E402
from core.sleeve import Sleeve  # noqa: E402
from data.continuous import annual_roll_drag, roll_cost_cash, stitch  # noqa: E402
from risk.build import build_engine  # noqa: E402
from strategies.trend import TrendFollowing  # noqa: E402


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


def load_expiries(root: str, folder: Path):
    out = {}
    for f in sorted((folder / root).glob("*.parquet")):
        ym = f.stem
        out[(int(ym[:4]), int(ym[4:6]))] = pd.read_parquet(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=["MES", "MGC"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data", default="data/futures")
    ap.add_argument("--since", type=int, default=2018)
    ap.add_argument("--equity", type=float, default=25_000.0)
    ap.add_argument("--max-positions", type=int, default=6)
    args = ap.parse_args()

    roots = {r: MICRO_UNIVERSE[r] for r in args.roots}
    start, end = date(args.since, 1, 1), date.today()
    levels = {"MES": 4000.0, "MNQ": 14000.0, "MGC": 1800.0, "M6E": 1.10, "MCL": 70.0, "ZN": 120.0}

    bars, specs, roll_log = {}, {}, {}
    for name, root in roots.items():
        exp = synthetic_expiries(root, start, end, levels[name]) if args.synthetic else load_expiries(name, Path(args.data))
        if not exp:
            print(f"{name}: no expiry data in {args.data}/{name}. Run scripts/download_databento.py, or use --synthetic.")
            return 1
        cont, rolls = stitch(root, exp, start=start, end=end)
        specs[name] = root.to_spec(name)
        bars[name] = cont
        roll_log[name] = rolls
        yrs = (end - start).days / 365.25
        print(f"{name}: {len(cont):,} continuous bars from {len(exp)} expiries, {len(rolls)} rolls, "
              f"term drag {annual_roll_drag(rolls, root, yrs):+.2f}/yr in price, "
              f"roll friction {roll_cost_cash(root, 1):.2f}/contract each")

    sleeves = [Sleeve("trend60", lambda s: TrendFollowing(lookback=60, ema_period=60), tuple(roots), timeframe="D1")]
    costs = CostModel.for_futures(roots)
    profile = RiskProfile.load("challenge")
    from dataclasses import replace
    profile = replace(profile, max_concurrent_positions=args.max_positions)
    engine = build_engine(profile, args.equity, specs, sleeves)

    result = PortfolioBacktester(sleeves, specs, engine, costs, starting_equity=args.equity,
                                 reset_on_halt=True).run({("trend60", r): bars[r] for r in roots})
    print(portfolio_report(result))
    if args.synthetic:
        print("\n  SYNTHETIC: the pipeline ran end to end. The numbers describe a random walk, not a market.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
