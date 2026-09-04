"""Run a backtest.

    python scripts/backtest.py --symbol EURUSD --timeframe D1
    python scripts/backtest.py --symbol EURUSD --stress 2      # 2x costs
    python scripts/backtest.py --symbol EURUSD --synthetic     # no data needed

`--stress 2` is not optional decoration. It is the cost-sensitivity gate from the
validation gauntlet: if the edge dies when costs double, it was never an edge.
Run it every time, and read the result before the headline one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.costs import CostModel  # noqa: E402
from backtest.engine import Backtester  # noqa: E402
from backtest.metrics import compute, r_histogram, report  # noqa: E402
from core.config import RiskProfile  # noqa: E402
from data.store import BarStore  # noqa: E402
from execution.paper import FIXTURE_SPECS  # noqa: E402
from risk.build import build_engine  # noqa: E402
from strategies.trend import BuyAndHold, TrendFollowing  # noqa: E402

STRATEGIES = {"trend": TrendFollowing, "hold": BuyAndHold}


def synthetic(symbol: str, bars: int = 1500, seed: int = 3) -> "pd.DataFrame":
    """Trending random walk. For exercising the harness, never for conclusions."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    spec = FIXTURE_SPECS[symbol]
    price = {"EURUSD": 1.08, "XAUUSD": 3300.0, "US30": 44000.0, "US500": 6000.0}[symbol]
    vol = price * 0.008

    # A slowly varying drift produces regimes, so a trend strategy has something
    # to find. Real markets are far less generous.
    drift = np.cumsum(rng.normal(0, vol * 0.02, bars))
    steps = rng.normal(0, vol, bars) + np.gradient(drift)
    closes = price + np.cumsum(steps)
    closes = np.maximum(closes, price * 0.2)

    opens = np.concatenate([[closes[0]], closes[:-1]])
    spread = np.abs(rng.normal(0, vol * 0.6, bars))
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread

    ts = pd.date_range("2020-01-01", periods=bars, freq="B", tz="UTC")
    return pd.DataFrame({
        "ts": ts, "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": rng.integers(100, 1000, bars).astype(float),
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--timeframe", default="D1")
    ap.add_argument("--strategy", default="trend", choices=sorted(STRATEGIES))
    ap.add_argument("--profile", default="challenge")
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--stress", type=float, default=1.0, help="cost multiplier")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--root", default="data/bars")
    ap.add_argument("--histogram", action="store_true")
    ap.add_argument("--compare-hold", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        df = synthetic(args.symbol)
        source = "SYNTHETIC - exercises the harness, proves nothing"
    else:
        df = BarStore(args.root).read(args.symbol, args.timeframe)
        source = f"{args.root}/{args.symbol}/{args.timeframe}"
        if df.empty:
            print(f"No data at {source}. Run scripts/download_history.py, or use --synthetic.")
            return 1

    spec = FIXTURE_SPECS[args.symbol]
    profile = RiskProfile.load(args.profile)
    costs = CostModel().stressed(args.stress) if args.stress != 1.0 else CostModel()

    print(f"\ndata     : {source}  ({len(df):,} bars)")
    print(f"profile  : {profile.name}, risk {profile.risk_per_trade:.2%}/trade")
    print(costs.summary({args.symbol: spec}))

    move = float((df["high"] - df["low"]).median())
    ratio = costs.edge_ratio(args.symbol, spec, move)
    flag = "DEAD" if ratio < 3 else "fragile" if ratio < 5 else "workable"
    print(f"\nedge ratio (median bar range / round-trip cost): {ratio:.1f}  -> {flag}")

    runs = [(args.strategy, STRATEGIES[args.strategy]())]
    if args.compare_hold and args.strategy != "hold":
        runs.append(("hold", BuyAndHold()))

    for _, strategy in runs:
        engine = build_engine(profile, args.equity, {args.symbol: spec})
        bt = Backtester(strategy, spec, engine, costs, starting_equity=args.equity)
        result = bt.run(df)
        m = compute(result)
        print(report(result, m))
        if args.histogram:
            print()
            print(r_histogram(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
