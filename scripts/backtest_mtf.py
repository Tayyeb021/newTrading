"""Backtest the multi-timeframe pullback strategy on real broker data.

    python scripts/backtest_mtf.py --symbol EURUSD --exec M15
    python scripts/backtest_mtf.py --symbol XAUUSD --exec M5 --stress 2

Specs and spreads come from config/broker.json (written by snapshot_broker.py),
so the cost model is calibrated to this account rather than to a placeholder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.costs import CostModel, SymbolCosts  # noqa: E402
from backtest.engine import Backtester  # noqa: E402
from backtest.metrics import compute, r_histogram, report  # noqa: E402
from core.config import RiskProfile  # noqa: E402
from core.types import SymbolSpec  # noqa: E402
from data.store import BarStore  # noqa: E402
from risk.build import build_engine  # noqa: E402
from strategies.mtf_pullback import MTFPullback, load_bias_frames  # noqa: E402


def load_broker(path="config/broker.json", require: list[str] | None = None):
    """Load measured specs and spreads. Refuses to fall back to placeholders.

    The bug this guards against: an earlier version built costs only for symbols
    that had a measured spread, and every other symbol silently kept a generic
    placeholder -- while the model still reported calibrated=True. A backtest on
    gold then ran with a fabricated 0.28 spread and zero swap, and looked exactly
    as trustworthy as a real one.

    Now a symbol without a measurement raises. A missing cost model must stop the
    run, not quietly become a plausible number.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    specs = {k: SymbolSpec(**v) for k, v in raw["specs"].items()}

    missing = [k for k in (require or specs) if k not in raw["spreads"]]
    if missing:
        raise ValueError(
            f"no measured spread for {missing} in {path}. Re-run "
            f"scripts/snapshot_broker.py while the market is open. Refusing to "
            f"substitute placeholder costs."
        )

    # Slippage is still ASSUMED, not measured, until verify_roundtrip.py has run
    # against this broker. Spreads are real; slippage is a guess, so the model is
    # not fully calibrated and must not claim to be.
    costs = CostModel(calibrated=False)
    for sym, s in raw["spreads"].items():
        spec = specs[sym]
        costs.costs[sym] = SymbolCosts(
            spread=s["median"],
            # Slippage unmeasured until verify_roundtrip has run; assume half a
            # spread, which is pessimistic but honest about being an assumption.
            slippage=s["median"] * 0.5,
            spread_multiple_at_open=2.0,
            swap_long=spec.swap_long,
            swap_short=spec.swap_short,
        )
    # Drop every symbol we did NOT measure, so `for_symbol` raises instead of
    # returning a leftover default.
    for stale in [k for k in list(costs.costs) if k not in raw["spreads"]]:
        del costs.costs[stale]
    return specs, costs, raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--exec", dest="exec_tf", default="M15", choices=["M5", "M15", "M30"])
    ap.add_argument("--bias", nargs="+", default=["H4", "H1"])
    ap.add_argument("--equity", type=float, default=500_000.0, help="in account currency (cents)")
    ap.add_argument("--stress", type=float, default=1.0)
    ap.add_argument("--profile", default="challenge")
    ap.add_argument("--histogram", action="store_true")
    args = ap.parse_args()

    specs, costs, raw = load_broker()
    if args.stress != 1.0:
        costs = costs.stressed(args.stress)

    store = BarStore("data/bars")
    df = store.read(args.symbol, args.exec_tf)
    if df.empty:
        print(f"no {args.exec_tf} data for {args.symbol}")
        return 1

    bias = load_bias_frames(store, args.symbol, tuple(args.bias))
    if not bias:
        print(f"no bias data for {args.symbol} on {args.bias}")
        return 1

    spec = specs[args.symbol]
    profile = RiskProfile.load(args.profile)

    print(f"\n{args.symbol}  exec={args.exec_tf}  bias={'+'.join(bias)}")
    print(f"  {len(df):,} execution bars  {df['ts'].iloc[0]:%Y-%m-%d} -> {df['ts'].iloc[-1]:%Y-%m-%d}")
    print(f"  equity {args.equity:,.0f} {raw['account_currency']}  "
          f"risk {profile.risk_per_trade:.2%}/trade")
    c = costs.for_symbol(args.symbol)
    print(f"  spread {c.spread / spec.point:.1f} pts, round trip "
          f"{c.round_trip_price() / spec.point:.1f} pts"
          + (f"  [STRESSED x{args.stress:g}]" if args.stress != 1.0 else ""))

    strategy = MTFPullback(
        execution_timeframe=args.exec_tf,
        bias_frames=bias,
        bias_timeframes=tuple(args.bias),
    )
    engine = build_engine(profile, args.equity, {args.symbol: spec})
    result = Backtester(strategy, spec, engine, costs, starting_equity=args.equity).run(df)
    m = compute(result)
    print(report(result, m))
    if args.histogram:
        print()
        print(r_histogram(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
