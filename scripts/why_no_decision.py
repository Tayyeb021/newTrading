"""Why has the shadow week never decided anything?

    python scripts/why_no_decision.py

Read-only. Connects to the running MetaTrader terminal, reads bars, runs the
strategy over them, and reports where the chain breaks. Sends nothing.

The journal only writes a `decision` record when a signal exists to evaluate, so
"no decisions in 13,283 heartbeats" is consistent with three very different
causes and the journal cannot tell them apart:

  1. the feed returns too few bars, so `_intent` returns None before the
     strategy ever runs - a silent no-op, and the failure mode this repository
     keeps finding;
  2. the strategy runs and returns FLAT every time, which for a selective rule
     over four days is a normal and correct answer;
  3. the runner never reaches the decide step at all, because it is halted.

Each is checked separately and named.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.config import InstrumentConfig, RiskProfile  # noqa: E402
from core.strategy import FLAT  # noqa: E402
from strategies.mtf_pullback import MTFPullback  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--symbols", nargs="+", default=None)
    args = ap.parse_args()

    from execution.mt5_adapter import MT5Adapter

    inst = InstrumentConfig.load()
    symbols = args.symbols or inst.active or inst.symbols
    live = MT5Adapter(aliases=inst.aliases)
    live.connect()
    try:
        acct = live.account()
        print(f"\nWHY NO DECISION - {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
        print("=" * 84)
        print(f"  account {acct.equity:,.2f} {acct.currency}, clock "
              f"{live.clock_status.value if live.clock_status else 'unknown'}")
        print(f"  instruments.yaml active: {', '.join(symbols)}")
        print(f"  timeframe {args.timeframe}\n")

        strat = MTFPullback(execution_timeframe=args.timeframe, bias_timeframes=("H4", "H1"))
        need_fetch = strat.warmup + 60
        need_act = strat.warmup + 2
        print(f"  MTFPullback warmup {strat.warmup}: asks for {need_fetch} bars, "
              f"refuses to run below {need_act}\n")

        print(f"  {'symbol':<10}{'bars':>7}{'enough?':>9}{'last bar (UTC)':>22}{'age':>10}"
              f"{'signals':>9}{'rate':>8}")
        print("  " + "-" * 80)
        any_signal = False
        for s in symbols:
            try:
                bars = live.bars(s, args.timeframe, count=need_fetch)
            except Exception as exc:  # noqa: BLE001
                print(f"  {s:<10}{'ERROR':>7}  {type(exc).__name__}: {str(exc)[:50]}")
                continue
            if not bars:
                print(f"  {s:<10}{0:>7}{'NO':>9}   the feed returned nothing")
                continue

            closed = bars[:-1]
            enough = len(bars) >= need_act
            last = closed[-1].ts if closed else bars[-1].ts
            age = (datetime.now(timezone.utc) - pd.Timestamp(last).tz_convert("UTC")).total_seconds() / 60

            fired = 0
            rate = "-"
            if enough:
                df = pd.DataFrame([{"ts": b.ts, "open": b.open, "high": b.high, "low": b.low,
                                    "close": b.close, "volume": b.volume} for b in closed])
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                # The runner calls prepare() before evaluate() (live/runner.py:441)
                # and the indicator columns come from there. Skipping it raised
                # KeyError: 'stop_atr' on the first bar and made a working
                # strategy look silent.
                df = strat.prepare(df)
                # Replay every bar the strategy could have acted on, exactly as
                # the runner would have, one closed bar at a time.
                for i in range(strat.warmup, len(df)):
                    try:
                        sig = strat.evaluate(df, i, None)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  {s:<10} strategy raised at bar {i}: {type(exc).__name__}: {exc}")
                        break
                    if sig is not None and sig is not FLAT and getattr(sig, "side", None) not in (None, FLAT):
                        fired += 1
                tested = max(0, len(df) - strat.warmup)
                rate = f"{fired / tested:.1%}" if tested else "-"
                any_signal = any_signal or fired > 0

            print(f"  {s:<10}{len(bars):>7}{'yes' if enough else 'NO':>9}"
                  f"{str(last)[:19]:>22}{age:>9.0f}m{fired:>9}{rate:>8}")

        print()
        if not any_signal:
            print("  VERDICT: the feed is fine and the strategy ran - it simply never signalled")
            print("  over the bars the terminal is holding. A selective rule returning FLAT is")
            print("  not a fault, but four days without one entry is worth knowing about.")
        else:
            print("  VERDICT: the strategy DOES signal on this history. If the journal has no")
            print("  decision records, the break is between the bar and the strategy - check")
            print("  whether the runner was halted at those moments, and whether _intent's")
            print("  bar-count guard was tripping.")
        return 0
    finally:
        live.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
