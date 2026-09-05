"""Shadow mode: the real system on live broker data, with no real orders.

Everything runs exactly as it would live -- the real `Runner`, the real
`RiskEngine` loaded from the same YAML, the real strategy, the real OMS with its
idempotency and retry logic, the real journal. The only substitution is the
execution adapter: orders go to `PaperAdapter` instead of the broker.

This is what closes the "does the live path actually work" question without
sending an order. It exercises the wiring that a backtest never touches: live
tick freshness, bar-close detection, the execution worker thread, state
persistence, and every account-level limit against real account numbers.

    python scripts/shadow.py --minutes 5
    python scripts/shadow.py --symbols EURUSD XAUUSD --minutes 30
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import InstrumentConfig, RiskProfile  # noqa: E402
from execution.paper import PaperAdapter, PaperConfig  # noqa: E402
from live.runner import Runner  # noqa: E402
from live.state import StateStore  # noqa: E402
from ops.journal import Journal  # noqa: E402
from risk.build import build_engine  # noqa: E402
from risk.killswitch import KillFile  # noqa: E402
from strategies.mtf_pullback import MTFPullback  # noqa: E402


class ShadowAdapter(PaperAdapter):
    """Live broker for reads, paper broker for writes.

    Prices, bars, specs and the account come from MT5. Orders never leave the
    process. The split is the entire point: everything upstream of `submit` is
    the production code path, exercised against real market state.
    """

    name = "shadow"

    def __init__(self, live, specs, config=None):
        super().__init__(specs, config or PaperConfig(starting_balance=live.account().equity))
        self._live = live
        self.rejected_live_writes = 0

    def connect(self) -> None:
        super().connect()

    def tick(self, symbol):
        t = self._live.tick(symbol)
        self.feed_tick(t)  # keep the paper book marked to real prices
        return t

    def bars(self, symbol, timeframe, count, end=None):
        return self._live.bars(symbol, timeframe, count, end)

    def spec(self, symbol):
        return self._live.spec(symbol)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--profile", default="challenge")
    args = ap.parse_args()

    from execution.mt5_adapter import MT5Adapter

    inst = InstrumentConfig.load()
    symbols = args.symbols or inst.active or inst.symbols
    profile = RiskProfile.load(args.profile)

    live = MT5Adapter(aliases=inst.aliases)
    live.connect()
    specs = {s: live.spec(s) for s in symbols}
    account = live.account()

    print(f"\nSHADOW MODE - live data, paper execution")
    print(f"  account   : {account.equity:,.2f} {account.currency} (real, read-only)")
    print(f"  clock     : {live.clock_status.value if live.clock_status else 'unknown'}")
    print(f"  symbols   : {', '.join(symbols)} on {args.timeframe}")
    print(f"  profile   : {profile.name}, risk {profile.risk_per_trade:.2%}/trade")
    print(f"  duration  : {args.minutes:g} min, polling every {args.poll:g}s\n")

    shadow = ShadowAdapter(live, specs)
    engine = build_engine(profile, account.equity, specs)
    runner = Runner(
        adapter=shadow, risk=engine,
        strategies={s: MTFPullback(execution_timeframe=args.timeframe,
                                   bias_timeframes=("H4", "H1")) for s in symbols},
        specs=specs, timeframe=args.timeframe, poll_seconds=args.poll,
        state=StateStore("state/shadow_session.json"),
        journal=Journal("state/shadow_journal.jsonl"),
        kill=KillFile("state/SHADOW_KILL"),
    )

    for note in runner.start():
        print(f"  {note}")
    print()

    deadline = time.time() + args.minutes * 60
    ticks = 0
    try:
        while time.time() < deadline:
            ticks += 1
            runner.tick()
            state_now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            acct = shadow.account()
            print(f"  [{state_now}] iter {ticks:<3} equity {acct.equity:>12,.2f}  "
                  f"positions {len(shadow.positions())}  "
                  f"halted={runner.halted}")
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        runner.shutdown()
        live.disconnect()

    print(f"\n  {ticks} iterations")
    print(f"  {runner.journal.summary()}")
    print(f"\n  No order reached the broker. Journal: {runner.journal.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
