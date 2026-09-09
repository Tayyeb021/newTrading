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
    python scripts/shadow.py --minutes 7200 --quiet      # a trading week, unattended

Unattended runs must outlive the broker: a failed iteration is journalled as
`loop_error`, three in a row trigger a reconnect, and the loop carries on. The
runner's own state file carries the session book across a process restart.
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

RECONNECT_AFTER = 3  # consecutive failed iterations before we assume the link is gone


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

    def reconnect(self) -> None:
        """Rebuild the LIVE link, which is the only half that can break.

        Found 2026-09-09, the morning after the MetaTrader terminal vanished.
        `Runner._heal_feed` had no `reconnect` to call here, so it fell back to
        `connect()` - which is `PaperAdapter.connect`, the paper book, which
        never fails. It bounced the healthy half every fifteen minutes and
        journalled `ok=True, "reconnected"` while every symbol was still
        returning `IPC send failed` and the runner stayed halted for 92 minutes
        after MetaTrader was already back and serving.
        """
        try:
            self._live.disconnect()
        except Exception:  # noqa: BLE001 - already broken; the reconnect is what matters
            pass
        self._live.connect()
        super().connect()

    def tick(self, symbol):
        t = self._live.tick(symbol)
        self.feed_tick(t)  # keep the paper book marked to real prices
        return t

    def bars(self, symbol, timeframe, count, end=None):
        return self._live.bars(symbol, timeframe, count, end)

    def spec(self, symbol):
        return self._live.spec(symbol)


class LiveMTFPullback(MTFPullback):
    """MTFPullback with its higher-timeframe bias refreshed from the live feed.

    Found 2026-09-08, after the shadow week produced 13,283 heartbeats and not a
    single decision in four days. `MTFPullback.prepare` builds its bias from
    `self.bias_frames`; with that dict empty it sets `bias = 0`, and
    `evaluate` returns FLAT on `bias == 0` before reading anything else. So a
    strategy constructed without bias frames cannot signal - ever, on any bar,
    in any market.

    Every backtest script passes `bias_frames=load_bias_frames(...)` and every
    test passes them explicitly. `scripts/shadow.py` - the only caller that runs
    live - passed none. The rule was validated in one configuration and deployed
    in another that is structurally incapable of trading, and nothing failed:
    the process ran, the feed was healthy, the journal filled with heartbeats.

    The runner fetches one timeframe per leg (`live/runner.py:_intent`), so the
    higher frames have to be pulled here, and pulled on every bar rather than
    snapshotted at startup - a bias frozen at Monday's open is a different bug
    with the same shape.
    """

    def __init__(self, adapter, symbol: str, bars_per_frame: int = 400, **kw) -> None:
        super().__init__(**kw)
        self._adapter = adapter
        self._symbol = symbol
        self._bars_per_frame = bars_per_frame

    def prepare(self, df):
        import pandas as pd

        frames = {}
        for tf in tuple(self.bias_timeframes) + ((self.stop_timeframe,) if self.stop_timeframe else ()):
            if tf in frames:
                continue
            try:
                bars = self._adapter.bars(self._symbol, tf, count=self._bars_per_frame)
            except Exception as exc:  # noqa: BLE001
                # A missing higher timeframe means no bias, which means no
                # trades. Say so rather than trading blind or silently flat.
                print(f"  [bias] {self._symbol} {tf}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if bars:
                frames[tf] = pd.DataFrame([{
                    "ts": b.ts, "open": b.open, "high": b.high,
                    "low": b.low, "close": b.close, "volume": b.volume} for b in bars])
        self.bias_frames = frames
        return super().prepare(df)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%m-%d %H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--until", default=None, metavar="ISO8601",
                    help="absolute UTC end time, e.g. 2026-09-11T21:00Z; overrides --minutes and makes "
                         "a restarted run end at the same moment as the original")
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--profile", default="challenge")
    ap.add_argument("--quiet", action="store_true",
                    help="print only when positions/halt change, plus an hourly line (unattended runs)")
    args = ap.parse_args()

    if args.until:
        end = datetime.fromisoformat(args.until.replace("Z", "+00:00"))
        end = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end.astimezone(timezone.utc)
        args.minutes = (end - datetime.now(timezone.utc)).total_seconds() / 60
        if args.minutes <= 0:
            print(f"deadline {end:%Y-%m-%d %H:%M} UTC has passed; nothing to do", flush=True)
            return 0

    from execution.mt5_adapter import MT5Adapter

    inst = InstrumentConfig.load()
    symbols = args.symbols or inst.active or inst.symbols
    profile = RiskProfile.load(args.profile)

    live = MT5Adapter(aliases=inst.aliases)
    live.connect()
    specs = {s: live.spec(s) for s in symbols}
    account = live.account()

    print(f"\nSHADOW MODE - live data, paper execution   ({_stamp()} UTC)")
    print(f"  account   : {account.equity:,.2f} {account.currency} (real, read-only)")
    print(f"  clock     : {live.clock_status.value if live.clock_status else 'unknown'}")
    print(f"  symbols   : {', '.join(symbols)} on {args.timeframe}")
    print(f"  profile   : {profile.name}, risk {profile.risk_per_trade:.2%}/trade")
    print(f"  duration  : {args.minutes:g} min, polling every {args.poll:g}s"
          f"{', quiet' if args.quiet else ''}\n", flush=True)

    shadow = ShadowAdapter(live, specs)
    engine = build_engine(profile, account.equity, specs)
    runner = Runner(
        adapter=shadow, risk=engine,
        strategies={s: LiveMTFPullback(shadow, s, execution_timeframe=args.timeframe,
                                       bias_timeframes=("H4", "H1")) for s in symbols},
        specs=specs, timeframe=args.timeframe, poll_seconds=args.poll,
        state=StateStore("state/shadow_session.json"),
        journal=Journal("state/shadow_journal.jsonl"),
        kill=KillFile("state/SHADOW_KILL"),
    )

    for note in runner.start():
        print(f"  {note}")
    print(flush=True)

    deadline = time.time() + args.minutes * 60
    ticks = failures = errors = 0
    last_shown: tuple | None = None
    next_hourly = time.time() + 3600
    try:
        while time.time() < deadline:
            ticks += 1
            try:
                runner.tick()
            except Exception as exc:  # noqa: BLE001 - a week-long run must outlive a broker hiccup
                failures += 1
                errors += 1
                runner.journal.write("loop_error", error=f"{type(exc).__name__}: {exc}", consecutive=failures)
                print(f"  [{_stamp()}] iteration failed ({failures} in a row): {type(exc).__name__}: {exc}", flush=True)
                if failures >= RECONNECT_AFTER:
                    try:
                        live.disconnect()
                        live.connect()
                        runner.journal.write("reconnect", ok=True, clock=str(live.clock_status))
                        print(f"  [{_stamp()}] reconnected to MT5 ({live.clock_status})", flush=True)
                        failures = 0
                    except Exception as exc2:  # noqa: BLE001
                        runner.journal.write("reconnect", ok=False, error=str(exc2))
                        print(f"  [{_stamp()}] reconnect failed: {exc2}", flush=True)
                time.sleep(min(60.0, args.poll * max(1, failures)))
                continue
            failures = 0

            acct = shadow.account()
            shown = (len(shadow.positions()), runner.halted)
            hourly = time.time() >= next_hourly
            if not args.quiet or shown != last_shown or hourly:
                print(f"  [{_stamp()}] iter {ticks:<6} equity {acct.equity:>12,.2f}  "
                      f"positions {shown[0]}  halted={shown[1]}  errors={errors}", flush=True)
                last_shown = shown
                if hourly:
                    next_hourly = time.time() + 3600
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        runner.shutdown()
        live.disconnect()

    print(f"\n  {ticks} iterations, {errors} failed")
    print(f"  {runner.journal.summary()}")
    print(f"\n  No order reached the broker. Journal: {runner.journal.path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
