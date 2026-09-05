"""Exercise the futures execution path: open, stop, modify, roll, close.

    python scripts/verify_roundtrip_ib.py --fake        # full sequence, no broker
    python scripts/verify_roundtrip_ib.py               # connect to TWS/Gateway, dry run
    python scripts/verify_roundtrip_ib.py --send        # one real paper order - yours to run

With `--fake` the entire sequence runs through the IB test double: this is
what proves the adapter's logic without a terminal. Against a real TWS the
default is a dry run - connect, verify the clock, resolve the front contract,
read specs and a quote, size the order - and stop before sending. `--send`
places one micro contract at minimum size on a PAPER account only; the script
refuses anything whose account id does not start with "DU".
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import RiskProfile  # noqa: E402
from core.contracts import MICRO_UNIVERSE  # noqa: E402
from core.types import OrderRequest, Side  # noqa: E402
from execution.ib_adapter import IBAdapter  # noqa: E402
from risk.sizing import size_position  # noqa: E402


def step(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")
    return ok


def sequence(ad: IBAdapter, symbol: str, send: bool) -> bool:
    ok = True
    root = MICRO_UNIVERSE[symbol]
    ok &= step("clock", True, ad.clock_message)
    ym = ad.front_month(symbol)
    ok &= step("front contract", True, f"{root.code(*ym)}  rolls {root.roll_date(*ym)}  last trade {root.last_trade(*ym)}")
    spec = ad.spec(symbol)
    ok &= step("spec from exchange", spec.tick_value > 0,
               f"tick {spec.tick_size} x {spec.contract_size} = ${spec.tick_value}/tick, ${spec.value_per_price_unit}/point")
    t = ad.tick(symbol)
    ok &= step("quote", t.ask > 0, f"bid {t.bid} ask {t.ask} spread {(t.ask - t.bid) / spec.tick_size:.0f} ticks")
    acct = ad.account()
    ok &= step("account", acct.equity > 0, f"equity {acct.equity:,.2f} {acct.currency}, margin used {acct.margin_used:,.0f}")

    profile = RiskProfile.load("challenge")
    stop_dist = 40 * spec.tick_size  # 10 points on MES; a sane placeholder for a plumbing test
    size = size_position(spec, acct.equity, profile.risk_per_trade, stop_dist)
    ok &= step("sizing", True,
               f"{size.volume:g} contracts risks {size.risk_fraction:.3%}" if size.tradeable else size.reason)

    if not send:
        step("DRY RUN - no order sent", True, f"would BUY 1 {symbol} with stop {t.ask - stop_dist}")
        return ok

    res = ad.submit(OrderRequest(symbol, Side.BUY, 1, stop_loss=t.ask - stop_dist, comment="rt#ib"))
    ok &= step("open 1 contract", res.ok, f"ticket {res.ticket} @ {res.fill_price} slip {res.slippage()}" if res.ok else res.reason)
    if not res.ok:
        return False
    pos = ad.positions(symbol)
    ok &= step("position visible with stop", bool(pos) and pos[0].stop_loss is not None,
               f"{pos[0].side.name} {pos[0].volume:g} stop {pos[0].stop_loss}" if pos else "none")
    ad.modify(res.ticket, stop_loss=t.ask - stop_dist * 0.75)
    pos = ad.positions(symbol)
    ok &= step("stop modified", bool(pos) and abs(pos[0].stop_loss - (t.ask - stop_dist * 0.75)) < spec.tick_size, f"{pos[0].stop_loss if pos else None}")

    if hasattr(ad, "_today"):
        ad._today = root.roll_date(*ym)
        rolled = ad.roll(symbol)
        after = ad.positions(symbol)
        ok &= step("roll to next contract", all(r.ok for r in rolled) and bool(after),
                   f"{len(rolled)} legs, now {root.code(*ad.front_month(symbol))}, stop {after[0].stop_loss if after else None}")
    closed = ad.close(res.ticket if res.ticket in ad._orders else next(iter(ad._orders)))
    ok &= step("close", closed.ok, f"@ {closed.fill_price}" if closed.ok else closed.reason)
    ok &= step("flat", not ad.positions(symbol), "no position")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="MES")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--allow-live", action="store_true")
    args = ap.parse_args()

    print(f"\nFUTURES ROUND TRIP - {args.symbol}  ({'FAKE' if args.fake else 'TWS'}{', SEND' if args.send else ', dry run'})\n")

    if args.fake:
        from execution.ib_fake import FakeIB
        fake = FakeIB(MICRO_UNIVERSE, {"MES": 6000.0, "MGC": 2650.0, "M6E": 1.08, "MCL": 70.0},
                      now=datetime.now(timezone.utc))
        ad = IBAdapter(ib=fake, today=date(2025, 12, 1), fill_timeout=1.0)
        ad.connect()
        ok = sequence(ad, args.symbol, send=True)  # the fake always sends; nothing is real
        print(f"\n  {'PATH VERIFIED on the test double' if ok else 'FAILED'}")
        return 0 if ok else 1

    ad = IBAdapter(port=args.port)
    try:
        ad.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"  cannot connect to TWS/Gateway on port {args.port}: {str(exc)[:90]}")
        print("  Start TWS (paper: port 7497) or IB Gateway (paper: 4002) with API enabled, then retry.")
        return 1
    try:
        accounts = ad.ib.managedAccounts()
        paper = all(a.startswith("DU") for a in accounts)
        step("paper account", paper or args.allow_live, f"{accounts}")
        if args.send and not paper and not args.allow_live:
            print("  refusing to send on a non-paper account")
            return 1
        ok = sequence(ad, args.symbol, send=args.send)
        print(f"\n  {'GATE MET' if ok else 'GATE NOT MET'}")
        return 0 if ok else 1
    finally:
        ad.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
