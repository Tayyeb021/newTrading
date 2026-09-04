"""Phase 1 gate: read contract specs and prove what this account can actually trade.

Two jobs.

1. Print every spec the broker reports, so you can check them against the broker's
   own contract page. A mismatch here is the single most expensive class of bug in
   retail algo trading — a position a hundred times the intended size because the
   index CFD was $10 a point, not $0.10.

2. Compute the *minimum viable equity* per instrument: the smallest account that
   can express your risk limit in the instrument's minimum lot. This is what
   decides v1 scope, and it is arithmetic rather than opinion.

Runs offline against fixture specs so you can see the shape of the answer before
connecting anything:

    python scripts/check_specs.py --offline
    python scripts/check_specs.py --equity 5000 --offline
    python scripts/check_specs.py                      # live, needs MT5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import InstrumentConfig, RiskProfile  # noqa: E402
from core.types import SymbolSpec  # noqa: E402
from execution.paper import FIXTURE_SPECS  # noqa: E402
from risk.sizing import minimum_viable_equity, size_position  # noqa: E402

# Indicative daily ATR, in price units, used only to make the arithmetic concrete.
# Replace with measured values from your own data as soon as you have them.
INDICATIVE_ATR = {
    "EURUSD": 0.0070,
    "GBPUSD": 0.0090,
    "USDJPY": 0.900,
    "XAUUSD": 50.0,
    "US30": 450.0,
    "US500": 60.0,
}


def load_specs(offline: bool, symbols: list[str], aliases: dict[str, str]) -> dict[str, SymbolSpec]:
    if offline:
        return {s: FIXTURE_SPECS[s] for s in symbols if s in FIXTURE_SPECS}

    from execution.mt5_adapter import MT5Adapter  # imported late: Windows-only

    adapter = MT5Adapter(aliases=aliases)
    adapter.connect()
    try:
        out: dict[str, SymbolSpec] = {}
        for symbol in symbols:
            try:
                out[symbol] = adapter.spec(symbol)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  ! {symbol}: {exc}")
        return out
    finally:
        adapter.disconnect()


def print_specs(specs: dict[str, SymbolSpec]) -> None:
    print("\nCONTRACT SPECIFICATIONS")
    print("Verify every row against the broker's own contract page before trading.\n")
    head = f"{'symbol':<10}{'digits':>7}{'tick_size':>12}{'tick_value':>12}{'min_lot':>10}{'step':>8}{'$/1.0 move':>13}"
    print(head)
    print("-" * len(head))
    for symbol, spec in specs.items():
        print(
            f"{symbol:<10}{spec.digits:>7}{spec.tick_size:>12.6g}{spec.tick_value:>12.4g}"
            f"{spec.volume_min:>10.4g}{spec.volume_step:>8.4g}{spec.value_per_price_unit:>13,.2f}"
        )


def print_viability(
    specs: dict[str, SymbolSpec],
    equity: float,
    risk_fraction: float,
    atr_multiple: float,
) -> list[str]:
    print(f"\nVIABILITY AT {equity:,.0f} EQUITY, {risk_fraction:.2%} RISK PER TRADE")
    print(f"Stop distance = {atr_multiple} x daily ATR (indicative values).\n")

    head = (
        f"{'symbol':<10}{'stop (price)':>14}{'min lot risk':>14}{'as % equity':>13}"
        f"{'min equity':>13}  verdict"
    )
    print(head)
    print("-" * len(head))

    tradeable: list[str] = []
    for symbol, spec in specs.items():
        atr = INDICATIVE_ATR.get(symbol)
        if atr is None:
            print(f"{symbol:<10}{'no ATR value':>14}")
            continue

        stop = atr * atr_multiple
        min_lot_risk = spec.risk_for(spec.volume_min, stop)
        pct = min_lot_risk / equity
        needed = minimum_viable_equity(spec, risk_fraction, stop)

        result = size_position(spec, equity, risk_fraction, stop)
        if result.tradeable:
            verdict = f"OK  {result.volume:g} lots, risks {result.risk_fraction:.2%}"
            tradeable.append(symbol)
        else:
            verdict = f"BLOCKED  needs {needed:,.0f}"

        print(
            f"{symbol:<10}{stop:>14.5g}{min_lot_risk:>14,.2f}{pct:>12.2%} "
            f"{needed:>12,.0f}  {verdict}"
        )
    return tradeable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="use fixture specs, no broker needed")
    ap.add_argument("--equity", type=float, default=None, help="account equity to test against")
    ap.add_argument("--profile", default="challenge", help="risk profile name")
    args = ap.parse_args()

    instruments = InstrumentConfig.load()
    profile = RiskProfile.load(args.profile)
    equity = args.equity if args.equity is not None else 100_000.0

    print(f"profile        : {profile.name}")
    print(f"risk per trade : {profile.risk_per_trade:.3%} (ceiling {profile.max_risk_per_trade:.3%})")
    print(f"daily loss     : soft {profile.daily_loss_soft:.2%} / hard {profile.daily_loss_hard:.2%}")
    print(f"max drawdown   : soft {profile.max_drawdown_soft:.2%} / hard {profile.max_drawdown_hard:.2%}"
          f" ({'trailing' if profile.drawdown_trailing else 'static'})")
    print(f"source         : {'fixtures (offline)' if args.offline else 'live broker'}")

    specs = load_specs(args.offline, instruments.symbols, instruments.aliases)
    if not specs:
        print("\nNo specs loaded. Nothing to check.")
        return 1

    print_specs(specs)
    tradeable = print_viability(specs, equity, profile.risk_per_trade, profile.atr_stop_multiple)

    print("\nV1 SCOPE")
    if tradeable:
        print(f"  tradeable at {equity:,.0f}: {', '.join(tradeable)}")
    else:
        print(f"  nothing is tradeable at {equity:,.0f} under this profile.")
    blocked = [s for s in specs if s not in tradeable]
    if blocked:
        print(f"  blocked by minimum lot size: {', '.join(blocked)}")
        print("  These are not strategy problems. They lift when equity rises or the")
        print("  stop distance shrinks (a lower timeframe), and not before.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
