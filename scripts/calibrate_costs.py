"""Turn measured round trips into a calibrated cost model.

    python scripts/calibrate_costs.py            # report what the measurements say
    python scripts/calibrate_costs.py --write    # save them to config/costs.measured.json

`verify_roundtrip.py` appends every real fill to `config/measured_fills.json`
and `spread_profile.py` appends quoted spreads to `state/spread_samples.jsonl`.
This reads both, fits each symbol's spread and slippage from the MEDIAN of the
observations, and prints what changes against the assumptions the backtests have
been running on.

Why this matters more than it looks: every backtest here charges an ASSUMED
half-spread of slippage and prints `calibrated=False` to say so. Entry 007 died
on friction at 295% of gross. If real costs are higher than assumed, verdicts get
worse, not better - which is why the measurement is taken before anything is
trusted, and why this refuses to pretend.

**The demo-fill trap.** On 2026-09-07 the first real round trips came back with
slippage of exactly zero on all eight legs, across instruments quoting from 0 to
120 points of spread, including a six-second-old quote on an index. A demo
server has no liquidity to consume, so it fills at the quote. Calibrating to
that would halve modelled friction and resurrect strategies that are dead, which
is the worst error this repository could make. So: slippage that is all-zero, or
implausibly small against the spread, is REFUSED, the assumption is kept, and
the symbol stays marked uncalibrated for slippage. Only a live account can
measure it. Spreads, by contrast, are quoted rather than filled and survive the
demo, so those are calibrated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.costs import CostModel  # noqa: E402
from core.config import InstrumentConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MEASURED = ROOT / "config" / "measured_fills.json"
SPREADS = ROOT / "state" / "spread_samples.jsonl"
OUT = ROOT / "config" / "costs.measured.json"

#: Below this fraction of the spread, a slippage measurement is a demo artefact
#: rather than a number. Real retail slippage on a market order sits near half
#: the spread; anything under a twentieth of it means the venue filled at quote.
MIN_CREDIBLE_SLIP_RATIO = 0.05


def slippage_is_credible(slips: list[float], spread: float) -> tuple[bool, str]:
    """Can this slippage sample be believed? Demo servers fill at the quote."""
    if not slips:
        return False, "no slippage observations"
    if spread <= 0:
        return False, "spread was zero, so the ratio means nothing"
    med = median(slips)
    if med <= 0:
        return False, f"median slippage is zero over {len(slips)} fills - the venue filled at quote"
    if med / spread < MIN_CREDIBLE_SLIP_RATIO:
        return False, f"median slippage is {med / spread:.3f} of the spread - implausibly good"
    return True, f"median {med / spread:.2f} of the spread over {len(slips)} fills"


def quoted_spreads(path: Path) -> dict[str, list[float]]:
    """Spreads from the read-only profiler, in price units. Many samples across
    hours beats the two taken during a round trip."""
    out: dict[str, list[float]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.setdefault(r["symbol"], []).append(float(r["spread"]))
    return out


def load_fills(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        print(f"no measurements at {path.relative_to(ROOT)}.")
        print("  Run:  python scripts/verify_roundtrip.py")
        print("  That places one minimum-lot round trip per instrument on the demo account")
        print("  and records what the fills actually cost. Nothing here can invent them.")
        raise SystemExit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fills", default=str(MEASURED))
    ap.add_argument("--spreads", default=str(SPREADS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--write", action="store_true", help="save the fitted costs")
    ap.add_argument("--min-observations", type=int, default=2,
                    help="a symbol needs this many fills before its median means anything")
    args = ap.parse_args()

    store = load_fills(Path(args.fills))
    quoted = quoted_spreads(Path(args.spreads))
    instruments = InstrumentConfig.load()
    baseline = CostModel()

    print(f"\nCOST CALIBRATION - {len(store)} instrument(s) with fills, "
          f"{len(quoted)} with quoted-spread samples\n")
    header = (f"  {'symbol':<9}{'fills':>6}{'quotes':>8}{'spread':>11}{'assumed':>10}"
              f"{'slip':>10}{'assumed':>10}  slippage")
    print(header)
    print("  " + "-" * (len(header) - 2))

    fitted: dict[str, dict] = {}
    thin: list[str] = []
    rejected: list[tuple[str, str]] = []
    for symbol in sorted(set(store) | set(quoted)):
        fills = store.get(symbol, [])
        fill_spreads = [f["spread"] for f in fills if f.get("spread") is not None]
        slips = [abs(f["slippage"]) for f in fills if f.get("slippage") is not None]
        samples = quoted.get(symbol, [])
        # Prefer the profiler's spread: hundreds of quotes across hours beat two
        # taken in the second a round trip happened to run.
        spreads = samples if len(samples) >= 30 else fill_spreads
        if not spreads or (not samples and len(fills) < args.min_observations):
            thin.append(symbol)
            continue

        base = baseline.for_symbol(symbol)
        spread = median(spreads)
        if spread <= 0:
            # A raw-spread account really can quote zero on EURUSD, but a cost
            # model that charges nothing to cross is not a cost model. Keep the
            # assumption and say so; the commission is what such an account
            # actually charges, and that has to be verified separately.
            rejected.append((symbol, f"median spread over {len(spreads)} samples is zero - "
                                     f"raw-spread account? commission must carry the cost instead"))
            spread = base.spread
            spread_ok = False
        else:
            spread_ok = True
        credible, why = slippage_is_credible(slips, spread) if spread_ok else (False, "spread not calibrated")
        if credible:
            slip = median(slips)
        else:
            slip = base.slippage  # keep the assumption; do not invent a better one
            rejected.append((symbol, why))
        print(f"  {symbol:<9}{len(fills):>6}{len(samples):>8}{spread:>11.5f}{base.spread:>10.5f}"
              f"{slip:>10.5f}{base.slippage:>10.5f}  "
              f"{'measured' if credible else 'ASSUMED'}{'' if spread_ok else ' (spread too)'}")
        fitted[symbol] = {
            "spread": spread, "slippage": slip,
            "spread_calibrated": spread_ok,
            "slippage_calibrated": credible,
            "slippage_note": why,
            "commission_per_lot_per_side": base.commission_per_lot_per_side,
            "swap_long": base.swap_long, "swap_short": base.swap_short,
            "spread_multiple_at_open": base.spread_multiple_at_open,
            "fills": len(fills), "quote_samples": len(samples),
        }

    if rejected:
        print(f"\n  REFUSED on {len(rejected)} count(s) - the assumption stands:")
        for symbol, why in rejected:
            print(f"    {symbol:<9}{why}")
        print("    A demo server has no liquidity to consume, so it fills at the quote.")
        print("    Modelling that as real would halve friction and revive dead strategies.")
        print("    Only a live account can measure slippage; until then it stays assumed.")

    if thin:
        print(f"\n  too few observations to fit: {', '.join(thin)}")
    missing = [s for s in instruments.symbols if s not in fitted]
    if missing:
        print(f"  never measured: {', '.join(missing)} - these stay fully UNCALIBRATED.")

    if not fitted:
        print("\n  nothing fitted.\n")
        return 1

    if args.write:
        payload = {
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "sources": {"fills": Path(args.fills).name, "quotes": Path(args.spreads).name},
            "slippage_measurable": not rejected,
            "costs": fitted,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\n  written to {Path(args.out).relative_to(ROOT)} for {len(fitted)} instrument(s)")
        if rejected:
            print("  Spreads are calibrated; slippage is not, and the file records which is which.")
    else:
        print(f"\n  nothing written. Re-run with --write to save to {Path(args.out).relative_to(ROOT)}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
