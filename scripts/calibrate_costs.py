"""Turn measured round trips into a calibrated cost model.

    python scripts/calibrate_costs.py            # report what the measurements say
    python scripts/calibrate_costs.py --write    # save them to config/costs.measured.json

`verify_roundtrip.py` appends every real fill to `config/measured_fills.json`.
This reads them, fits each symbol's spread and slippage from the MEDIAN of its
observations, and prints what changes against the placeholder assumptions the
backtests have been running on.

Why this matters more than it looks: every backtest in this repository has been
charging an ASSUMED half-spread of slippage and printing `calibrated=False` on
its report. Entry 007 died on friction at 295% of gross. If the real number is
higher than assumed, verdicts get worse, not better - which is exactly why the
measurement is taken before anything is trusted, and why the model refuses to
pretend. A symbol with no measurements stays uncalibrated and says so.
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
OUT = ROOT / "config" / "costs.measured.json"


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
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--write", action="store_true", help="save the fitted costs")
    ap.add_argument("--min-observations", type=int, default=2,
                    help="a symbol needs this many fills before its median means anything")
    args = ap.parse_args()

    store = load_fills(Path(args.fills))
    instruments = InstrumentConfig.load()
    baseline = CostModel()

    print(f"\nCOST CALIBRATION from {len(store)} instrument(s)\n")
    header = f"  {'symbol':<10}{'fills':>6}{'spread now':>13}{'assumed':>11}{'slip now':>11}{'assumed':>11}   verdict"
    print(header)
    print("  " + "-" * (len(header) - 2))

    fitted: dict[str, dict] = {}
    thin: list[str] = []
    for symbol in sorted(store):
        fills = store[symbol]
        spreads = [f["spread"] for f in fills if f.get("spread") is not None]
        slips = [abs(f["slippage"]) for f in fills if f.get("slippage") is not None]
        if len(fills) < args.min_observations or not spreads:
            thin.append(symbol)
            continue

        base = baseline.for_symbol(symbol)
        spread, slip = median(spreads), median(slips) if slips else base.slippage
        ratio = slip / spread if spread else float("inf")
        verdict = ("slippage under the assumed half-spread" if ratio < 0.5
                   else "slippage AT the assumed half-spread" if ratio < 0.6
                   else f"slippage {ratio:.2f}x the spread - worse than assumed")
        print(f"  {symbol:<10}{len(fills):>6}{spread:>13.6f}{base.spread:>11.6f}"
              f"{slip:>11.6f}{base.slippage:>11.6f}   {verdict}")
        fitted[symbol] = {
            "spread": spread, "slippage": slip,
            "commission_per_lot_per_side": base.commission_per_lot_per_side,
            "swap_long": base.swap_long, "swap_short": base.swap_short,
            "spread_multiple_at_open": base.spread_multiple_at_open,
            "observations": len(fills),
        }

    if thin:
        print(f"\n  too few observations to fit: {', '.join(thin)} "
              f"(need {args.min_observations}; run the round trip again on those)")
    missing = [s for s in instruments.symbols if s not in fitted]
    if missing:
        print(f"  never measured: {', '.join(missing)} - these stay UNCALIBRATED and every")
        print("  backtest touching them will keep saying so.")

    if not fitted:
        print("\n  nothing fitted.\n")
        return 1

    if args.write:
        payload = {
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "source": str(Path(args.fills).name),
            "costs": fitted,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\n  written to {Path(args.out).relative_to(ROOT)} for {len(fitted)} instrument(s)")
        print("  Backtests that load it may now report calibrated=True for those symbols.")
    else:
        print(f"\n  nothing written. Re-run with --write to save to {Path(args.out).relative_to(ROOT)}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
