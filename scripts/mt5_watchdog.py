"""Keep the MetaTrader 5 terminal alive, and say clearly when only a human can fix it.

    python scripts/mt5_watchdog.py            # check, relaunch if the terminal is gone
    python scripts/mt5_watchdog.py --check    # report only, change nothing

Runs every ten minutes from Task Scheduler, and exists because of the night of
2026-09-08. The shadow week traded properly for the first time that evening -
three round trips, closed by the rollover rule at 19:15 - and by 08:27 the next
morning it had been halted for hours with no prices. `terminal64.exe` was simply
gone from the process list. IB Gateway had had a watchdog since Sunday;
MetaTrader had none, so nothing noticed and nothing acted. Relaunching it by
hand fixed it in under a minute.

The same three states as `gateway_watchdog.py`, because they need different
answers:

- **Serving.** Process up, the Python API answers, an account and a price come
  back. Nothing to do.
- **Gone.** No process at all. Relaunch it. Unlike IB Gateway, MetaTrader
  normally restores its saved login on its own, so a relaunch usually is the
  whole fix - that is exactly what happened on the morning of 2026-09-09.
- **Up but not serving.** The process exists and the API refuses, or the account
  reads zero. That is a terminal sitting at a login prompt, or one whose broker
  connection has dropped, or Algo Trading switched off. Relaunching does not
  help and would throw away a session that may recover on its own, so this only
  reports.

The terminal must run in the same Windows session as this task, which is why
the task is registered "run only when user is logged on" - disconnect RDP, do
not sign out.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ops.journal import Journal  # noqa: E402

JOURNAL = ROOT / "state" / "mt5_watchdog.jsonl"
PROCESS = "terminal64.exe"

#: Where MetaTrader installs itself. The environment variable wins so a
#: non-standard install does not need a code change.
CANDIDATES = [
    Path(os.environ["MT5_TERMINAL"]) if os.environ.get("MT5_TERMINAL") else None,
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "MetaTrader 5" / PROCESS,
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "MetaTrader 5" / PROCESS,
]

SERVING, GONE, NOT_SERVING = "serving", "gone", "up_but_not_serving"


def terminal_path() -> Path | None:
    for p in CANDIDATES:
        if p is not None and p.exists():
            return p
    return None


def process_running() -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {PROCESS}"],
                             capture_output=True, text=True, timeout=60)
        return PROCESS.lower() in out.stdout.lower()
    except Exception:  # noqa: BLE001
        return False


def probe() -> tuple[str, str]:
    """Which of the three states are we in, and why."""
    if not process_running():
        return GONE, f"no {PROCESS} process"
    try:
        from core.config import InstrumentConfig
        from execution.mt5_adapter import MT5Adapter

        inst = InstrumentConfig.load()
        ad = MT5Adapter(aliases=inst.aliases)
        ad.connect()
        try:
            a = ad.account()
            if a.equity <= 0:
                return NOT_SERVING, "API answers but the account reads zero - logged out"
            symbol = (inst.active or inst.symbols or ["EURUSD"])[0]
            tick = ad.tick(symbol)
            clock = ad.clock_status.value if ad.clock_status else "unknown"
            return SERVING, f"{a.equity:,.2f} {a.currency}, {symbol} {tick.bid}/{tick.ask}, clock {clock}"
        finally:
            try:
                ad.disconnect()
            except Exception:  # noqa: BLE001 - already reporting; a failed close is not the story
                pass
    except Exception as exc:  # noqa: BLE001
        return NOT_SERVING, f"{type(exc).__name__}: {str(exc)[:100]}"


def relaunch() -> tuple[bool, str]:
    exe = terminal_path()
    if exe is None:
        return False, ("not found - set MT5_TERMINAL to the full path of terminal64.exe "
                       "if MetaTrader is installed somewhere unusual")
    try:
        subprocess.Popen([str(exe)], close_fds=True)
        return True, f"launched {exe}; it normally restores its saved login by itself"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    args = ap.parse_args()

    state, detail = probe()
    action = "none"
    if state == GONE and not args.check:
        ok, note = relaunch()
        action = f"relaunched ({note})" if ok else f"relaunch FAILED ({note})"

    stamp = datetime.now(timezone.utc)
    # Only failures are journalled, so an empty journal means an uneventful
    # history rather than a watchdog that never ran.
    if state != SERVING:
        JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        Journal(JOURNAL).write("mt5", state=state, detail=detail, action=action)

    line = f"[{stamp:%Y-%m-%d %H:%M} UTC] mt5 {state}: {detail}"
    if action != "none":
        line += f" -> {action}"
    if state == NOT_SERVING:
        line += ("\n  Only a person can fix this: open the MetaTrader window and sign in, "
                 "and check that Algo Trading is enabled.")
    print(line)
    return 0 if state == SERVING else 1


if __name__ == "__main__":
    raise SystemExit(main())
