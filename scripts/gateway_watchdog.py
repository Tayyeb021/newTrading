"""Keep IB Gateway alive, and say clearly when only a human can fix it.

    python scripts/gateway_watchdog.py            # check, relaunch if the process is gone
    python scripts/gateway_watchdog.py --check    # report only, change nothing

Runs every ten minutes from Task Scheduler. There are three distinct states and
they need different answers, which is the whole reason this is not a one-line
`if not running: start`:

- **Serving.** The process is up, the API answers, an account and a price come
  back. Nothing to do.
- **Gone.** No process at all - a crash, or the machine rebooted. Relaunch it.
  It will come up at the login screen and still need a human, but a Gateway at
  a login screen is one click from working and an absent one is not.
- **Up but not serving.** The process exists and the API refuses or returns an
  empty account. This is what a logged-out Gateway looks like, and it is the
  state the nightly restart and the weekly re-authentication both leave behind.
  Relaunching does not help and would throw away a session that may still be
  recoverable, so this only reports.

Interactive Brokers restarts the Gateway every day. With **Auto restart** set
(Configure, Lock and Exit) that restart is silent and keeps the session, and a
human is needed only once a week, at about 01:00 New York time on Sunday. With
**Auto logoff** set instead, a human is needed every single day. The difference
is one radio button and it is the difference between a record with gaps and one
without.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ops.journal import Journal  # noqa: E402

GATEWAY = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "ibgateway" / "ibgateway.exe"
JOURNAL = ROOT / "state" / "gateway_watchdog.jsonl"
PORT = 4002

SERVING, GONE, NOT_SERVING = "serving", "gone", "up_but_not_serving"

def _quiet_ib_noise() -> None:
    """Silence the delayed-data warning ib_async logs on every quote.

    Error 354 is not an error here - this account has no live subscription and
    the adapter deliberately falls back to delayed. Left alone it writes four
    lines every ten minutes forever, and a log nobody can read is a log nobody
    reads.
    """
    import logging
    for name in ("ib_async", "ib_async.wrapper", "ib_async.client", "ib_insync"):
        logging.getLogger(name).setLevel(logging.CRITICAL)



def process_running() -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq ibgateway.exe"],
                             capture_output=True, text=True, timeout=60)
        return "ibgateway.exe" in out.stdout.lower()
    except Exception:  # noqa: BLE001
        return False


def probe() -> tuple[str, str]:
    """Which of the three states are we in, and why."""
    _quiet_ib_noise()
    if not process_running():
        return GONE, "no ibgateway.exe process"
    try:
        from execution.ib_adapter import IBAdapter
        ad = IBAdapter(port=PORT, client_id=96)
        ad.connect()
        try:
            a = ad.account()
            if a.equity <= 0:
                return NOT_SERVING, "API answers but the account reads zero - logged out"
            tick = ad.tick("MES")
            return SERVING, f"{a.equity:,.2f} {a.currency}, MES {tick.bid}/{tick.ask}"
        finally:
            ad.disconnect()
    except Exception as exc:  # noqa: BLE001
        return NOT_SERVING, f"{type(exc).__name__}: {str(exc)[:100]}"


def relaunch() -> tuple[bool, str]:
    if not GATEWAY.exists():
        return False, f"not installed at {GATEWAY}"
    try:
        subprocess.Popen([str(GATEWAY)], close_fds=True)
        return True, "launched; it will sit at the login screen until someone signs in"
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
    if state != SERVING:
        JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        Journal(JOURNAL).write("gateway", state=state, detail=detail, action=action)

    line = f"[{stamp:%Y-%m-%d %H:%M} UTC] gateway {state}: {detail}"
    if action != "none":
        line += f" -> {action}"
    if state == NOT_SERVING:
        line += ("\n  Only a person can fix this: sign in on the Gateway window. "
                 "If this happens every morning rather than weekly, set Configure -> "
                 "Lock and Exit -> Auto restart, not Auto logoff.")
    print(line)
    return 0 if state == SERVING else 1


if __name__ == "__main__":
    raise SystemExit(main())
