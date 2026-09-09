"""Is everything that should be running actually running, and being fed?

    python scripts/healthcheck.py            # human-readable, exit 1 if anything is wrong
    python scripts/healthcheck.py --json     # for a monitor
    python scripts/healthcheck.py --quiet    # print only problems

Checks the things that fail silently. A process can be alive and doing nothing:
the shadow week sat "Ready" for an hour on 2026-09-07 after a console interrupt
killed it, and IB Gateway forces a daily re-login after which it is up, listening
and useless. Both look fine from the outside, which is exactly why this exists.

Every check answers one question with a yes, a no, or a number, and anything
stale is a failure rather than a shrug.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ops.journal import Journal  # noqa: E402

OK, WARN, FAIL = "ok", "warn", "FAIL"

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



@dataclass
class Check:
    name: str
    status: str
    detail: str
    data: dict = field(default_factory=dict)


def _proc_matching(pattern: str) -> list[int]:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             f"Where-Object {{ $_.CommandLine -match '{pattern}' }} | "
             f"Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=60)
        return [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    except Exception:  # noqa: BLE001
        return []


def _process_up(name: str) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {name}"],
                             capture_output=True, text=True, timeout=60)
        return name.lower() in out.stdout.lower()
    except Exception:  # noqa: BLE001
        return False


def journal_freshness(path: Path, label: str, max_age_minutes: float) -> Check:
    if not path.exists():
        return Check(label, FAIL, f"no journal at {path.name}")
    beats = Journal(path).read("heartbeat")
    if not beats:
        return Check(label, FAIL, "journal has no heartbeat")
    last = beats[-1]
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(last["ts"])).total_seconds() / 60
    status = OK if age <= max_age_minutes else FAIL
    return Check(label, status,
                 f"last heartbeat {age:.0f} min ago (limit {max_age_minutes:.0f}), "
                 f"{len(beats):,} total, equity {last.get('equity', 0):,.2f}"
                 + (", HALTED" if last.get("halted") else ""),
                 {"age_minutes": round(age, 1), "heartbeats": len(beats),
                  "equity": last.get("equity"), "halted": bool(last.get("halted"))})


def gateway_check() -> Check:
    """Up is not the same as logged in. Ask it for an account and a price."""
    _quiet_ib_noise()
    if not _process_up("ibgateway.exe"):
        return Check("IB Gateway", FAIL, "process is not running")
    try:
        from execution.ib_adapter import IBAdapter
        ad = IBAdapter(port=4002, client_id=97)
        ad.connect()
        try:
            a = ad.account()
            if a.equity <= 0:
                return Check("IB Gateway", FAIL, "connected but the account reads zero - logged out?")
            tick = ad.tick("MES")
            kind = {1: "live", 3: "delayed", 4: "frozen"}.get(ad.market_data_type, "?")
            return Check("IB Gateway", OK,
                         f"logged in, {a.equity:,.2f} {a.currency}, MES {tick.bid}/{tick.ask} ({kind})",
                         {"equity": a.equity, "market_data_type": ad.market_data_type})
        finally:
            ad.disconnect()
    except Exception as exc:  # noqa: BLE001
        return Check("IB Gateway", FAIL,
                     f"up but not serving: {type(exc).__name__}: {str(exc)[:90]}. "
                     f"IB forces a daily re-login; log back in on port 4002.")


def kill_switch(path: Path, label: str) -> Check:
    if not path.exists():
        return Check(label, OK, "clear")
    try:
        text = path.read_text(encoding="utf-8")[:120].replace("\n", " ")
    except OSError:
        text = "unreadable, which is read as ENGAGED"
    return Check(label, WARN, f"ENGAGED: {text}")


def run_checks() -> list[Check]:
    state = ROOT / "state"
    checks: list[Check] = []

    fwd = _proc_matching("run_forward")
    checks.append(Check("forward record process", OK if fwd else FAIL,
                        f"PID {fwd[0]}" if fwd else "not running - the task should restart it within 15 min",
                        {"pids": fwd}))
    # hourly poll, so two hours of silence means it has stopped ticking
    checks.append(journal_freshness(state / "forward_journal.jsonl", "forward record heartbeat", 150))
    checks.append(gateway_check())
    checks.append(kill_switch(state / "FORWARD_KILL", "forward kill switch"))

    shadow = _proc_matching("shadow.py")
    if shadow or (state / "shadow_journal.jsonl").exists():
        checks.append(Check("shadow week process", OK if shadow else WARN,
                            f"PID {shadow[0]}" if shadow else "not running (finished, or waiting for the trigger)",
                            {"pids": shadow}))
        if shadow:
            checks.append(journal_freshness(state / "shadow_journal.jsonl", "shadow week heartbeat", 15))
        up = _process_up("terminal64.exe")
        checks.append(Check("MetaTrader 5", OK if up else FAIL,
                            "running (TradingMT5Watchdog relaunches it within 10 min)" if up
                            else "not running - shadow cannot see prices; the watchdog should "
                                 "relaunch it within 10 min"))
        checks.append(kill_switch(state / "SHADOW_KILL", "shadow kill switch"))

    try:
        free_gb = int(subprocess.run(
            ["powershell", "-NoProfile", "-Command", "[math]::Round((Get-PSDrive C).Free / 1GB)"],
            capture_output=True, text=True, timeout=60).stdout.strip() or 0)
        checks.append(Check("disk", OK if free_gb > 5 else FAIL, f"{free_gb} GB free on C:",
                            {"free_gb": free_gb}))
    except Exception:  # noqa: BLE001
        pass
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="print only problems")
    args = ap.parse_args()

    checks = run_checks()
    worst = FAIL if any(c.status == FAIL for c in checks) else (
        WARN if any(c.status == WARN for c in checks) else OK)

    if args.json:
        print(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(), "status": worst,
                          "checks": [{"name": c.name, "status": c.status, "detail": c.detail, **c.data}
                                     for c in checks]}, indent=2))
    else:
        shown = [c for c in checks if not args.quiet or c.status != OK]
        if shown:
            print(f"\nHEALTH {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC - {worst.upper()}\n")
            for c in shown:
                mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[c.status]
                print(f" [{mark}] {c.name:<26} {c.detail}")
            print()
    return 0 if worst == OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
