"""The kill switch.

A file on disk, because a file is the one channel that works when everything else
does not. Any process can set it: the runner, a monitoring alert, a cron job, or
you from a second terminal at three in the morning. It survives a crash, needs no
network, no database and no running Python.

    python -c "from risk.killswitch import KillFile; KillFile().engage('manual')"

The runner checks it at the top of every loop iteration and again immediately
before every order. Two checks rather than one because the gap between deciding to
trade and trading is exactly where a stop instruction arrives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class KillRecord:
    engaged: bool
    reason: str
    at: str
    by: str

    def __str__(self) -> str:
        if not self.engaged:
            return "kill switch: clear"
        return f"kill switch: ENGAGED at {self.at} by {self.by} - {self.reason}"


class KillFile:
    def __init__(self, path: str | Path = "state/KILL") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def engaged(self) -> bool:
        return self.path.exists()

    def read(self) -> KillRecord:
        if not self.path.exists():
            return KillRecord(False, "", "", "")
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # An unreadable kill file still means kill. Fail closed: the only
            # safe interpretation of "someone tried to stop the system" is stop.
            return KillRecord(True, "kill file present but unreadable", "", "unknown")
        return KillRecord(True, raw.get("reason", ""), raw.get("at", ""), raw.get("by", ""))

    def engage(self, reason: str, by: str = "system") -> KillRecord:
        record = KillRecord(True, reason, datetime.now(timezone.utc).isoformat(), by)
        self.path.write_text(
            json.dumps({"reason": reason, "at": record.at, "by": by}, indent=2),
            encoding="utf-8",
        )
        return record

    def clear(self) -> None:
        """Deliberately manual. Nothing in the system clears its own kill switch.

        If a limit breach engaged it, a human decides whether the cause has been
        addressed. Auto-clearing turns a stop into a pause, and the whole value of
        this file is that it is not a pause.
        """
        self.path.unlink(missing_ok=True)
