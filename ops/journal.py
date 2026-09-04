"""Append-only decision journal.

Records not just what happened but **the state that produced it**. When live
results diverge from the backtest — and they will — this is the only thing that
distinguishes the three possible causes: the edge decayed, execution got worse, or
the code is wrong. Without it you are guessing between them for months.

JSONL, one event per line, flushed on write. Append-only so a crash truncates at
worst the last line, and every earlier line is still readable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def _encode(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _encode(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value if not isinstance(obj.value, int) else obj.name
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_encode(v) for v in obj]
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class Journal:
    def __init__(self, path: str | Path = "state/journal.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **{k: _encode(v) for k, v in fields.items()},
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()
        return record

    # -------------------------------------------------------------- shortcuts

    def decision(self, signal, decision, state_summary: dict) -> None:
        """Every risk decision, approved or not, with the state behind it."""
        self.write(
            "decision",
            symbol=signal.symbol,
            strategy=signal.strategy,
            side=signal.side.name,
            stop_distance=signal.stop_distance,
            confidence=signal.confidence,
            approved=decision.approved,
            note=decision.note,
            breaches=[
                {"limit": b.limit, "severity": b.severity.value, "message": b.message,
                 "observed": b.observed, "threshold": b.threshold}
                for b in decision.breaches
            ],
            volume=decision.order.volume if decision.order else None,
            risk_fraction=decision.size.risk_fraction if decision.size else None,
            state=state_summary,
        )

    def fill(self, result, cid: str) -> None:
        self.write(
            "fill",
            client_id=cid,
            status=result.status.value,
            symbol=result.request.symbol,
            side=result.request.side.name,
            ticket=result.ticket,
            requested=result.requested_price,
            filled=result.fill_price,
            volume=result.filled_volume,
            slippage=result.slippage(),
            reason=result.reason,
        )

    def breach(self, breach) -> None:
        self.write(
            "breach", limit=breach.limit, severity=breach.severity.value,
            message=breach.message, observed=breach.observed, threshold=breach.threshold,
        )

    def heartbeat(self, equity: float, positions: int, **extra: Any) -> None:
        self.write("heartbeat", equity=equity, positions=positions, **extra)

    # ------------------------------------------------------------------ replay

    def read(self, event: str | None = None, limit: int | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a crash can truncate the final line; skip it
                if event is None or record.get("event") == event:
                    out.append(record)
        return out[-limit:] if limit else out

    def known_tickets(self) -> set[int]:
        """Tickets this system has recorded opening. Anything else is an orphan."""
        return {
            r["ticket"] for r in self.read("fill")
            if r.get("ticket") and r.get("status") == "filled"
        }

    def summary(self, limit: int = 200) -> str:
        records = self.read(limit=limit)
        if not records:
            return "journal is empty"
        counts: dict[str, int] = {}
        for r in records:
            counts[r["event"]] = counts.get(r["event"], 0) + 1

        lines = [f"journal: {len(records)} recent events at {self.path}"]
        for event, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {event:<14}{n:>6}")

        rejected = [
            r for r in records
            if r["event"] == "decision" and not r.get("approved")
        ]
        if rejected:
            by_limit: dict[str, int] = {}
            for r in rejected:
                for b in r.get("breaches") or [{"limit": r.get("note", "unknown")}]:
                    by_limit[b["limit"]] = by_limit.get(b["limit"], 0) + 1
            lines.append("  rejections by cause:")
            for name, n in sorted(by_limit.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {name:<26}{n:>6}")
        return "\n".join(lines)
