"""What happened overnight, in one screen.

    python scripts/overnight.py            # since 18:00 UTC yesterday
    python scripts/overnight.py --hours 36

Written to answer one question each morning without anyone having to correlate
four logs: **did the systems survive the night, and if not, what needs a human?**

The specific thing it was built for is Interactive Brokers' nightly restart. With
**Auto restart** set, the Gateway does a soft restart around 23:45 local, keeps
its session, and this report shows an uninterrupted night. With **Auto logoff**
set instead, the session ends and the report shows a gap plus a run of
"up but not serving" - which is the signature that the setting is wrong, and the
only reliable way to tell the two apart from outside.

Appends to state/overnight.md so a week of mornings reads as one file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ops.journal import Journal  # noqa: E402

STATE = ROOT / "state"


def _since(hours: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def _in_window(records: list[dict], cutoff: datetime) -> list[dict]:
    out = []
    for r in records:
        try:
            ts = datetime.fromisoformat(r["ts"])
        except (KeyError, ValueError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= cutoff:
            out.append(r)
    return out


def runner_section(path: Path, label: str, poll_minutes: float, cutoff: datetime) -> list[str]:
    if not path.exists():
        return [f"- **{label}**: no journal"]
    recs = _in_window(Journal(path).read(), cutoff)
    beats = [r for r in recs if r["event"] == "heartbeat"]
    if not beats:
        return [f"- **{label}**: SILENT for the whole window - it was not running"]

    first, last = beats[0], beats[-1]
    halted = sum(1 for b in beats if b.get("halted"))
    gaps = []
    for a, b in zip(beats, beats[1:]):
        mins = (datetime.fromisoformat(b["ts"]) - datetime.fromisoformat(a["ts"])).total_seconds() / 60
        if mins > poll_minutes * 3:
            gaps.append((a["ts"][11:16], b["ts"][11:16], mins))

    lines = [f"- **{label}**: {len(beats)} heartbeats, halted on {halted} ({halted / len(beats):.0%})"]
    eq0, eq1 = first.get("equity"), last.get("equity")
    if eq0 and eq1:
        lines.append(f"    equity {eq0:,.2f} -> {eq1:,.2f} ({(eq1 / eq0 - 1) * 100:+.2f}%)")
    if gaps:
        lines.append("    GAPS: " + "; ".join(f"{s}->{e} {m:.0f}min" for s, e, m in gaps[:4]))
    else:
        lines.append("    no gaps - it ran the whole window")

    for kind in ("decision", "fill", "feed_reconnect", "loop_error", "roll", "kill_engaged"):
        n = sum(1 for r in recs if r["event"] == kind)
        if n:
            lines.append(f"    {kind}: {n}")
    breaches: dict[str, int] = {}
    for r in recs:
        if r["event"] == "breach":
            breaches[r["limit"]] = breaches.get(r["limit"], 0) + 1
    if breaches:
        lines.append("    breaches: " + ", ".join(f"{k} {v}" for k, v in breaches.items()))
    return lines


def gateway_section(cutoff: datetime) -> list[str]:
    path = STATE / "gateway_watchdog.jsonl"
    if not path.exists():
        return ["- **IB Gateway**: no problem ever recorded - the watchdog only writes failures"]
    events = _in_window(Journal(path).read(), cutoff)
    if not events:
        return ["- **IB Gateway**: clean night, nothing to report",
                "    Auto restart is doing its job; the session survived."]

    states: dict[str, int] = {}
    for e in events:
        states[e.get("state", "?")] = states.get(e.get("state", "?"), 0) + 1
    lines = [f"- **IB Gateway**: {len(events)} problem checks - " +
             ", ".join(f"{k} {v}" for k, v in states.items())]
    lines.append(f"    first {events[0]['ts'][11:16]}, last {events[-1]['ts'][11:16]} UTC")
    if states.get("up_but_not_serving"):
        lines.append("    **The session ended and a human must sign in.** If this happens every")
        lines.append("    night rather than weekly, the setting is still Auto logoff: change it to")
        lines.append("    Auto restart under Configure, Lock and Exit.")
    if states.get("gone"):
        lines.append("    The process vanished; the watchdog relaunched it, but it needs a login.")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=None,
                    help="window; default is since 18:00 UTC yesterday")
    ap.add_argument("--out", default=str(STATE / "overnight.md"))
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    cutoff = _since(args.hours) if args.hours else now.replace(
        hour=18, minute=0, second=0, microsecond=0) - timedelta(days=1)
    hours = (now - cutoff).total_seconds() / 3600

    lines = [f"## Overnight {cutoff:%Y-%m-%d %H:%M} -> {now:%m-%d %H:%M} UTC ({hours:.0f}h)", ""]
    lines += gateway_section(cutoff)
    lines += runner_section(STATE / "forward_journal.jsonl", "forward record", 60, cutoff)
    lines += runner_section(STATE / "shadow_journal.jsonl", "shadow week", 0.2, cutoff)

    manifest = STATE / "forward_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        started = datetime.fromisoformat(m["started_at"])
        lines += ["", f"- forward record running {(now - started).days}d, "
                      f"{len(m['universe'])} markets, sleeves {', '.join(m['sleeves'])}",
                  "  first monthly decision 2026-10-01"]

    text = "\n".join(lines)
    print("\n" + text + "\n")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
