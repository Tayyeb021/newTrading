"""Daily digest of a shadow-mode run, from its journal.

    python scripts/shadow_report.py                 # last 24 hours
    python scripts/shadow_report.py --hours 168     # the whole week

Reads state/shadow_journal.jsonl - the same append-only journal the live
runner writes - and answers the questions that matter after a day of
unattended running: did it stay up, did it halt and why, what did the risk
engine decide, what would have filled, and how stale was the feed.

Appends the digest to state/shadow_reports.md so a week reads as one file.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ops.journal import Journal  # noqa: E402


def within(records, hours):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
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


def digest(journal: Journal, hours: float) -> str:
    recs = within(journal.read(), hours)
    now = datetime.now(timezone.utc)
    L = [f"## Shadow report - {now:%Y-%m-%d %H:%M} UTC - last {hours:g}h", ""]
    if not recs:
        L.append("no journal entries in the window - the runner was not running, or wrote nothing")
        return "\n".join(L)

    kinds = Counter(r["event"] for r in recs)
    beats = [r for r in recs if r["event"] == "heartbeat"]
    L.append(f"- events: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))

    if beats:
        halted = sum(1 for b in beats if b.get("halted"))
        eq = [b["equity"] for b in beats if b.get("equity") is not None]
        ages = [b["feed_age_s"] for b in beats if b.get("feed_age_s") is not None]
        first, last = beats[0], beats[-1]
        span = (datetime.fromisoformat(last["ts"]) - datetime.fromisoformat(first["ts"])).total_seconds() / 3600
        L.append(f"- uptime: {len(beats)} heartbeats over {span:.1f}h; halted on {halted} of them ({halted / len(beats):.0%})")
        if eq:
            L.append(f"- equity: start {eq[0]:,.2f}  end {eq[-1]:,.2f}  min {min(eq):,.2f}  max {max(eq):,.2f}")
        if ages:
            L.append(f"- feed age: median {statistics.median(ages):.0f}s  max {max(ages):.0f}s"
                     + ("  (stale = market closed or feed lost)" if max(ages) > 600 else ""))
        gaps = []
        for a, b in zip(beats, beats[1:]):
            d = (datetime.fromisoformat(b["ts"]) - datetime.fromisoformat(a["ts"])).total_seconds()
            if d > 300:
                gaps.append((a["ts"][11:16], b["ts"][11:16], d / 60))
        if gaps:
            L.append(f"- GAPS in heartbeats (>5 min): " + "; ".join(f"{s}->{e} {m:.0f}min" for s, e, m in gaps[:5]))

    decisions = [r for r in recs if r["event"] == "decision"]
    if decisions:
        ok = sum(1 for d in decisions if d.get("approved"))
        reasons = Counter()
        for d in decisions:
            if not d.get("approved"):
                for b in d.get("breaches") or []:
                    reasons[b["limit"]] += 1
                if not d.get("breaches"):
                    reasons[(d.get("note") or "unknown").split(":")[0]] += 1
        L.append(f"- decisions: {len(decisions)} ({ok} approved, {len(decisions) - ok} refused)"
                 + (f"; refused by " + ", ".join(f"{k} {v}" for k, v in reasons.most_common(5)) if reasons else ""))
        by_sleeve = Counter((d.get("strategy"), d.get("symbol"), d.get("side")) for d in decisions if d.get("approved"))
        for (s, sym, side), n in by_sleeve.most_common(8):
            L.append(f"    approved: {s} {sym} {side} x{n}")

    breaches = Counter(r["limit"] for r in recs if r["event"] == "breach")
    if breaches:
        L.append("- limit breaches: " + ", ".join(f"{k} {v}" for k, v in breaches.most_common()))

    fills = [r for r in recs if r["event"] == "fill"]
    if fills:
        filled = [f for f in fills if f.get("status") == "filled"]
        slips = [abs(f["slippage"]) for f in filled if f.get("slippage") is not None]
        L.append(f"- paper fills: {len(filled)} of {len(fills)} attempts"
                 + (f"; median |slippage| {statistics.median(slips):.6f}" if slips else ""))
        for f in filled[-5:]:
            L.append(f"    {f['ts'][5:16]} {f.get('symbol')} {f.get('side')} {f.get('volume')} @ {f.get('filled')}  [{f.get('client_id')}]")

    for kind in ("kill_engaged", "loop_error", "strategy_error", "worker_error", "roll", "roll_error", "startup", "shutdown"):
        items = [r for r in recs if r["event"] == kind]
        if items:
            L.append(f"- {kind}: {len(items)}" + (f" - last: {items[-1].get('error') or items[-1].get('reason') or ''}"[:120] if kind.endswith("error") or kind == "kill_engaged" else ""))

    L.append("")
    L.append("No order reached the broker: execution is routed to the paper adapter in shadow mode.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--journal", default="state/shadow_journal.jsonl")
    ap.add_argument("--out", default="state/shadow_reports.md")
    args = ap.parse_args()

    text = digest(Journal(args.journal), args.hours)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
