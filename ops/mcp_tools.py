"""Read-only views of the trading system, for an AI assistant over MCP.

Every function here READS. None can place, modify or close an order, engage
or clear the kill switch, edit a config, or touch a credential. That is not a
policy layered on top; there is simply no code path from this module to the
execution adapter's write methods. An assistant connected through the MCP
server can tell you what the system is doing and what the research found. It
cannot trade, and it cannot be talked into trading.

The functions are plain Python so they can be tested without the MCP runtime;
`ops/mcp_server.py` registers them as tools.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"


def _journal(path: str | Path):
    from ops.journal import Journal
    return Journal(path)


def _iso(ts: str | None) -> str | None:
    return ts


def status(journal_path: str = "state/journal.jsonl", state_path: str = "state/session.json",
           kill_path: str = "state/KILL") -> dict[str, Any]:
    """Where the live runner stands: last heartbeat, halted flag, equity, kill
    switch, and the persisted session book (day-start and high-water equity)."""
    from live.state import StateStore
    from risk.killswitch import KillFile

    j = _journal(ROOT / journal_path)
    beats = j.read("heartbeat", limit=1)
    last = beats[-1] if beats else None
    age = None
    if last and last.get("ts"):
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(last["ts"])).total_seconds()
        except ValueError:
            age = None
    kill = KillFile(ROOT / kill_path)
    session = None
    try:
        st = StateStore(ROOT / state_path).load()
        if st is not None:
            session = {"session_date": st.session_date, "starting_equity": st.starting_equity,
                       "day_start_equity": st.day_start_equity, "high_water_equity": st.high_water_equity,
                       "killed": st.killed, "kill_reason": st.kill_reason, "updated_at": st.updated_at}
    except ValueError as exc:
        session = {"error": str(exc)}
    counts: dict[str, int] = {}
    for r in j.read(limit=500):
        counts[r["event"]] = counts.get(r["event"], 0) + 1
    return {
        "journal": str(ROOT / journal_path),
        "last_heartbeat": last,
        "heartbeat_age_seconds": age,
        "running": age is not None and age < 120,
        "halted": bool(last.get("halted")) if last else None,
        "kill_switch_engaged": kill.engaged(),
        "session": session,
        "recent_event_counts": counts,
    }


def journal(event: str | None = None, n: int = 20, journal_path: str = "state/journal.jsonl") -> list[dict]:
    """The last n journal records, optionally of one event type
    (decision, fill, breach, heartbeat, resize, roll, allocation, ...)."""
    return _journal(ROOT / journal_path).read(event, limit=max(1, min(int(n), 500)))


def decisions(n: int = 20, approved: bool | None = None, journal_path: str = "state/journal.jsonl") -> list[dict]:
    """Recent risk-engine decisions: symbol, side, approved or not, the limits
    that refused it, and the sizing note. Pass approved=False for refusals only."""
    recs = _journal(ROOT / journal_path).read("decision")
    if approved is not None:
        recs = [r for r in recs if bool(r.get("approved")) == approved]
    out = []
    for r in recs[-max(1, min(int(n), 500)):]:
        out.append({
            "ts": r.get("ts"), "symbol": r.get("symbol"), "strategy": r.get("strategy"), "side": r.get("side"),
            "approved": r.get("approved"), "volume": r.get("volume"), "risk_fraction": r.get("risk_fraction"),
            "note": r.get("note"), "refused_by": [b.get("limit") for b in (r.get("breaches") or [])],
        })
    return out


def shadow_report(hours: float = 24.0, journal_path: str = "state/shadow_journal.jsonl") -> str:
    """Digest of the shadow (paper) run over the last `hours`: uptime, gaps,
    equity, feed age, decisions by refusal reason, paper fills, errors."""
    import sys
    sys.path.append(str(ROOT / "scripts"))
    from shadow_report import digest
    return digest(_journal(ROOT / journal_path), float(hours))


def verdicts(state_dir: str = "state") -> dict[str, Any]:
    """Every pre-registered research verdict on disk (state/gauntlet_*.json):
    pass or fail per threshold, with the numbers."""
    out: dict[str, Any] = {}
    for p in sorted((ROOT / state_dir).glob("gauntlet_*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            out[p.stem] = {"error": str(exc)}
            continue
        book = data.get("book") or data.get("carry") or data.get("filtered") or {}
        out[p.stem] = {
            "passed": data.get("passed"),
            "net_sharpe": book.get("net_sharpe"),
            "max_drawdown": book.get("max_drawdown"),
            "verdict": data.get("verdict", []),
        }
    return out


def research_log(entry: str | None = None, path: str = "RESEARCH_LOG.md") -> str:
    """The research log, or the sections for one entry number (e.g. '010'):
    every hypothesis, its thresholds declared before data, and its verdict."""
    text = (ROOT / path).read_text(encoding="utf-8")
    if not entry:
        return text
    key = str(entry).strip()
    parts = re.split(r"(?m)^(?=## )", text)
    hits = [p for p in parts if re.match(rf"## {re.escape(key)}\b", p)]
    return "\n".join(hits) if hits else f"no entry {key!r} in {path}"


def capital_ladder(equity: float, risk_per_trade: float = 0.005) -> list[dict[str, Any]]:
    """What this equity can hold, per micro contract and horizon, at this risk
    per trade: the number of contracts, or the minimum equity that would allow one."""
    import sys
    sys.path.append(str(ROOT / "scripts"))
    from capital_ladder import STOPS
    from core.contracts import MICRO_UNIVERSE
    from risk.sizing import minimum_viable_equity, size_position

    rows = []
    for (root, horizon), stop in STOPS.items():
        spec = MICRO_UNIVERSE[root].to_spec()
        r = size_position(spec, float(equity), float(risk_per_trade), stop)
        rows.append({"instrument": root, "horizon": horizon, "contracts": r.volume if r.tradeable else 0,
                     "min_equity_for_one": round(minimum_viable_equity(spec, float(risk_per_trade), stop))})
    return rows


def live_account() -> dict[str, Any]:
    """Read the broker account and open positions from the running MetaTrader
    terminal. Read-only: connects, reads, disconnects. Sends nothing."""
    from core.config import InstrumentConfig
    from execution.mt5_adapter import MT5Adapter

    inst = InstrumentConfig.load()
    live = MT5Adapter(aliases=inst.aliases)
    live.connect()
    try:
        a = live.account()
        positions = [{
            "symbol": p.symbol, "side": p.side.name, "volume": p.volume, "entry_price": p.entry_price,
            "stop_loss": p.stop_loss, "ticket": p.ticket, "comment": p.comment,
        } for p in live.positions()]
        return {
            "equity": a.equity, "balance": a.balance, "currency": a.currency, "margin_level": a.margin_level,
            "clock": str(live.clock_status.value) if live.clock_status else "unknown",
            "positions": positions,
        }
    finally:
        live.disconnect()
