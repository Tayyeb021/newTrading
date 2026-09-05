"""The MCP tool functions: read-only views, tested without the MCP runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ops import mcp_tools  # noqa: E402
from ops.journal import Journal  # noqa: E402


def _fake_state(tmp_path, monkeypatch):
    """Point the tools at a temporary project root with a small journal."""
    (tmp_path / "state").mkdir()
    j = Journal(tmp_path / "state" / "journal.jsonl")
    j.write("startup", notes=["connected"], equity=100_000.0)
    j.write("decision", symbol="EURUSD", strategy="tsmom", side="BUY", approved=False, note="trade-level limit",
            breaches=[{"limit": "sleeve_budget", "severity": "reject", "message": "", "observed": 1, "threshold": 1}])
    j.write("decision", symbol="XAUUSD", strategy="tsmom", side="SELL", approved=True, note="", breaches=[],
            volume=0.5, risk_fraction=0.005)
    j.heartbeat(equity=100_050.0, positions=1, halted=False, feed_age_s=3.0)
    (tmp_path / "state" / "gauntlet_010.json").write_text(json.dumps({
        "book": {"net_sharpe": 0.28, "max_drawdown": 0.47}, "passed": False,
        "verdict": [{"test": "1. sharpe", "pass": False, "detail": "0.28"}]}))
    (tmp_path / "RESEARCH_LOG.md").write_text(
        "# log\n\n## 009 — carry\nbody nine\n\n## 010 — trend\nbody ten\n\n## 010 — VERDICT\nverdict ten\n",
        encoding="utf-8")
    monkeypatch.setattr(mcp_tools, "ROOT", tmp_path)
    monkeypatch.setattr(mcp_tools, "STATE", tmp_path / "state")
    return j


def test_status_reads_the_last_heartbeat_and_kill_switch(tmp_path, monkeypatch):
    _fake_state(tmp_path, monkeypatch)
    s = mcp_tools.status()
    assert s["last_heartbeat"]["equity"] == 100_050.0 and s["halted"] is False
    assert s["running"] is True and s["kill_switch_engaged"] is False
    assert s["recent_event_counts"]["decision"] == 2 and s["session"] is None


def test_decisions_filter_and_name_the_refusing_limit(tmp_path, monkeypatch):
    _fake_state(tmp_path, monkeypatch)
    refused = mcp_tools.decisions(approved=False)
    assert len(refused) == 1 and refused[0]["refused_by"] == ["sleeve_budget"]
    ok = mcp_tools.decisions(approved=True)
    assert ok[0]["symbol"] == "XAUUSD" and ok[0]["volume"] == 0.5
    assert len(mcp_tools.journal("decision")) == 2 and len(mcp_tools.journal(n=1)) == 1


def test_verdicts_and_research_log_sections(tmp_path, monkeypatch):
    _fake_state(tmp_path, monkeypatch)
    v = mcp_tools.verdicts()
    assert v["gauntlet_010"]["passed"] is False and v["gauntlet_010"]["net_sharpe"] == 0.28
    text = mcp_tools.research_log("010")
    assert "body ten" in text and "verdict ten" in text and "body nine" not in text
    assert "no entry" in mcp_tools.research_log("999")
    assert "body nine" in mcp_tools.research_log()


def test_capital_ladder_scales_with_equity():
    small = {(r["instrument"], r["horizon"]): r for r in mcp_tools.capital_ladder(5_000)}
    large = {(r["instrument"], r["horizon"]): r for r in mcp_tools.capital_ladder(250_000)}
    assert small[("MES", "daily")]["contracts"] == 0 and large[("MES", "daily")]["contracts"] >= 1
    assert small[("M6E", "intraday")]["min_equity_for_one"] == large[("M6E", "intraday")]["min_equity_for_one"]


def test_no_tool_can_reach_an_execution_write():
    """The property that makes the server safe to hand to an assistant."""
    source = Path(mcp_tools.__file__).read_text(encoding="utf-8")
    for forbidden in (".submit(", ".close(", ".modify(", "close_all", "engage(", "clear(", "order_send"):
        assert forbidden not in source, f"mcp_tools must never call {forbidden}"
