"""The daily shadow digest reads the journal the runner actually writes."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ops.journal import Journal
from shadow_report import digest


def _journal(tmp_path, now):
    j = Journal(tmp_path / "j.jsonl")
    t0 = now - timedelta(hours=3)
    # heartbeats every 10s would be thousands; a handful with one 20-minute hole is enough
    for k, minutes in enumerate([0, 1, 2, 22, 23, 24]):
        rec = j.write("heartbeat", equity=100_000 - k * 10, positions=1 if k > 3 else 0,
                      halted=(k == 0), feed_age_s=5.0 if k else 900.0)
        rec["ts"] = (t0 + timedelta(minutes=minutes)).isoformat()
        _rewrite(j, k, rec)
    j.write("decision", symbol="EURUSD", strategy="mtf_pullback", side="BUY", approved=False,
            note="risk limit", breaches=[{"limit": "SpreadGuard", "severity": "reject",
                                          "message": "", "observed": 3.0, "threshold": 2.0}])
    j.write("decision", symbol="XAUUSD", strategy="mtf_pullback", side="SELL", approved=True,
            note="", breaches=[])
    j.write("fill", client_id="mtf_pullback#abc", status="filled", symbol="XAUUSD", side="SELL",
            ticket=1, requested=2400.0, filled=2399.9, volume=0.01, slippage=-0.1, reason="")
    j.write("loop_error", error="RuntimeError: Call failed", consecutive=1)
    return j


def _rewrite(journal, index, record):
    """The journal stamps `ts` itself; backdate one line so the window logic is exercised."""
    import json
    lines = journal.path.read_text(encoding="utf-8").splitlines()
    lines[index] = json.dumps(record)
    journal.path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_digest_reports_uptime_gaps_decisions_and_fills(tmp_path):
    now = datetime.now(timezone.utc)
    text = digest(_journal(tmp_path, now), hours=24)
    assert "6 heartbeats" in text
    assert "halted on 1 of them" in text
    assert "GAPS in heartbeats" in text and "20min" in text
    assert "equity: start 100,000.00  end 99,950.00" in text
    assert "feed age: median 5s  max 900s" in text
    assert "decisions: 2 (1 approved, 1 refused); refused by SpreadGuard 1" in text
    assert "approved: mtf_pullback XAUUSD SELL x1" in text
    assert "paper fills: 1 of 1 attempts; median |slippage| 0.100000" in text
    assert "loop_error: 1 - last: RuntimeError: Call failed" in text
    assert "No order reached the broker" in text


def test_digest_window_excludes_old_entries(tmp_path):
    now = datetime.now(timezone.utc)
    j = _journal(tmp_path, now)
    text = digest(j, hours=1)  # the backdated heartbeats are 3h old
    assert "heartbeats" not in text
    assert "decisions: 2" in text  # the rest was written just now


def test_digest_on_empty_journal(tmp_path):
    text = digest(Journal(tmp_path / "none.jsonl"), hours=24)
    assert "no journal entries" in text
