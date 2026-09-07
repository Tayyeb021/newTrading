"""Cost calibration, and the demo-fill trap it exists to refuse.

On 2026-09-07 four real round trips on the IC Markets demo came back with
slippage of exactly zero on all eight legs, across instruments quoting from 0
to 120 points, one of them on a six-second-old index quote. A demo server has
no liquidity to consume, so it fills at the quote. Writing that into the cost
model would halve modelled friction, and entry 007 died on friction at 295% of
gross - so it would revive strategies that are dead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from calibrate_costs import (  # noqa: E402
    MIN_CREDIBLE_SLIP_RATIO, quoted_spreads, slippage_is_credible,
)
from verify_roundtrip import record_fill  # noqa: E402


def test_zero_slippage_is_refused_however_many_fills():
    ok, why = slippage_is_credible([0.0] * 8, spread=1.2)
    assert not ok and "filled at quote" in why


def test_implausibly_small_slippage_is_refused():
    ok, why = slippage_is_credible([0.0001], spread=1.2)  # a ten-thousandth of the spread
    assert not ok and "implausibly good" in why


def test_realistic_slippage_is_accepted():
    ok, why = slippage_is_credible([0.5, 0.7, 0.6], spread=1.2)
    assert ok and "0.50 of the spread" in why


def test_a_zero_spread_makes_the_ratio_meaningless():
    ok, why = slippage_is_credible([0.0], spread=0.0)
    assert not ok and "spread was zero" in why


def test_threshold_sits_below_a_real_half_spread():
    """Half the spread is the standard retail assumption; the guard must not eat it."""
    assert 0.0 < MIN_CREDIBLE_SLIP_RATIO < 0.5
    assert slippage_is_credible([0.5], spread=1.0)[0]


def test_record_fill_writes_both_legs_and_appends(tmp_path):
    path = tmp_path / "fills.json"
    record_fill("EURUSD", entry_spread=0.00012, entry_slip=0.00003, exit_slip=-0.00002, path=path)
    record_fill("EURUSD", entry_spread=0.00014, entry_slip=0.00005, exit_slip=0.00001, path=path)
    record_fill("XAUUSD", entry_spread=0.08, entry_slip=0.02, exit_slip=0.03, path=path)

    store = json.loads(path.read_text(encoding="utf-8"))
    assert len(store["EURUSD"]) == 4 and len(store["XAUUSD"]) == 2
    assert [f["leg"] for f in store["EURUSD"]] == ["entry", "exit", "entry", "exit"]
    assert store["EURUSD"][1]["slippage"] == -0.00002  # the exit's own number, kept signed
    assert all("ts" in f for f in store["EURUSD"])


def test_corrupt_fill_store_does_not_lose_the_next_write(tmp_path):
    path = tmp_path / "fills.json"
    path.write_text("{not json", encoding="utf-8")
    record_fill("EURUSD", 0.0001, 0.00002, 0.00003, path=path)
    assert len(json.loads(path.read_text(encoding="utf-8"))["EURUSD"]) == 2


def test_quoted_spreads_reads_the_profiler_and_skips_bad_lines(tmp_path):
    path = tmp_path / "samples.jsonl"
    good = [{"symbol": "US30", "spread": 1.2}, {"symbol": "US30", "spread": 0.9},
            {"symbol": "EURUSD", "spread": 0.00012}]
    path.write_text("\n".join(json.dumps(r) for r in good) + "\n{truncated\n", encoding="utf-8")
    out = quoted_spreads(path)
    assert out["US30"] == [1.2, 0.9] and out["EURUSD"] == [0.00012]
    assert quoted_spreads(tmp_path / "absent.jsonl") == {}
