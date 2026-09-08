"""Per-market accounting, and the market list that made it answerable.

"Which instruments does this work on" turned out to be the question the whole
research had not asked. A sector total of +$15M is compatible with five markets
earning three each and with one earning fifteen while four lose, and the
difference decided whether the forward record was trading the edge or trading
around it. These tests guard the two pieces of machinery that answered it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.contracts import CORE_UNIVERSE, FULL_UNIVERSE, MICRO_OF, PARENT_OF, tradeable  # noqa: E402


class _Trade:
    def __init__(self, symbol, net, gross, costs, sleeve="s"):
        self.symbol, self.net_pnl, self.gross_pnl, self.costs = symbol, net, gross, costs
        self.sleeve = sleeve


def _markets(trades):
    """The accounting block from futures_gauntlet.evaluate, in isolation."""
    markets = {}
    for t in trades:
        m = markets.setdefault(t.symbol, {"net_pnl": 0.0, "gross_pnl": 0.0, "friction": 0.0,
                                          "trades": 0, "wins": 0,
                                          "bucket": FULL_UNIVERSE[t.symbol].bucket})
        m["net_pnl"] += t.net_pnl
        m["gross_pnl"] += t.gross_pnl
        m["friction"] += t.costs
        m["trades"] += 1
        m["wins"] += 1 if t.net_pnl > 0 else 0
    return markets


def test_per_market_totals_reconcile_with_the_book():
    """If the per-market net does not add up to the book's net, the table is
    decoration. Every number reported to the user comes off this sum."""
    trades = [_Trade("ES", 100.0, 110.0, 10.0), _Trade("ES", -40.0, -35.0, 5.0),
              _Trade("ZT", 7770.0, 8170.0, 400.0), _Trade("GC", 50.0, 55.0, 5.0)]
    m = _markets(trades)
    assert sum(v["net_pnl"] for v in m.values()) == pytest.approx(sum(t.net_pnl for t in trades))
    assert sum(v["gross_pnl"] for v in m.values()) == pytest.approx(sum(t.gross_pnl for t in trades))
    assert sum(v["friction"] for v in m.values()) == pytest.approx(sum(t.costs for t in trades))
    assert m["ES"]["trades"] == 2 and m["ES"]["wins"] == 1
    assert m["ZT"]["bucket"] == "rates"


def test_a_market_with_no_trades_is_absent_rather_than_zero():
    """A market that never traded and a market that traded to exactly zero are
    different facts, and only one of them means the signal never fired."""
    m = _markets([_Trade("ES", 1.0, 1.0, 0.0)])
    assert "GC" not in m


def test_the_forward_universe_is_exactly_the_markets_that_have_a_micro():
    """The defence of the forward record's 13 markets is that they were chosen
    by contract availability, not by performance. That is only a defence if it
    is true, so it is asserted rather than assumed."""
    import json
    manifest = ROOT / "state" / "forward_manifest.json"
    if not manifest.exists():
        pytest.skip("no forward record running")
    traded = json.loads(manifest.read_text(encoding="utf-8"))["universe"]
    parents = {PARENT_OF.get(m, m) for m in traded}

    have_micro = {r for r in CORE_UNIVERSE if r in MICRO_OF}
    researched = parents & set(CORE_UNIVERSE)

    # Every researched market that is traded must have a micro, or ZN which is
    # traded full-size because it has none and is the only rates market small
    # enough to hold.
    for root in researched:
        assert root in have_micro or root == "ZN", (
            f"{root} is traded forward but has no micro - that is a choice, not a constraint")

    # And no market with a micro was silently left out.
    missing = have_micro - parents
    assert not missing, f"markets with a micro that are NOT traded forward: {sorted(missing)}"


def test_tradeable_returns_the_micro_when_one_exists():
    assert tradeable("ES").root == "MES"
    assert tradeable("ZT").root == "ZT", "no micro exists, so the full-size contract stands"


def _resolve(requested):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fg", ROOT / "research" / "futures_gauntlet.py")
    fg = importlib.util.module_from_spec(spec)
    sys.modules["fg"] = fg
    spec.loader.exec_module(fg)
    return fg.resolve_markets(requested)


def test_market_list_maps_micros_to_their_parent_history():
    """--markets MES must research ES: the micro has the same price and a much
    shorter history, and researching the micro would silently shorten the
    sample without anything failing."""
    wanted, absent = _resolve(["MES", "MGC", "ZN"])
    assert wanted == ["ES", "GC", "ZN"]
    assert not absent


def test_a_root_with_no_history_is_reported_not_dropped_in_silence():
    wanted, absent = _resolve(["MES", "NOSUCHROOT"])
    assert wanted == ["ES"]
    assert absent == ["NOSUCHROOT"], "a typo must be visible, not a quietly smaller universe"


def test_duplicates_collapse_so_a_market_is_not_double_weighted():
    """MES and ES name one market. Passing both must not run it twice, which
    would double its weight in the book and quietly change the answer."""
    wanted, _ = _resolve(["MES", "ES", "MGC"])
    assert wanted == ["ES", "GC"]
