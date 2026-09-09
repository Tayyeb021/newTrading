"""The MetaTrader watchdog, and the one decision it must not get wrong.

Written after the night of 2026-09-08. The shadow week traded properly for the
first time that evening and by 08:27 the next morning it had been halted for
hours: `terminal64.exe` was gone from the process list and nothing on the
machine restarted it. IB Gateway had had a watchdog since Sunday; MetaTrader
had none.

The decision that matters is *gone* versus *up but not serving*. A missing
process should be relaunched. A process that is up but refusing the API is a
terminal at a login prompt or one whose broker link dropped, and relaunching
that throws away a session that often recovers on its own - so it must report
and do nothing. Getting these two the same way round is the whole job.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("mt5_watchdog", ROOT / "scripts" / "mt5_watchdog.py")
wd = importlib.util.module_from_spec(_spec)
sys.modules["mt5_watchdog"] = wd
_spec.loader.exec_module(wd)


@pytest.fixture(autouse=True)
def _isolate_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "JOURNAL", tmp_path / "mt5_watchdog.jsonl")
    return tmp_path / "mt5_watchdog.jsonl"


def _run(monkeypatch, state, detail="because", argv=("mt5_watchdog.py",)):
    monkeypatch.setattr(wd, "probe", lambda: (state, detail))
    monkeypatch.setattr(sys, "argv", list(argv))
    return wd.main()


def test_a_missing_terminal_is_relaunched(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(wd, "relaunch", lambda: (calls.append(1) or (True, "launched")))
    assert _run(monkeypatch, wd.GONE) == 1
    assert calls == [1]
    assert "relaunched" in capsys.readouterr().out


def test_a_logged_out_terminal_is_NOT_relaunched(monkeypatch, capsys):
    """The decision this file exists for. Restarting a terminal that is up but
    refusing the API discards a session that may come back on its own, and
    cannot supply the login it is missing either way."""
    monkeypatch.setattr(wd, "relaunch", lambda: pytest.fail("must not relaunch a live process"))
    assert _run(monkeypatch, wd.NOT_SERVING, "logged out") == 1
    out = capsys.readouterr().out
    assert "Only a person can fix this" in out
    assert "Algo Trading" in out, "the other common cause must be named too"


def test_a_serving_terminal_is_left_completely_alone(monkeypatch, capsys, _isolate_journal):
    monkeypatch.setattr(wd, "relaunch", lambda: pytest.fail("nothing to fix"))
    assert _run(monkeypatch, wd.SERVING, "100,000.00 USD") == 0
    assert not _isolate_journal.exists(), "a quiet night must leave no journal"
    assert "serving" in capsys.readouterr().out


def test_check_mode_reports_without_touching_anything(monkeypatch, capsys):
    monkeypatch.setattr(wd, "relaunch", lambda: pytest.fail("--check must change nothing"))
    assert _run(monkeypatch, wd.GONE, argv=("mt5_watchdog.py", "--check")) == 1
    assert "gone" in capsys.readouterr().out


def test_a_failed_relaunch_is_reported_not_raised(monkeypatch, capsys):
    """A watchdog that dies on a bad launch stops watching."""
    monkeypatch.setattr(wd, "relaunch", lambda: (False, "not found"))
    assert _run(monkeypatch, wd.GONE) == 1
    assert "relaunch FAILED" in capsys.readouterr().out


def test_only_failures_are_journalled(monkeypatch, _isolate_journal):
    monkeypatch.setattr(wd, "relaunch", lambda: (True, "launched"))
    _run(monkeypatch, wd.GONE)
    _run(monkeypatch, wd.NOT_SERVING)
    monkeypatch.setattr(wd, "relaunch", lambda: (True, "unused"))
    _run(monkeypatch, wd.SERVING)

    rows = [json.loads(l) for l in _isolate_journal.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["state"] for r in rows] == [wd.GONE, wd.NOT_SERVING]


def test_the_exit_code_says_whether_a_human_is_needed(monkeypatch):
    monkeypatch.setattr(wd, "relaunch", lambda: (True, "launched"))
    assert _run(monkeypatch, wd.SERVING) == 0
    assert _run(monkeypatch, wd.GONE) == 1
    assert _run(monkeypatch, wd.NOT_SERVING) == 1


def test_an_env_override_wins_over_the_standard_install(monkeypatch, tmp_path):
    """A non-standard MetaTrader install must not need a code change."""
    fake = tmp_path / "terminal64.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv("MT5_TERMINAL", str(fake))
    monkeypatch.setattr(wd, "CANDIDATES", [Path(str(fake)), Path(r"C:\nope\terminal64.exe")])
    assert wd.terminal_path() == fake


def test_a_missing_terminal_binary_is_named_rather_than_crashing(monkeypatch):
    monkeypatch.setattr(wd, "CANDIDATES", [Path(r"C:\nowhere\terminal64.exe")])
    assert wd.terminal_path() is None
    ok, note = wd.relaunch()
    assert ok is False
    assert "MT5_TERMINAL" in note, "the note must say how to fix it"


def test_it_finds_the_terminal_on_this_machine():
    """The integration guard: if this returns None the watchdog can detect a
    dead terminal and never revive one."""
    p = wd.terminal_path()
    if p is None:
        pytest.skip("MetaTrader is not installed on this machine")
    assert p.name == "terminal64.exe" and p.exists()
