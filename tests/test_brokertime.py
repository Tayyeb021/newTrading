"""Tests for broker server time conversion.

This bug shipped once already: MT5 epochs were read as UTC when they are the
server's wall clock, putting every stored timestamp 3 hours out. Nothing errored.
Session filters fired at the wrong hour and swap rollovers were counted at the
wrong instant, and it was only found by auditing the data against a known market
event.

These tests exist so it cannot happen silently again.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.brokertime import (  # noqa: E402
    server_epoch_to_utc,
    server_offset_hours,
    utc_to_server_naive,
    verify_offset,
)


def server_epoch(y, m, d, hh, mm=0) -> float:
    """An epoch as MT5 produces it: the server's wall clock encoded as UTC."""
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp()


def test_offset_is_three_during_us_dst():
    assert server_offset_hours(datetime(2026, 7, 15, 12)) == 3
    assert server_offset_hours(datetime(2026, 9, 4, 13)) == 3


def test_offset_is_two_outside_us_dst():
    assert server_offset_hours(datetime(2026, 1, 15, 12)) == 2
    assert server_offset_hours(datetime(2026, 12, 20, 12)) == 2


def test_us_cash_open_lands_at_the_right_utc_time_year_round():
    """The anchor the offset was derived from, asserted in both directions.

    The US equity open is 09:30 New York: 13:30 UTC in summer, 14:30 in winter.
    On this broker it sits at 16:30 server time all year. If the conversion is
    right, 16:30 server maps to each of those in the correct season.
    """
    summer = server_epoch_to_utc(server_epoch(2026, 7, 15, 16, 30))
    assert (summer.hour, summer.minute) == (13, 30), "summer cash open is wrong"

    winter = server_epoch_to_utc(server_epoch(2026, 1, 15, 16, 30))
    assert (winter.hour, winter.minute) == (14, 30), "winter cash open is wrong"


def test_server_midnight_is_the_new_york_close():
    """Server midnight = 17:00 New York. That is why the offset is what it is."""
    from zoneinfo import ZoneInfo

    for month in (1, 7):
        utc = server_epoch_to_utc(server_epoch(2026, month, 15, 0, 0))
        ny = utc.astimezone(ZoneInfo("America/New_York"))
        assert ny.hour == 17, f"month {month}: server midnight is {ny.hour}:00 NY, expected 17:00"


def test_round_trip_is_lossless():
    for moment in (
        datetime(2026, 3, 2, 8, 15, tzinfo=timezone.utc),
        datetime(2026, 7, 20, 21, 45, tzinfo=timezone.utc),
        datetime(2026, 11, 30, 0, 5, tzinfo=timezone.utc),
    ):
        naive = utc_to_server_naive(moment)
        back = server_epoch_to_utc(naive.replace(tzinfo=timezone.utc).timestamp())
        assert back == moment, f"round trip lost {moment}"


def test_conversion_is_never_a_no_op():
    """Regression: the original bug was reading the epoch straight through."""
    epoch = server_epoch(2026, 7, 15, 12, 0)
    naive_read = datetime.fromtimestamp(epoch, tz=timezone.utc)
    assert server_epoch_to_utc(epoch) != naive_read, "conversion did nothing"
    assert (naive_read - server_epoch_to_utc(epoch)) == timedelta(hours=3)


def test_all_conversions_are_timezone_aware():
    ts = server_epoch_to_utc(server_epoch(2026, 5, 1, 10))
    assert ts.tzinfo is not None and ts.utcoffset() == timedelta(0)


def test_verify_offset_accepts_a_correct_clock():
    now = datetime.now(timezone.utc)
    offset = server_offset_hours(now.replace(tzinfo=None))
    fake = (now + timedelta(hours=offset)).replace(tzinfo=timezone.utc).timestamp()
    ok, message = verify_offset(fake)
    assert ok, message


def test_verify_offset_rejects_a_changed_clock():
    """A broker moving its server timezone must stop the system, not corrupt it."""
    now = datetime.now(timezone.utc)
    wrong = (now + timedelta(hours=1)).replace(tzinfo=timezone.utc).timestamp()
    ok, message = verify_offset(wrong)
    assert not ok
    assert "MISMATCH" in message
