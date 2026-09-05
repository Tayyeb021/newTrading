"""Broker server time to UTC.

MT5 hands you epochs from the **server's** clock, not UTC, and the Python package
does nothing to tell you which. Read them as UTC and every timestamp in the system
is silently wrong by two or three hours: session filters fire at the wrong hour,
the swap rollover is counted at the wrong instant, and a strategy that looks
sensible in the backtest trades a different market live.

Nothing errors. That is what makes it dangerous.

## What this broker does, and how it was established

Most MT5 servers set **midnight = 17:00 New York**, the equity close, so the
trading week produces five clean daily bars. That makes the server offset:

    server = New York time + 7 hours
           = UTC+3 during US daylight saving
           = UTC+2 otherwise

Deduced from the data rather than assumed. The US cash open is the sharpest
recurring event in the dataset, and on US30 M15 it sits at **16:30 server time in
both summer and winter**. It can only stay fixed across the DST boundary if the
server's own clock shifts with US DST:

    summer  16:30 server = 13:30 UTC  ->  UTC+3
    winter  16:30 server = 14:30 UTC  ->  UTC+2

Cross-checked against a live tick: server 13:18 minus 7h is 06:18 New York, which
was 10:18 UTC. Matches.

`verify_offset()` re-checks this against the live terminal at startup, because a
broker changing its server timezone is exactly the kind of silent change that
should stop the system rather than quietly corrupt a year of research.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from zoneinfo import ZoneInfo

#: The exchange whose clock the server is pinned to.
BROKER_ANCHOR_TZ = ZoneInfo("America/New_York")

#: Hours the server clock runs ahead of the anchor. 7 puts server midnight at the
#: 17:00 New York close.
BROKER_ANCHOR_OFFSET_HOURS = 7


def server_offset_hours(server_naive: datetime) -> int:
    """Hours the server clock runs ahead of UTC at a given server-clock reading.

    Returns 3 during US daylight saving and 2 outside it.
    """
    # Resolve the instant approximately first, then read the true anchor offset.
    # An hour of error here cannot change the answer, because the candidate
    # offsets are 2 and 3 and the DST boundary falls at 02:00 local.
    approx = (server_naive - timedelta(hours=3)).replace(tzinfo=timezone.utc)
    anchor_offset = approx.astimezone(BROKER_ANCHOR_TZ).utcoffset()
    assert anchor_offset is not None
    return BROKER_ANCHOR_OFFSET_HOURS + int(anchor_offset.total_seconds() // 3600)


def server_epoch_to_utc(epoch: int | float) -> datetime:
    """Convert one MT5 server epoch to a true, timezone-aware UTC datetime."""
    # MT5 epochs are the server's wall clock encoded as if it were UTC.
    server_naive = datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None)
    offset = server_offset_hours(server_naive)
    return (server_naive - timedelta(hours=offset)).replace(tzinfo=timezone.utc)


def utc_to_server_naive(moment: datetime) -> datetime:
    """Inverse: a true UTC instant to the server's wall clock, naive.

    Needed whenever a UTC boundary has to be handed back to the terminal.
    """
    utc = moment.astimezone(timezone.utc)
    anchor_offset = utc.astimezone(BROKER_ANCHOR_TZ).utcoffset()
    assert anchor_offset is not None
    offset = BROKER_ANCHOR_OFFSET_HOURS + int(anchor_offset.total_seconds() // 3600)
    return (utc + timedelta(hours=offset)).replace(tzinfo=None)


class ClockCheck(Enum):
    VERIFIED = "verified"
    MISMATCH = "mismatch"
    UNVERIFIABLE = "unverifiable"


def verify_offset(
    server_epoch: int | float,
    tolerance_minutes: int = 15,
    stale_after_minutes: int = 30,
) -> tuple[ClockCheck, str]:
    """Check the assumed offset against the live terminal.

    Three outcomes, and the third one matters. A stale tick and a changed server
    timezone look identical from a single reading: both put the converted time
    far from now. Treating "market closed" as a timezone error would block every
    connect over a weekend, which is exactly what the first version did.

    So a tick older than `stale_after_minutes` yields UNVERIFIABLE rather than
    MISMATCH. The caller warns and continues; only a *fresh* tick that lands in
    the wrong place is treated as a real timezone change and stops the system.
    """
    converted = server_epoch_to_utc(server_epoch)
    now = datetime.now(timezone.utc)
    drift_minutes = (now - converted).total_seconds() / 60.0

    server_naive = datetime.fromtimestamp(server_epoch, tz=timezone.utc).replace(tzinfo=None)
    assumed = server_offset_hours(server_naive)

    if abs(drift_minutes) <= tolerance_minutes:
        return ClockCheck.VERIFIED, (
            f"server clock is UTC+{assumed} as expected (drift {drift_minutes:+.1f} min)"
        )

    # A quote from the past is an old quote, not a clock change. A quote from the
    # FUTURE cannot be staleness, so that stays a mismatch.
    if drift_minutes > stale_after_minutes:
        return ClockCheck.UNVERIFIABLE, (
            f"cannot verify the server clock: last tick is {drift_minutes / 60:.1f}h old "
            f"({converted:%a %H:%M} UTC) - the market is closed. Assuming UTC+{assumed}. "
            f"Re-check when trading resumes."
        )

    actual = (server_naive - now.replace(tzinfo=None)).total_seconds() / 3600.0
    return ClockCheck.MISMATCH, (
        f"SERVER TIMEZONE MISMATCH: assumed UTC+{assumed}, measured UTC{actual:+.1f}. "
        f"Every stored timestamp would be wrong by {actual - assumed:+.1f}h. "
        f"Fix BROKER_ANCHOR_OFFSET_HOURS in execution/brokertime.py and re-download."
    )
