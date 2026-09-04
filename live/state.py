"""Durable session state, and the restart bug that kills evaluation accounts.

Positions are the broker's truth and are always re-read from it. But some state
has no broker equivalent and cannot be rebuilt from one:

- `day_start_equity` — the broker does not know when *your* trading day began
- `high_water_equity` — needs history the terminal will not give you
- `starting_equity` — the evaluation's opening balance
- consecutive losses per strategy

**The failure this module exists to prevent.** Restart the process at 14:00 after
a morning that already lost 3%. If `day_start_equity` is re-initialised from
current equity, the daily-loss limit silently re-arms against the *lower* base,
and the system will happily lose another 3.5% on a day it had already spent its
budget. Nothing errors. The equity curve simply falls through a limit that was
supposed to be uncrossable — and on a prop evaluation that is the account.

So: risk bookkeeping is persisted here, positions come from the broker, and
`reconcile_on_start` cross-checks the two before trading resumes.

Writes are atomic (temp file plus rename) because a crash *during* the write of
the file that protects you from crashes would be a poor joke.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from risk.engine import SessionBook

SCHEMA_VERSION = 1


@dataclass
class PersistedState:
    session_date: str
    starting_equity: float
    day_start_equity: float
    high_water_equity: float
    consecutive_losses: dict[str, int] = field(default_factory=dict)
    last_loss_ts: dict[str, str] = field(default_factory=dict)
    killed: bool = False
    kill_reason: str = ""
    updated_at: str = ""
    schema: int = SCHEMA_VERSION

    # ------------------------------------------------------------------ codec

    @classmethod
    def from_book(cls, book: SessionBook) -> "PersistedState":
        return cls(
            session_date=book.trading_day.isoformat(),
            starting_equity=book.starting_equity,
            day_start_equity=book.day_start_equity,
            high_water_equity=book.high_water_equity,
            consecutive_losses=dict(book.consecutive_losses),
            last_loss_ts={k: v.isoformat() for k, v in book.last_loss_ts.items()},
            killed=book.killed,
            kill_reason=book.kill_reason,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_book(self) -> SessionBook:
        book = SessionBook(
            starting_equity=self.starting_equity,
            day_start_equity=self.day_start_equity,
            high_water_equity=self.high_water_equity,
            trading_day=date.fromisoformat(self.session_date),
            consecutive_losses=dict(self.consecutive_losses),
            killed=self.killed,
            kill_reason=self.kill_reason,
        )
        book.last_loss_ts = {
            k: datetime.fromisoformat(v) for k, v in self.last_loss_ts.items()
        }
        return book


class StateStore:
    def __init__(self, path: str | Path = "state/session.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, book: SessionBook) -> None:
        """Atomic write. A half-written state file is worse than none."""
        state = PersistedState.from_book(book)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(asdict(state), fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def load(self) -> PersistedState | None:
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if raw.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"{self.path}: state schema {raw.get('schema')} != {SCHEMA_VERSION}. "
                f"Migrate or delete it deliberately - do not let the system guess."
            )
        return PersistedState(**raw)


def restore_book(
    store: StateStore,
    current_equity: float,
    today: date | None = None,
) -> tuple[SessionBook, list[str]]:
    """Rebuild the session book across a restart. Returns (book, notes).

    Three cases, and the middle one is the dangerous one:

    - **No saved state.** First run. Open a fresh book at current equity.
    - **Saved state from today.** Resume it. `day_start_equity` is carried
      forward *unchanged*, which is the entire point of this module.
    - **Saved state from an earlier day.** Roll the day: `day_start_equity`
      becomes current equity, but `starting_equity` and `high_water_equity`
      survive, because max-drawdown limits span the whole evaluation.
    """
    today = today or datetime.now(timezone.utc).date()
    saved = store.load()
    notes: list[str] = []

    if saved is None:
        notes.append(f"no saved state; opening a fresh session at {current_equity:,.2f}")
        return SessionBook.open(current_equity, today), notes

    book = saved.to_book()
    saved_day = date.fromisoformat(saved.session_date)

    if saved_day == today:
        spent = (book.day_start_equity - current_equity) / book.day_start_equity
        notes.append(
            f"resumed today's session: day started at {book.day_start_equity:,.2f}, "
            f"{spent:+.2%} used of the daily budget"
        )
        if spent > 0:
            notes.append(
                "daily loss already incurred is CARRIED FORWARD - the limit was not reset"
            )
    else:
        notes.append(f"new trading day (saved {saved_day}, now {today}); rolling day_start")
        book.trading_day = today
        book.day_start_equity = current_equity

    book.high_water_equity = max(book.high_water_equity, current_equity)
    dd = (book.high_water_equity - current_equity) / book.high_water_equity
    notes.append(
        f"carried forward: starting {book.starting_equity:,.2f}, "
        f"high-water {book.high_water_equity:,.2f}, drawdown {dd:.2%}"
    )
    if book.killed:
        notes.append(f"KILL SWITCH IS STILL ENGAGED: {book.kill_reason}")
    return book, notes
