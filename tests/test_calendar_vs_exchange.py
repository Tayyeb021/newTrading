"""Every expiry rule against the exchange's own definition records.

`data/futures/_defs/GLBX_*.parquet` are definition snapshots bought from
Databento (a few cents each). They carry the exchange's `expiration` per
contract. The calendar is business-day only and knows no holidays, so a rule
is allowed to miss by at most two business days on any contract (Juneteenth,
Christmas, a Brazilian bank holiday) and must be exact on most of them. Skipped
when no snapshot is on disk, because the data is not in the repository.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from core.contracts import ALL_ROOTS, CODE_MONTHS, _add_busdays  # noqa: E402
from download_databento import contract_year  # noqa: E402

SNAPSHOTS = sorted((ROOT / "data" / "futures" / "_defs").glob("GLBX_*.parquet"))
OUTRIGHT = re.compile(r"^(?P<root>[A-Z0-9]{2,3}?)(?P<m>[FGHJKMNQUVXZ])(?P<yy>\d{1,2})$")


def _busdays_between(a: date, b: date) -> int:
    n, d, step = 0, a, 1 if b >= a else -1
    while d != b:
        d = _add_busdays(d, step)
        n += 1
    return n


@pytest.mark.skipif(not SNAPSHOTS, reason="no exchange definition snapshot on disk")
def test_rules_agree_with_exchange_expirations():
    frames = [pd.read_parquet(p) for p in SNAPSHOTS]
    d = pd.concat(frames)
    d = d[d["instrument_class"] == "F"].copy()
    d["exp"] = pd.to_datetime(d["expiration"], errors="coerce", utc=True).dt.date
    d["day"] = pd.to_datetime(d["ts_recv"] if "ts_recv" in d else d["ts_event"], utc=True).dt.date
    checked: dict[str, list[int]] = {}
    for _, row in d.dropna(subset=["exp"]).iterrows():
        sym = str(row["raw_symbol"])
        for root in sorted(ALL_ROOTS, key=len, reverse=True):
            if sym.startswith(root) and len(sym) > len(root):
                rest = sym[len(root):]
                if rest[0] in CODE_MONTHS and rest[1:].isdigit():
                    break
        else:
            continue
        month, digits = CODE_MONTHS[rest[0]], rest[1:]
        r = ALL_ROOTS[root]
        if month not in r.months:
            continue  # a serial month the universe does not trade
        year = contract_year(r, month, digits, row["day"])
        predicted = r.last_trade(year, month)
        miss = _busdays_between(predicted, row["exp"])
        checked.setdefault(root, []).append(miss)

    assert checked, "no contracts matched any root"
    report = []
    for root, misses in sorted(checked.items()):
        exact = sum(1 for m in misses if m == 0) / len(misses)
        worst = max(misses)
        report.append(f"{root}: {len(misses)} contracts, exact {exact:.0%}, worst miss {worst} bd")
        assert worst <= 2, f"{root}: a rule is wrong, not a holiday: worst miss {worst} business days"
        if len(misses) >= 5:  # on three contracts two holidays are a majority; on many they are not
            assert exact >= 0.6, f"{root}: only {exact:.0%} exact; the rule is misaligned"
    print("\n".join(report))
