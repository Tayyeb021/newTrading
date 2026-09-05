"""Turning a raw Databento download into one file per contract month.

Every case here was a real file. NQZ9 is December 2019 and December 2029, and
one stray print of the latter once overwrote the former. June 2024 gas began
printing under NGM4 within a year of June 2014 expiring, so a gap heuristic
glued the two. July 2025 gas is NGN25 for its whole life. Sunday UTC stubs
made 300 bars a year with a third of the range. WTI settled below zero.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from core.contracts import FULL_UNIVERSE  # noqa: E402
from data.store import DataError, validate  # noqa: E402
from download_databento import (  # noqa: E402
    contract_year, contracts_from_family, merge_sunday_stubs, outright_codes, write_contracts,
)

NQ, NG, ES, CL, GC = (FULL_UNIVERSE[r] for r in ("NQ", "NG", "ES", "CL", "GC"))


def _rows(symbol, days, level):
    ts = pd.to_datetime(days, utc=True)
    return pd.DataFrame({"symbol": symbol, "ts": ts, "open": level, "high": level + 1.0,
                         "low": level - 1.0, "close": level, "volume": 10.0})


def test_contract_year_follows_the_exchange_calendar():
    # NGM4: June 2014 expires 2014-05-28; a print in early May is 2014, one in July is 2024
    assert contract_year(NG, 6, "4", date(2014, 5, 2)) == 2014
    assert contract_year(NG, 6, "4", date(2014, 7, 15)) == 2024
    assert contract_year(NG, 6, "4", date(2018, 1, 3)) == 2024
    assert contract_year(NG, 6, "4", date(2010, 6, 7)) == 2014
    # two digits are literal
    assert contract_year(NG, 7, "25", date(2019, 6, 3)) == 2025
    assert contract_year(CL, 7, "31", date(2022, 1, 3)) == 2031
    # a bar a few days past a holiday-shifted last trade still belongs to that contract
    assert contract_year(ES, 12, "5", date(2025, 12, 22)) == 2025
    assert contract_year(ES, 12, "5", date(2026, 1, 15)) == 2035


def test_far_dated_print_does_not_overwrite_the_earlier_decade():
    raw = pd.concat([
        _rows("NQZ9", pd.bdate_range("2018-09-25", "2019-12-20"), 7500.0),   # December 2019
        _rows("NQZ9", ["2025-02-14"], 24600.0),                              # one print of December 2029
        _rows("NQZ0", pd.bdate_range("2010-06-07", "2010-12-17"), 1900.0),   # December 2010
        _rows("NQZ0", pd.bdate_range("2019-12-02", "2020-12-18"), 9000.0),   # December 2020
    ])
    got = {(y, m): seg for y, m, seg in contracts_from_family(raw, NQ)}
    assert set(got) == {(2019, 12), (2029, 12), (2010, 12), (2020, 12)}
    assert len(got[(2019, 12)]) == len(pd.bdate_range("2018-09-25", "2019-12-20"))
    assert len(got[(2029, 12)]) == 1 and got[(2029, 12)]["close"].iloc[0] == 24600.0
    assert got[(2010, 12)]["close"].iloc[0] == 1900.0 and got[(2020, 12)]["close"].iloc[0] == 9000.0


def test_next_decade_printing_soon_after_expiry_is_split_by_calendar_not_gap():
    raw = pd.concat([
        _rows("NGM4", pd.bdate_range("2013-01-02", "2014-05-28"), 4.5),   # June 2014 to its last trade
        _rows("NGM4", ["2014-11-03", "2015-03-02"], 3.9),                 # June 2024, sparse, within a year
        _rows("NGM4", pd.bdate_range("2022-01-03", "2024-05-29"), 2.8),   # June 2024, the real run
        _rows("NGN25", pd.bdate_range("2019-06-03", "2025-06-26"), 3.1),  # July 2025, two-digit for life
    ])
    got = {(y, m): seg for y, m, seg in contracts_from_family(raw, NG)}
    assert set(got) == {(2014, 6), (2024, 6), (2025, 7)}
    assert got[(2014, 6)]["ts"].max().date() == date(2014, 5, 28)
    assert len(got[(2024, 6)]) == 2 + len(pd.bdate_range("2022-01-03", "2024-05-29"))
    assert got[(2025, 7)]["close"].iloc[0] == 3.1


def test_spreads_and_options_are_ignored():
    raw = pd.concat([
        _rows("CLZ5", pd.bdate_range("2025-10-01", "2025-11-20"), 60.0),
        _rows("CLZ5-CLF6", pd.bdate_range("2025-10-01", "2025-11-20"), 0.5),
        _rows("CL:BF Z5-F6-G6", pd.bdate_range("2025-10-01", "2025-11-20"), 0.1),
        _rows("LOZ5 C6000", pd.bdate_range("2025-10-01", "2025-11-20"), 2.0),
    ])
    assert [(y, m) for y, m, _ in contracts_from_family(raw, CL)] == [(2025, 12)]


def test_sunday_stub_folds_into_monday():
    days = ["2024-11-01", "2024-11-03", "2024-11-04", "2024-11-05", "2024-11-10"]  # Fri, SUN, Mon, Tue, Sun (Mon holiday)
    bars = _rows("ESZ4", days, 5800.0)
    bars.loc[1, ["open", "high", "low", "close", "volume"]] = [5790.0, 5792.0, 5780.0, 5791.0, 3.0]   # the stub
    bars.loc[2, ["open", "high", "low", "close", "volume"]] = [5791.0, 5850.0, 5785.0, 5840.0, 100.0]  # Monday
    out = merge_sunday_stubs(bars.drop(columns=["symbol"]))
    assert [d.strftime("%a") for d in out["ts"]] == ["Fri", "Mon", "Tue", "Sun"]
    mon = out.iloc[1]
    assert mon["open"] == 5790.0 and mon["high"] == 5850.0 and mon["low"] == 5780.0
    assert mon["close"] == 5840.0 and mon["volume"] == 103.0
    assert out.iloc[3]["ts"].strftime("%a") == "Sun"  # a stub with no Monday after it is left alone


def test_negative_futures_prices_are_kept(tmp_path):
    days = pd.bdate_range("2020-03-20", "2020-04-21")
    raw = _rows("CLK0", days, 20.0)
    raw.loc[raw.index[-2], ["open", "high", "low", "close"]] = [0.0, 0.5, -40.0, -37.63]  # 2020-04-20
    assert write_contracts(raw, CL, tmp_path / "CL") == 1
    bars = pd.read_parquet(tmp_path / "CL" / "202005.parquet")
    assert bars["close"].min() == pytest.approx(-37.63)
    with pytest.raises(DataError, match="non-positive"):
        validate(bars, "CL202005")  # the default validator still refuses this for anything else


def test_outright_codes_cover_both_ticker_forms():
    codes = outright_codes(GC)
    assert "GCZ5" in codes and "GCZ25" in codes and "GCG0" in codes and "GCH5" not in codes
    assert len(codes) == 6 * (10 + 30)
    assert len(outright_codes(CL)) == 12 * 40
