"""CFTC Commitments of Traders: download, parse, and join without look-ahead.

The one free positioning dataset. Every Friday the CFTC publishes who held what
in each US futures market as of the preceding Tuesday: commercial hedgers,
large speculators, and the small non-reportables.

The academic record is genuinely mixed and this module does not pretend
otherwise. Wang (2001, 2003) finds speculator positioning a continuation signal
and commercial positioning a contrarian one in agricultural and index futures.
Sanders et al. (2004) find no pervasive predictive power in energy and warn that
a result in one market says nothing about another. So: a testable hypothesis,
market by market, through the same screen as everything else.

**The look-ahead trap is the release lag.** The report is *as of* Tuesday but
*published* Friday afternoon. Join it to Tuesday's bar and you are trading on
data nobody had. Every row here carries `available_at` -- Friday 21:00 UTC,
after the release and after the weekly close -- and the join uses that, never
`as_of`.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LEGACY_URL = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"

#: Substrings that identify each market in the legacy report's name column.
#: Names drift across years; every alias is tried, case-insensitively.
MARKET_ALIASES: dict[str, tuple[str, ...]] = {
    "GC": ("GOLD - COMMODITY EXCHANGE",),
    "ES": ("E-MINI S&P 500 - CHICAGO MERCANTILE", "E-MINI S&P 500 STOCK INDEX"),
    "NQ": ("NASDAQ-100 STOCK INDEX (MINI)", "NASDAQ-100 CONSOLIDATED", "E-MINI NASDAQ-100",
           "NASDAQ MINI", "MICRO E-MINI NASDAQ-100"),
    "6E": ("EURO FX - CHICAGO MERCANTILE",),
    "CL": ("CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE", "WTI-PHYSICAL", "CRUDE OIL, LIGHT SWEET"),
    "ZN": ("10-YEAR U.S. TREASURY NOTES", "UST 10Y NOTE"),
}

#: Micro roots share the parent's COT market.
ROOT_TO_COT = {"MES": "ES", "MNQ": "NQ", "MGC": "GC", "M6E": "6E", "MCL": "CL", "ZN": "ZN",
               "ES": "ES", "NQ": "NQ", "GC": "GC", "6E": "6E", "CL": "CL"}

COLUMNS = {
    "Market and Exchange Names": "market",
    "As of Date in Form YYYY-MM-DD": "as_of",
    "Open Interest (All)": "oi",
    "Noncommercial Positions-Long (All)": "spec_long",
    "Noncommercial Positions-Short (All)": "spec_short",
    "Commercial Positions-Long (All)": "comm_long",
    "Commercial Positions-Short (All)": "comm_short",
    "Nonreportable Positions-Long (All)": "small_long",
    "Nonreportable Positions-Short (All)": "small_short",
}


@dataclass(frozen=True)
class COTFetchResult:
    year: int
    rows: int
    markets_matched: dict[str, str]


def parse_legacy(raw_csv: bytes | str) -> pd.DataFrame:
    """Parse the legacy futures-only annual file into a normalised frame."""
    df = pd.read_csv(io.BytesIO(raw_csv) if isinstance(raw_csv, bytes) else io.StringIO(raw_csv),
                     low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"COT file is missing expected columns: {missing}")
    out = df[list(COLUMNS)].rename(columns=COLUMNS)
    out["as_of"] = pd.to_datetime(out["as_of"], errors="coerce")
    out = out.dropna(subset=["as_of"])
    for c in ("oi", "spec_long", "spec_short", "comm_long", "comm_short", "small_long", "small_short"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["market"] = out["market"].astype(str).str.strip()
    return out.reset_index(drop=True)


def select_market(df: pd.DataFrame, code: str) -> tuple[pd.DataFrame, str]:
    """Rows for one market code, and the exact name that matched."""
    names = df["market"].unique()
    for alias in MARKET_ALIASES[code]:
        hits = [n for n in names if alias.lower() in n.lower()]
        if hits:
            # Prefer the shortest match: it is the base contract, not a spread.
            name = sorted(hits, key=len)[0]
            return df[df["market"] == name].copy(), name
    return df.iloc[0:0].copy(), ""


def download_year(year: int, timeout: float = 60.0) -> pd.DataFrame:
    import urllib.request

    url = LEGACY_URL.format(year=year)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
        return parse_legacy(z.read(name))


def features(market_df: pd.DataFrame, lookback_weeks: int = 52) -> pd.DataFrame:
    """Positioning features, all causal, plus the availability stamp."""
    f = market_df.sort_values("as_of").reset_index(drop=True).copy()
    f["spec_net"] = f["spec_long"] - f["spec_short"]
    f["comm_net"] = f["comm_long"] - f["comm_short"]
    f["small_net"] = f["small_long"] - f["small_short"]
    oi = f["oi"].replace(0, np.nan)
    f["spec_net_pct"] = f["spec_net"] / oi
    f["comm_net_pct"] = f["comm_net"] / oi

    # The classic "COT index": where this week's net sits in its trailing range.
    for col in ("spec_net", "comm_net"):
        lo = f[col].rolling(lookback_weeks, min_periods=lookback_weeks // 2).min()
        hi = f[col].rolling(lookback_weeks, min_periods=lookback_weeks // 2).max()
        f[f"{col}_index"] = (f[col] - lo) / (hi - lo).replace(0, np.nan)
    f["spec_net_chg4"] = f["spec_net"].diff(4)

    # Published Friday ~15:30 ET. Stamp Friday 21:00 UTC: after release, after
    # the FX close, before any bar that could legitimately use it.
    as_of = pd.to_datetime(f["as_of"]).dt.tz_localize("UTC")
    f["available_at"] = as_of + pd.Timedelta(days=3, hours=21)
    return f


def join_to_bars(bars: pd.DataFrame, cot: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Attach the most recent COT row that was PUBLISHED before each bar.

    merge_asof on `available_at`, never on `as_of`. A bar stamped Wednesday
    gets last week's report, not this week's, because this week's does not
    exist yet.
    """
    columns = columns or ["spec_net_pct", "comm_net_pct", "spec_net_index", "comm_net_index", "spec_net_chg4"]
    left = bars.copy()
    left["ts"] = pd.to_datetime(left["ts"], utc=True)
    right = cot[["available_at", "as_of", *columns]].sort_values("available_at")
    right = right.rename(columns={c: f"cot_{c}" for c in columns} | {"as_of": "cot_as_of"})
    merged = pd.merge_asof(left.sort_values("ts"), right, left_on="ts", right_on="available_at",
                           direction="backward")
    return merged.drop(columns=["available_at"]).reset_index(drop=True)


class COTStore:
    def __init__(self, root: str | Path = "data/cot") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, code: str) -> Path:
        return self.root / f"{code}.parquet"

    def write(self, code: str, df: pd.DataFrame) -> int:
        p = self.path(code)
        if p.exists():
            old = pd.read_parquet(p)
            df = pd.concat([old, df]).drop_duplicates(subset="as_of", keep="last")
        df = df.sort_values("as_of").reset_index(drop=True)
        df.to_parquet(p, index=False)
        return len(df)

    def read(self, code: str) -> pd.DataFrame:
        p = self.path(code)
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def fetch_range(store: COTStore, codes: list[str], years: range) -> list[COTFetchResult]:
    results = []
    for year in years:
        try:
            df = download_year(year)
        except Exception as exc:  # noqa: BLE001
            results.append(COTFetchResult(year, 0, {"error": str(exc)[:80]}))
            continue
        matched = {}
        for code in codes:
            sub, name = select_market(df, code)
            if not sub.empty:
                store.write(code, sub)
                matched[code] = name
        results.append(COTFetchResult(year, len(df), matched))
    return results
