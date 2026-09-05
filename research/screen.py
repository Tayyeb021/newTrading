"""Signal screen: many cheap hypotheses, one honest bar.

The expensive way to research is one idea, one backtest, one week. The cheap
way is the way a quant desk does it: state twenty hypotheses, each with a
*structural reason* the money might exist, and measure every one at the signal
level in seconds -- forward return in the signalled direction, in ATR units,
no stops, no targets, no costs. Only what survives gets a backtest.

Three disciplines make the numbers mean something:

1. **Drift is removed.** Forward returns are demeaned within the sample before
   they are signed. Gold rose 86% through one window and made "long on Mondays"
   look brilliant; demeaning is what stops the bull market from posing as a
   signal. What is left is covariance between the signal and the return.

2. **The bar rises with the count.** Screening 45 things at p < 0.05 hands you
   two false positives on average. Bonferroni: a hypothesis has to clear
   p < 0.05 / N -- with 45 tests that is |t| > 3.3, not 2.

3. **It has to repeat.** The location effect from the first strategy had
   t = 2.61 in one two-year window and was positive in 3 of 9 years everywhere
   else. So every hypothesis is scored per year, and needs to be positive in
   at least 70% of them. A real effect shows up most years. Noise shows up in
   the one you happened to look at.

Adding a hypothesis is five lines. Every signal function MUST be causal -- it
may read row i and earlier, never later. Windows are non-overlapping.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.store import BarStore  # noqa: E402
from features.indicators import atr, rolling_return  # noqa: E402

SignalFn = Callable[[pd.DataFrame], np.ndarray]


@dataclass
class Hypothesis:
    name: str
    why: str  # required. "the indicator crossed" is not a why.
    symbols: list[str]
    timeframe: str
    horizon: int  # bars held
    signal: SignalFn  # +1 long, -1 short, 0 no view -- CAUSAL
    aux: str | None = None  # a second symbol the signal reads


@dataclass
class Score:
    hypothesis: str
    symbol: str
    n: int
    excess_atr: float  # drift-removed, signed
    t: float
    p: float
    hit: float
    years_pos: int
    years: int
    trials: int

    @property
    def bonferroni(self) -> bool:
        return self.p < 0.05 / self.trials

    @property
    def consistent(self) -> bool:
        return self.years >= 6 and self.years_pos / self.years >= 0.70

    @property
    def verdict(self) -> str:
        if self.n < 60:
            return "too few"
        if self.bonferroni and self.consistent:
            return "SURVIVES"
        if self.bonferroni:
            return "significant, inconsistent"
        if self.consistent and abs(self.t) > 2:
            return "consistent, weak"
        return "-"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _nonoverlap(idx: np.ndarray, horizon: int) -> np.ndarray:
    keep, last = [], -10**9
    for i in idx:
        if i - last >= horizon:
            keep.append(i)
            last = i
    return np.array(keep, dtype=int)


def score(h: Hypothesis, df: pd.DataFrame, symbol: str, trials: int,
          aux_df: pd.DataFrame | None = None) -> Score:
    df = df.reset_index(drop=True)
    a = atr(df, 14).to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    fwd = np.full(n, np.nan)
    fwd[: n - h.horizon] = (close[h.horizon:] - close[: n - h.horizon]) / a[: n - h.horizon]

    sig = h.signal(df) if aux_df is None else h.signal(df, aux_df)  # type: ignore[call-arg]
    sig = np.asarray(sig, dtype=float)
    sig[~np.isfinite(sig)] = 0.0

    valid = np.isfinite(fwd) & (sig != 0)
    valid[:300] = False  # warmup for any indicator inside the signal
    idx = _nonoverlap(np.flatnonzero(valid), h.horizon)
    if idx.size < 10:
        return Score(h.name, symbol, idx.size, 0.0, 0.0, 1.0, 0.0, 0, 0, trials)

    # Demean against the WHOLE sample's forward return, not just the signalled
    # bars, so a signal that fires in an up-trending window is not credited with
    # the trend.
    all_idx = _nonoverlap(np.flatnonzero(np.isfinite(fwd) & (np.arange(n) >= 300)), h.horizon)
    drift = float(np.nanmean(fwd[all_idx]))
    signed = (fwd[idx] - drift) * sig[idx]

    t, p = stats.ttest_1samp(signed, 0.0)
    years = pd.DatetimeIndex(df["ts"].iloc[idx]).year
    by_year = pd.Series(signed).groupby(years.values).mean()
    return Score(
        hypothesis=h.name, symbol=symbol, n=idx.size,
        excess_atr=float(signed.mean()), t=float(t), p=float(p),
        hit=float((signed > 0).mean()),
        years_pos=int((by_year > 0).sum()), years=int(len(by_year)), trials=trials,
    )


# --------------------------------------------------------------------------- #
# Signal library -- every one causal, every one with a reason
# --------------------------------------------------------------------------- #

def _ret(df, k):
    return rolling_return(df["close"], k).to_numpy(dtype=float)


def sig_weekday(day: int) -> SignalFn:
    return lambda df: (pd.to_datetime(df["ts"]).dt.dayofweek.to_numpy() == day).astype(float)


def sig_turn_of_month(df):
    d = pd.to_datetime(df["ts"])
    month = d.dt.month.to_numpy()
    nxt = np.roll(month, -1); prv = np.roll(month, 1)
    last2 = (nxt != month) | (np.roll(nxt, -1) != month)
    first2 = (prv != month) | (np.roll(prv, 1) != month)
    out = (last2 | first2).astype(float)
    out[-2:] = 0
    return out


def sig_reversal(k: int) -> SignalFn:
    return lambda df: -np.sign(np.nan_to_num(_ret(df, k)))


def sig_momentum(k: int) -> SignalFn:
    return lambda df: np.sign(np.nan_to_num(_ret(df, k)))


def sig_streak_fade(k: int) -> SignalFn:
    def f(df):
        r = np.sign(df["close"].diff().to_numpy(dtype=float))
        out = np.zeros(len(df))
        for i in range(k, len(df)):
            w = r[i - k + 1 : i + 1]
            if np.all(w > 0):
                out[i] = -1.0
            elif np.all(w < 0):
                out[i] = 1.0
        return out
    return f


def sig_range_expansion_continue(df):
    a = atr(df, 14).to_numpy(dtype=float)
    rng = (df["high"] - df["low"]).to_numpy(dtype=float)
    r = np.sign(df["close"].diff().to_numpy(dtype=float))
    return np.where(rng > 2.0 * a, r, 0.0)


def sig_narrow_range_breakout(df):
    rng = (df["high"] - df["low"])
    narrow = rng < rng.rolling(20).mean() * 0.6
    hi = df["high"].rolling(5).max().shift(1); lo = df["low"].rolling(5).min().shift(1)
    out = np.where(df["close"] > hi, 1.0, np.where(df["close"] < lo, -1.0, 0.0))
    return np.where(narrow.shift(1).fillna(False).to_numpy(), out, 0.0)


def sig_hour(hour: int, horizon_note: str = "") -> SignalFn:
    return lambda df: (pd.to_datetime(df["ts"]).dt.hour.to_numpy() == hour).astype(float)


def _us_open_hour(ts: pd.Series) -> np.ndarray:
    m = ts.dt.month.to_numpy()
    return np.where((m >= 4) & (m <= 10), 13, 14)  # 13:30 UTC in US DST, 14:30 otherwise


def sig_us_open_continuation(df):
    """Direction of the first 30 minutes of US cash, on M15. Held to the close."""
    ts = pd.to_datetime(df["ts"]); h = ts.dt.hour.to_numpy(); mnt = ts.dt.minute.to_numpy()
    oh = _us_open_hour(ts)
    at = (h == oh) & (mnt == 45)  # the bar closing 30 min after a :30 open... 13:45 bar = 13:45-14:00
    two_bar = (df["close"] - df["open"].shift(1)).to_numpy(dtype=float)
    return np.where(at, np.sign(np.nan_to_num(two_bar)), 0.0)


def sig_us_open_fade(df):
    return -sig_us_open_continuation(df)


def sig_overnight_long_indices(df):
    """Long at the US cash close, out at the next open. The NY Fed 'overnight drift'."""
    ts = pd.to_datetime(df["ts"]); h = ts.dt.hour.to_numpy()
    close_h = np.where((ts.dt.month.to_numpy() >= 4) & (ts.dt.month.to_numpy() <= 10), 20, 21)
    return (h == close_h).astype(float)


def sig_london_open_continuation(df):
    """Direction of 07:00-08:00 UTC, held into the session. On H1."""
    h = pd.to_datetime(df["ts"]).dt.hour.to_numpy()
    r = np.sign((df["close"] - df["open"]).to_numpy(dtype=float))
    return np.where(h == 7, r, 0.0)


def sig_asian_range_break(df):
    """At 08:00 UTC, above the 00-07 range -> long, below -> short. On H1."""
    ts = pd.to_datetime(df["ts"]); h = ts.dt.hour.to_numpy()
    date = ts.dt.date
    out = np.zeros(len(df))
    hi = df["high"].where(h < 7).groupby(date).transform("max").to_numpy()
    lo = df["low"].where(h < 7).groupby(date).transform("min").to_numpy()
    c = df["close"].to_numpy()
    at = h == 7
    out[at & (c > hi)] = 1.0
    out[at & (c < lo)] = -1.0
    return out


def sig_gold_follows_dollar(gold: pd.DataFrame, eur: pd.DataFrame) -> np.ndarray:
    """EURUSD up yesterday = dollar down = gold up today? Cross-asset, lagged."""
    e = eur[["ts", "close"]].copy(); e["ts"] = pd.to_datetime(e["ts"]).dt.normalize()
    e["eur_ret"] = e["close"].pct_change().shift(1)  # YESTERDAY's move, known today
    g = gold[["ts"]].copy(); g["ts"] = pd.to_datetime(g["ts"]).dt.normalize()
    m = g.merge(e[["ts", "eur_ret"]], on="ts", how="left")
    return np.sign(np.nan_to_num(m["eur_ret"].to_numpy(dtype=float)))


def sig_gold_risk_off(gold: pd.DataFrame, spx: pd.DataFrame) -> np.ndarray:
    """Stocks down yesterday -> gold bid today? Lagged, cross-asset."""
    s = spx[["ts", "close"]].copy(); s["ts"] = pd.to_datetime(s["ts"]).dt.normalize()
    s["spx_ret"] = s["close"].pct_change().shift(1)
    g = gold[["ts"]].copy(); g["ts"] = pd.to_datetime(g["ts"]).dt.normalize()
    m = g.merge(s[["ts", "spx_ret"]], on="ts", how="left")
    return -np.sign(np.nan_to_num(m["spx_ret"].to_numpy(dtype=float)))


FX = ["EURUSD"]
ALL = ["EURUSD", "XAUUSD", "US30", "US500"]
IDX = ["US30", "US500"]

HYPOTHESES: list[Hypothesis] = [
    # --- calendar: mechanical structure --------------------------------------
    Hypothesis("monday_long", "weekend information arrives Monday; documented equity effect", ALL, "D1", 1, sig_weekday(0)),
    Hypothesis("friday_long", "position squaring before the weekend", ALL, "D1", 1, sig_weekday(4)),
    Hypothesis("turn_of_month", "pension and fund flows at month boundaries", ALL, "D1", 1, sig_turn_of_month),
    # --- behavioural: reversal and momentum at several horizons --------------
    Hypothesis("reversal_1d", "short-horizon overreaction; well documented in equities", ALL, "D1", 1, sig_reversal(1)),
    Hypothesis("reversal_5d", "one-week overreaction", ALL, "D1", 5, sig_reversal(5)),
    Hypothesis("momentum_20d", "one-month continuation", ALL, "D1", 5, sig_momentum(20)),
    Hypothesis("momentum_120d", "six-month continuation, mid-TSMOM band", ALL, "D1", 10, sig_momentum(120)),
    Hypothesis("momentum_250d", "twelve-month, the classic TSMOM lookback", ALL, "D1", 20, sig_momentum(250)),
    Hypothesis("streak3_fade", "three same-direction days exhausts short-term flow", ALL, "D1", 1, sig_streak_fade(3)),
    Hypothesis("range_expansion_continue", "a 2-ATR day is news; news trends", ALL, "D1", 3, sig_range_expansion_continue),
    Hypothesis("narrow_range_breakout", "compression precedes expansion", ALL, "D1", 3, sig_narrow_range_breakout),
    # --- intraday structure --------------------------------------------------
    Hypothesis("us_open_continuation", "opening imbalance persists through the session", IDX, "M15", 24, sig_us_open_continuation),
    Hypothesis("us_open_fade", "opening imbalance is absorbed; the day reverts", IDX, "M15", 24, sig_us_open_fade),
    Hypothesis("overnight_long_indices", "dealer inventory premium overnight (NY Fed says decayed since 2021)", IDX, "H1", 16, sig_overnight_long_indices),
    Hypothesis("london_open_continuation", "liquidity arrival sets the session's direction", ["EURUSD", "XAUUSD"], "H1", 4, sig_london_open_continuation),
    Hypothesis("asian_range_break", "the retail 'London breakout' -- folklore, cheap to test", ["EURUSD", "XAUUSD"], "H1", 8, sig_asian_range_break),
    # --- cross-asset ---------------------------------------------------------
    Hypothesis("gold_follows_dollar", "gold is priced in dollars; dollar moves lead by a day?", ["XAUUSD"], "D1", 1, sig_gold_follows_dollar, aux="EURUSD"),
    Hypothesis("gold_risk_off", "gold as the haven bid after equity selloffs", ["XAUUSD"], "D1", 1, sig_gold_risk_off, aux="US500"),
]

# Hour-of-day is 24 hypotheses in one. Counted as 24 trials.
HOUR_SYMBOLS = ["EURUSD", "XAUUSD"]
#: 20:00-02:00 UTC is the FX rollover and the dead zone after it. Spreads run
#: 8-15x normal, gold is closed for an hour, and bid-based bars produce a V that
#: sums to nothing. The screen found it once; it is not allowed to find it twice.
EXCLUDED_HOURS = {20, 21, 22, 23, 0, 1}


def run(store: BarStore, min_year: int = 2010) -> tuple[list[Score], int]:
    cache: dict[tuple[str, str], pd.DataFrame] = {}

    def load(sym, tf):
        if (sym, tf) not in cache:
            df = store.read(sym, tf)
            cache[(sym, tf)] = df[pd.to_datetime(df["ts"]).dt.year >= min_year].reset_index(drop=True)
        return cache[(sym, tf)]

    trials = sum(len(h.symbols) for h in HYPOTHESES) + (24 - len(EXCLUDED_HOURS)) * len(HOUR_SYMBOLS)
    scores: list[Score] = []

    for h in HYPOTHESES:
        for sym in h.symbols:
            df = load(sym, h.timeframe)
            if df.empty:
                continue
            aux = load(h.aux, h.timeframe) if h.aux else None
            scores.append(score(h, df, sym, trials, aux))

    for sym in HOUR_SYMBOLS:
        df = load(sym, "H1")
        for hour in range(24):
            if hour in EXCLUDED_HOURS:
                continue
            h = Hypothesis(f"hour_{hour:02d}_long", "time-of-day flow", [sym], "H1", 1, sig_hour(hour))
            scores.append(score(h, df, sym, trials))

    return scores, trials


def report(scores: list[Score], trials: int) -> str:
    L = []
    L.append(f"SIGNAL SCREEN  -  {len(scores)} tests, {trials} trials counted for Bonferroni")
    L.append(f"bar: |t| > {stats.norm.ppf(1 - 0.025 / trials):.2f} (p < {0.05 / trials:.5f})  AND  positive in >= 70% of years\n")
    L.append(f"  {'hypothesis':<26}{'symbol':<8}{'n':>6}{'excess':>9}{'t':>7}{'hit':>6}{'years+':>9}  verdict")
    L.append("  " + "-" * 84)
    ranked = sorted(scores, key=lambda s: -abs(s.t))
    for s in ranked:
        if s.n < 10:
            continue
        L.append(f"  {s.hypothesis:<26}{s.symbol:<8}{s.n:>6,}{s.excess_atr:>+9.3f}{s.t:>7.2f}"
                 f"{s.hit:>6.0%}{s.years_pos:>5}/{s.years:<3}  {s.verdict}")
    survivors = [s for s in scores if s.verdict == "SURVIVES"]
    L.append("")
    L.append(f"  survivors: {len(survivors)}")
    for s in survivors:
        L.append(f"    {s.hypothesis} on {s.symbol}: {s.excess_atr:+.3f} ATR, t={s.t:.2f}, {s.years_pos}/{s.years} years")
    return "\n".join(L)


if __name__ == "__main__":
    scores, trials = run(BarStore("data/bars"))
    print(report(scores, trials))
