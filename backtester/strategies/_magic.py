"""Shared pieces of the Magic Tradez (MM) models: 1H candles, C2 closures, SMT and PSP against the partner index.

Bearish setups reuse the bullish code on flipped prices (-open, -low as high, -high as low, -close),
so every rule below is written for longs only.
"""

import numpy as np
from numba import njit

from ..data import partner_bars
from ..indicators import hhmm, last_bar_of_day
from .base import SESSION_START, aggregate, build, to_timeframe

# 1H candle close times allowed for entry (entry is at the next candle's open, i.e. the C3 open)
HOURS = {
    "ny": (hhmm(1000), hhmm(1500)),
    "london_ny": (hhmm(300), hhmm(1500)),
    "all": None,  # any hour from the 18:00 open until 15:00
}


def force_exit(bars):
    """Flat by 15:55 New York. Unlike RTH strategies, trades opened overnight are allowed."""
    return ((bars.minute >= hhmm(1555)) & (bars.minute < SESSION_START)) | last_bar_of_day(bars.day)


def hour_ok(close_min, hours):
    if HOURS[hours] is None:
        return (close_min <= hhmm(1500)) | (close_min > SESSION_START)
    lo, hi = HOURS[hours]
    return (close_min >= lo) & (close_min <= hi)


def hourly(bars, need_partner):
    """1H candles for this market, plus the partner's 1H candles over the exact same minutes (or None)."""
    h1, ends = to_timeframe(bars, 60)
    if not need_partner:
        return h1, None, ends
    pb = partner_bars(bars)
    if pb is None:
        return h1, None, ends
    starts = np.concatenate([[0], ends[:-1] + 1])
    return h1, aggregate(pb, starts, ends), ends


def flip(c):
    """(open, high, low, close) of a bar set, as bullish code sees it: long side or flipped short side."""
    return [(c.open, c.high, c.low, c.close), (-c.open, -c.low, -c.high, -c.close)]


def stop_distance(mode, o, c, protected_low):
    """How far below the C2 close the stop goes (TTrades stop rules, longs; shorts use flipped prices).

    swing: just beyond the protected swing, the low that was swept and closed back above.
    body:  the refined stop at the C2 candle's body, only when the body sits in the lower half of the run from the
           protected low to the close. Otherwise the body stop would cut into the real invalidation, so it falls
           back to the swing stop.
    """
    swing = c - protected_low
    if mode == "swing":
        return swing
    body_low = np.minimum(o, c)
    ok = (body_low <= protected_low + 0.5 * swing) & (body_low < c)
    return np.where(ok, c - body_low, swing)


def r_target(target, stop, range_pts):
    if target == "range":
        return range_pts
    return stop * float(target.rstrip("R"))


def finish(bars, ends, h1, longs, shorts, side):
    """longs/shorts: (mask, stop_pts, target_pts) on 1H candles. Mapped onto the 1-minute bars at each candle's close."""
    n = len(bars)
    entry_l = np.zeros(n, bool)
    entry_s = np.zeros(n, bool)
    stop = np.full(n, np.nan)
    tgt = np.zeros(n)
    for mask, s, t, out in ((*longs, entry_l), (*shorts, entry_s)):
        ok = mask & np.isfinite(s) & (s > 0) & np.isfinite(t) & (t > 0)
        out[ends[ok]] = True
        stop[ends[ok]] = s[ok]
        tgt[ends[ok]] = t[ok]
    both = entry_l & entry_s
    entry_l[both] = entry_s[both] = False  # conflicting signals on the same candle: skip
    return build(bars, entry_l, entry_s, stop, tgt, force_exit(bars), side)


@njit(cache=True)
def group_cummin(x, g):
    """Running minimum of x inside each group (groups are consecutive runs of equal g)."""
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = x[i] if i == 0 or g[i] != g[i - 1] else min(out[i - 1], x[i])
    return out


@njit(cache=True)
def open_fvgs(h, l, c, lookback):
    """For each candle j: the newest bullish fair value gap formed in the last `lookback` candles and still open.

    A bullish FVG at candle k: high[k-2] < low[k]; the gap is high[k-2]..low[k]. It is dead once a later candle
    before j closes below its bottom. Returns (bottom, top) arrays, NaN where there is none.
    """
    n = len(h)
    bot = np.full(n, np.nan)
    top = np.full(n, np.nan)
    for j in range(n):
        for k in range(j - 1, max(j - lookback, 2) - 1, -1):
            if h[k - 2] < l[k]:
                dead = False
                for m in range(k + 1, j):
                    if c[m] < h[k - 2]:
                        dead = True
                        break
                if not dead:
                    bot[j] = h[k - 2]
                    top[j] = l[k]
                    break
    return bot, top
