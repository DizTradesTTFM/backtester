"""New NQ strategies built for fast prop evals (Capo, 2026-10-31, option C).

Both trade at most once a day, right after the New York open, so they get ~20 chances a month.
Flat by 3:55 PM. Sized by the firm's fixed $ risk like everything else.
"""

from types import SimpleNamespace

import numpy as np

from ..indicators import atr, hhmm
from .base import RTH_OPEN, first_per_day, to_timeframe
from .quantjourney import make


def _per_day(bars, mask, values, how):
    """Reduce `values` over `mask` per trading day (how = 'max' | 'min' | 'first' | 'last'); NaN where none."""
    out = np.full(int(bars.day[-1]) + 1, np.nan)
    d = bars.day[mask]
    v = values[mask]
    if len(d) == 0:
        return out
    if how == "max":
        np.fmax.at(out, d, v)
    elif how == "min":
        np.fmin.at(out, d, v)
    else:
        idx = np.flatnonzero(np.concatenate([[True], d[1:] != d[:-1]])) if how == "first" else \
            np.flatnonzero(np.concatenate([d[1:] != d[:-1], [True]]))
        out[d[idx]] = v[idx]
    return out


def _atr30(bars):
    """ATR(14) of 30-minute bars, known on every 1-minute bar (last finished 30-min bar)."""
    big, ends = to_timeframe(bars, 30)
    a = atr(big.high, big.low, big.close, 14)
    out = np.full(len(bars), np.nan)
    out[ends] = a
    idx = np.where(np.isfinite(out), np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    return out[idx]


def _opening_range(bars, minutes):
    in_or = (bars.minute >= RTH_OPEN) & (bars.minute < RTH_OPEN + minutes)
    hi = _per_day(bars, in_or, bars.high, "max")[bars.day]
    lo = _per_day(bars, in_or, bars.low, "min")[bars.day]
    o = _per_day(bars, in_or, bars.open, "first")[bars.day]
    c = _per_day(bars, in_or, bars.close, "last")[bars.day]
    after = bars.minute >= RTH_OPEN + minutes
    return hi, lo, o, c, after


def orb_fast(bars, p):
    """Break of the opening range: first 1-min bar that trades through the range, one trade a day."""
    hi, lo, _, _, after = _opening_range(bars, p["or_minutes"])
    rng = hi - lo
    a = _atr30(bars)
    tick = 0.25
    long = after & (bars.high > hi + tick)
    short = after & (bars.low < lo - tick)
    if p["filter"] == "overnight":  # only trade in the direction of the overnight move (18:00 open -> 9:30)
        sess_open = _per_day(bars, np.ones(len(bars), bool), bars.open, "first")[bars.day]
        rth_open = _per_day(bars, bars.minute == RTH_OPEN, bars.open, "first")[bars.day]
        up = rth_open > sess_open
        long &= up
        short &= ~up
    elif p["filter"] == "range":  # skip days where the opening range is already huge (move is spent)
        ok = rng < 1.5 * a * np.sqrt(p["or_minutes"] / 30)
        long &= ok
        short &= ok
    first = first_per_day(long | short, bars.day)
    long, short = long & first, short & first & ~long
    stop = rng * p["stop_frac"]
    return make(bars, bars, np.arange(len(bars)), 1, long, stop, short=short, target=stop * p["rr"],
                window=(930, p["end"]))


def first_pullback(bars, p):
    """Opening drive, then buy/sell the first pullback into it with a limit order."""
    hi, lo, o, c, after = _opening_range(bars, p["or_minutes"])
    a = _atr30(bars)
    move = c - o
    strong = np.abs(move) >= p["min_move"] * a
    at_or_end = bars.minute == RTH_OPEN + p["or_minutes"]
    up = strong & (move > 0) & at_or_end
    dn = strong & (move < 0) & at_or_end
    lim = np.where(move > 0, c - p["retrace"] * (hi - lo), c + p["retrace"] * (hi - lo))
    if p["stop"] == "extreme":  # beyond the far side of the opening range
        stop = np.where(move > 0, lim - lo, hi - lim) + 0.5
    else:
        stop = 1.0 * a
    stop = np.where(np.isfinite(stop) & (stop > 0), stop, np.nan)
    expiry = hhmm(1100) - (RTH_OPEN + p["or_minutes"])
    return make(bars, bars, np.arange(len(bars)), 1, up, stop, short=dn, target=stop * p["rr"],
                limit_px=lim, expiry=expiry, window=(930, 1200))


def _drive(bars, start, end):
    """High, low, first open and last close of each day's [start, end) window, on every bar."""
    w = (bars.minute >= hhmm(start)) & (bars.minute < hhmm(end))
    return (_per_day(bars, w, bars.high, "max")[bars.day], _per_day(bars, w, bars.low, "min")[bars.day],
            _per_day(bars, w, bars.open, "first")[bars.day], _per_day(bars, w, bars.close, "last")[bars.day])


def pullback_plus(bars, p):
    """First Pullback with more chances a day: morning pullback, optional afternoon pullback,
    optional breakout entry at 11:00 if the morning pullback never filled."""
    n = len(bars)
    a = _atr30(bars)
    entry = np.zeros(n, np.int8)
    lim = np.full(n, np.nan)
    stp = np.full(n, np.nan)
    exp = np.ones(n, np.int64)

    def setup(start, end, until, kind):
        hi, lo, o, c = _drive(bars, start, end)
        move = c - o
        strong = np.abs(move) >= p["min_move"] * a
        at = (bars.minute == hhmm(end)) & strong & np.isfinite(move)
        d = np.where(move > 0, 1, -1).astype(np.int8)
        if kind == "pullback":
            px = np.where(move > 0, c - p["retrace"] * (hi - lo), c + p["retrace"] * (hi - lo))
            lim[at] = px[at]
        else:  # breakout through the drive's extreme
            px = np.where(move > 0, hi + 0.25, lo - 0.25)
            stp[at] = px[at]
        entry[at] = d[at]
        exp[at] = hhmm(until) - hhmm(end)

    setup(930, 1000, 1100, "pullback")
    if p["breakout"]:
        setup(930, 1101, 1300, "breakout")
    if p["afternoon"]:
        setup(1200, 1330, 1500, "pullback")
    stop = p["stop_atr"] * a
    from .base import default_force_exit
    from ..engine import Signals
    ok = np.isfinite(stop) & (stop > 0)
    entry[~ok] = 0
    tgt = np.where(ok, stop * p["rr"], 0.0)
    return Signals(entry, np.where(ok, stop, 0.0), tgt, default_force_exit(bars),
                   exit_sig=np.zeros(n, np.int8), max_hold=np.zeros(n, np.int64), trail_pts=np.zeros(n),
                   limit_px=lim, stop_px=stp, stop_cap=np.full(n, np.nan), expiry=exp)


def _s(name, fn, desc, grid):
    return SimpleNamespace(NAME=name, DESCRIPTION=desc, PARAM_GRID=grid, generate=fn, MINE=True, SOURCE="Claude (prop)")


PROP = [
    _s("nq_pullback_plus", pullback_plus, "NQ First Pullback with more chances a day: the 9:30-10:00 drive pullback "
       "(as in First Pullback), plus optionally a breakout entry at 11:00 through the morning extreme if no pullback came, "
       "and an afternoon pullback after a strong 12:00-1:30 PM move. Stop 1 ATR, target 2-3R.",
       {"min_move": [0.75, 1.0], "retrace": [0.382, 0.5], "rr": [2, 3], "stop_atr": [1.0], "breakout": [False, True],
        "afternoon": [False, True]}),
    _s("nq_orb_fast", orb_fast, "Prop-fast opening range breakout (NQ): mark the high/low of the first 5/15/30 minutes after "
       "9:30, take the first break either way (one trade a day), stop at half or all of the range, target 1.5-3R. "
       "Optional filters: only with the overnight direction, or skip days with an oversized range.",
       {"or_minutes": [5, 15, 30], "stop_frac": [0.5, 1.0], "rr": [1.5, 2, 3], "filter": ["none", "overnight", "range"],
        "end": [1100, 1200]}),
    _s("nq_first_pullback", first_pullback, "Prop-fast first pullback (NQ): if the first 15/30 minutes move strongly one "
       "way, place a limit order to join that move on a 38-62% pullback of the opening range (until 11:00), stop beyond "
       "the range or 1 ATR, target 1.5-3R. One trade a day.",
       {"or_minutes": [15, 30], "min_move": [0.5, 1.0, 1.5], "retrace": [0.382, 0.5, 0.618], "stop": ["extreme", "atr"],
        "rr": [1.5, 2, 3]}),
]
