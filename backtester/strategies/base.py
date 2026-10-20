"""Shared helpers for strategy templates.

A strategy module defines:
    NAME        short id, e.g. "orb"
    DESCRIPTION one line, plain English
    PARAM_GRID  dict of setting -> list of values to try. Every combination is one backtest.
    generate(bars, params) -> Signals
"""

import numpy as np

from ..engine import Signals
from ..indicators import hhmm, last_bar_of_day

RTH_OPEN = hhmm(930)
FLAT_BY = hhmm(1555)  # everything closed before the cash close; prop firms want you flat


def entry_window(bars, start_hhmm: int, end_hhmm: int) -> np.ndarray:
    return (bars.minute >= hhmm(start_hhmm)) & (bars.minute < hhmm(end_hhmm))


def default_force_exit(bars, flat_hhmm: int = 1555) -> np.ndarray:
    return (bars.minute >= hhmm(flat_hhmm)) | last_bar_of_day(bars.day)


def first_per_day(mask: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Keep only the first True of each trading day."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return mask.copy()
    keep = np.ones(len(idx), dtype=bool)
    keep[1:] = day[idx[1:]] != day[idx[:-1]]
    out = np.zeros_like(mask)
    out[idx[keep]] = True
    return out


def build(bars, long_mask, short_mask, stop_pts, target_pts, force_exit, side: str = "both") -> Signals:
    entry = np.zeros(len(bars), dtype=np.int8)
    if side in ("both", "long"):
        entry[long_mask] = 1
    if side in ("both", "short"):
        entry[short_mask & ~long_mask] = -1
    n = len(bars)
    stop = np.broadcast_to(np.asarray(stop_pts, dtype=np.float64), (n,)).copy()
    tgt = np.broadcast_to(np.asarray(target_pts, dtype=np.float64), (n,)).copy()
    # never enter with a missing or zero stop: every trade must have defined risk
    bad = ~np.isfinite(stop) | (stop <= 0)
    entry[bad] = 0
    tgt[~np.isfinite(tgt)] = 0
    return Signals(entry, stop, tgt, force_exit)


SESSION_START = 18 * 60  # CME equity futures sessions open at 18:00 New York


def to_timeframe(bars, tf: int):
    """Combine 1-minute bars into tf-minute bars (never across trading days).

    Bars are counted from the 18:00 session open, so 4-hour candles are 18-22, 22-02, 02-06, 06-10, 10-14, 14-18.
    Returns (bigger bars, ends) where ends[k] is the 1-minute bar index on which bigger bar k closes.
    A signal on bigger bar k is acted on at the next 1-minute open after ends[k].
    """
    if tf <= 1:
        return bars, np.arange(len(bars))
    key = bars.day.astype(np.int64) * 1440 + ((bars.minute - SESSION_START) % 1440) // tf
    starts = np.flatnonzero(np.concatenate([[True], key[1:] != key[:-1]]))
    ends = np.concatenate([starts[1:] - 1, [len(bars) - 1]])
    return aggregate(bars, starts, ends), ends


def aggregate(bars, starts, ends):
    """Bigger bars from 1-minute bars, given where each bigger bar starts and ends."""
    return type(bars)(
        bars.symbol,
        bars.ts[ends],
        bars.open[starts],
        np.maximum.reduceat(bars.high, starts),
        np.minimum.reduceat(bars.low, starts),
        bars.close[ends],
        np.add.reduceat(bars.volume, starts),
        bars.minute[ends] + 1,  # time the bigger bar closes (a 9:30-9:34 bar closes at 9:35)
        bars.day[starts],
        bars.day_dates,
    )


def expand(bars, ends, sig: Signals, force_exit=None) -> Signals:
    """Map signals made on bigger bars back onto the 1-minute bars the engine trades on."""
    n = len(bars)
    entry = np.zeros(n, np.int8)
    stop = np.zeros(n)
    tgt = np.zeros(n)
    entry[ends] = sig.entry
    stop[ends] = sig.stop_pts
    tgt[ends] = sig.target_pts
    return Signals(entry, stop, tgt, default_force_exit(bars) if force_exit is None else force_exit)
