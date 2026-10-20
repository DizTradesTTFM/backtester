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
