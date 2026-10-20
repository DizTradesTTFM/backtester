"""Bar-by-bar trade simulator, compiled with numba for speed.

How fills work (deliberately pessimistic):
- A signal on bar i's close enters at bar i+1's open, plus slippage against you.
- Stops fill at the stop price plus slippage, or at the open if price gapped through.
- Targets fill only if price trades one tick THROUGH the target (touching is not enough). No slippage on targets.
- If a bar hits both stop and target, we assume the stop hit first.
- Forced exits (end of session) fill at the bar's close minus slippage.
- Commission is charged on every round trip.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from .instruments import Instrument


@dataclass
class Signals:
    """What a strategy hands to the engine. All arrays are the same length as the bars.

    entry: +1 = go long at next bar open, -1 = go short, 0 = nothing. Ignored while in a trade.
    stop_pts / target_pts: distance in points from the fill price. 0 or less = no stop / no target.
    force_exit: True on bars where any open trade must be closed at the bar's close.
    """

    entry: np.ndarray
    stop_pts: np.ndarray
    target_pts: np.ndarray
    force_exit: np.ndarray


@dataclass
class Trades:
    entry_i: np.ndarray
    exit_i: np.ndarray
    direction: np.ndarray
    entry_px: np.ndarray
    exit_px: np.ndarray
    pnl: np.ndarray  # dollars, after commission and slippage
    mae: np.ndarray  # worst open loss during the trade, dollars (<= 0), incl. commission
    day: np.ndarray  # trading-day index the trade closed on

    def __len__(self) -> int:
        return len(self.pnl)


@njit(cache=True)
def _simulate(o, h, l, c, entry, stop_pts, target_pts, force_exit, tick, slip, pv, comm, qty):
    n = len(c)
    cap = n // 2 + 1
    e_i = np.empty(cap, np.int64)
    x_i = np.empty(cap, np.int64)
    dirs = np.empty(cap, np.int8)
    e_px = np.empty(cap, np.float64)
    x_px = np.empty(cap, np.float64)
    mae = np.empty(cap, np.float64)
    k = 0

    pos = 0
    pending = 0
    pend_stop = 0.0
    pend_tgt = 0.0
    fill = 0.0
    stop = 0.0
    tgt = 0.0
    worst = 0.0
    start = 0

    for i in range(n):
        if pending != 0 and pos == 0:
            pos = pending
            start = i
            fill = o[i] + pos * slip
            stop = fill - pos * pend_stop if pend_stop > 0 else np.nan
            tgt = fill + pos * pend_tgt if pend_tgt > 0 else np.nan
            worst = fill
        pending = 0

        if pos != 0:
            exit_px = np.nan
            if pos == 1:
                worst = min(worst, l[i])
                if not np.isnan(stop) and l[i] <= stop:
                    exit_px = min(o[i], stop) - slip
                    worst = min(worst, exit_px)
                elif not np.isnan(tgt) and h[i] >= tgt + tick:
                    exit_px = max(o[i], tgt) if i > start else tgt
            else:
                worst = max(worst, h[i])
                if not np.isnan(stop) and h[i] >= stop:
                    exit_px = max(o[i], stop) + slip
                    worst = max(worst, exit_px)
                elif not np.isnan(tgt) and l[i] <= tgt - tick:
                    exit_px = min(o[i], tgt) if i > start else tgt
            if np.isnan(exit_px) and force_exit[i]:
                exit_px = c[i] - pos * slip
            if not np.isnan(exit_px):
                e_i[k] = start
                x_i[k] = i
                dirs[k] = pos
                e_px[k] = fill
                x_px[k] = exit_px
                mae[k] = min(0.0, (worst - fill) * pos * pv * qty) - comm * qty
                k += 1
                pos = 0
                continue  # no re-entry on the same bar we exited

        if pos == 0 and entry[i] != 0 and not force_exit[i] and i + 1 < n:
            pending = entry[i]
            pend_stop = stop_pts[i]
            pend_tgt = target_pts[i]

    pnl = (x_px[:k] - e_px[:k]) * dirs[:k] * pv * qty - comm * qty
    return e_i[:k], x_i[:k], dirs[:k], e_px[:k], x_px[:k], pnl, mae[:k]


def run_backtest(bars, signals: Signals, inst: Instrument, slippage_ticks: float = 1.0, qty: int = 1) -> Trades:
    e_i, x_i, d, e_px, x_px, pnl, mae = _simulate(
        bars.open,
        bars.high,
        bars.low,
        bars.close,
        signals.entry.astype(np.int8),
        signals.stop_pts.astype(np.float64),
        signals.target_pts.astype(np.float64),
        signals.force_exit.astype(np.bool_),
        inst.tick_size,
        slippage_ticks * inst.tick_size,
        inst.point_value,
        inst.commission_rt,
        qty,
    )
    return Trades(e_i, x_i, d, e_px, x_px, pnl, mae, bars.day[x_i])
