"""Bar-by-bar trade simulator, compiled with numba for speed.

How fills work (deliberately pessimistic):
- A signal on bar i's close enters at bar i+1's open, plus slippage against you.
- Stops fill at the stop price plus slippage, or at the open if price gapped through.
- Targets fill only if price trades one tick THROUGH the target (touching is not enough). No slippage on targets.
- If a bar hits both stop and target, we assume the stop hit first.
- Forced exits (end of session) fill at the bar's close minus slippage.
- Commission is charged on every round trip.

Optional order features (all off by default):
- Limit / stop entry orders: instead of entering at the next open, wait for price to reach a level.
  Limit fills at the level (or a better open), no slippage. Stop fills at the level (or a worse open) plus slippage.
  If both are set it is an OCO pair: whichever is hit first fills. Unfilled orders expire after `expiry` bars.
- Trailing stop: the stop follows the best price since entry at `trail_pts` distance (never moves back).
- Time exit: close at the bar's close after `max_hold` bars in the trade.
- Signal exit: exit_sig = +1 closes longs / -1 closes shorts at the bar's close (e.g. "sell on the reverse cross").
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
    exit_sig: np.ndarray | None = None  # int8: +1 close longs, -1 close shorts, at this bar's close
    max_hold: np.ndarray | None = None  # int: bars to hold the trade entered from this signal (0 = no limit)
    trail_pts: np.ndarray | None = None  # trailing stop distance for the trade entered from this signal (0 = none)
    limit_px: np.ndarray | None = None  # entry limit price (NaN = none)
    stop_px: np.ndarray | None = None  # entry stop price (NaN = none)
    stop_cap: np.ndarray | None = None  # stop-limit entry: no fill above (long) / below (short) this price
    expiry: np.ndarray | None = None  # int: bars a limit/stop entry order stays working


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
def _simulate(o, h, l, c, entry, stop_pts, target_pts, force_exit, exit_sig, max_hold, trail_pts, limit_px, stop_px,
              stop_cap, expiry, tick, slip, pv, comm, qty):
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
    pending = 0  # direction of a working entry order
    p_stop = 0.0
    p_tgt = 0.0
    p_hold = 0
    p_trail = 0.0
    p_lim = np.nan
    p_stp = np.nan
    p_cap = np.nan
    p_until = 0
    fill = 0.0
    stop = 0.0
    tgt = 0.0
    trail = 0.0
    hold = 0
    best = 0.0
    worst = 0.0
    start = 0

    for i in range(n):
        if pending != 0 and pos == 0:
            d = pending
            px = np.nan
            if np.isnan(p_lim) and np.isnan(p_stp):
                px = o[i] + d * slip  # market order: next open
            else:
                if not np.isnan(p_stp) and ((d == 1 and h[i] >= p_stp) or (d == -1 and l[i] <= p_stp)):
                    px = (max(o[i], p_stp) if d == 1 else min(o[i], p_stp)) + d * slip
                    if not np.isnan(p_cap) and (px - d * slip - p_cap) * d > 0:
                        px = np.nan  # stop-limit: triggered beyond the limit, no fill
                        pending = 0
                elif not np.isnan(p_lim) and ((d == 1 and l[i] <= p_lim) or (d == -1 and h[i] >= p_lim)):
                    px = min(o[i], p_lim) if d == 1 else max(o[i], p_lim)
                if np.isnan(px) and (i >= p_until or force_exit[i]):
                    pending = 0  # order expired / session over
            if not np.isnan(px):
                pos = d
                start = i
                fill = px
                stop = fill - pos * p_stop if p_stop > 0 else np.nan
                tgt = fill + pos * p_tgt if p_tgt > 0 else np.nan
                trail = p_trail
                hold = p_hold
                best = fill
                worst = fill
                pending = 0

        if pos != 0:
            exit_px = np.nan
            by_signal = False
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
            if np.isnan(exit_px) and (force_exit[i] or (hold > 0 and i - start >= hold)):
                exit_px = c[i] - pos * slip
            if np.isnan(exit_px) and exit_sig[i] == pos:
                exit_px = c[i] - pos * slip
                by_signal = True
            if not np.isnan(exit_px):
                e_i[k] = start
                x_i[k] = i
                dirs[k] = pos
                e_px[k] = fill
                x_px[k] = exit_px
                mae[k] = min(0.0, (worst - fill) * pos * pv * qty) - comm * qty
                k += 1
                pos = 0
                if not by_signal:
                    continue  # no re-entry on the same bar a stop/target/time exit hit
            elif trail > 0:
                # ratchet the trailing stop after this bar (pessimistic: never helps the bar it is set on)
                if pos == 1:
                    best = max(best, h[i])
                    stop = best - trail if np.isnan(stop) else max(stop, best - trail)
                else:
                    best = min(best, l[i])
                    stop = best + trail if np.isnan(stop) else min(stop, best + trail)

        if pos == 0 and pending == 0 and entry[i] != 0 and not force_exit[i] and i + 1 < n:
            pending = entry[i]
            p_stop = stop_pts[i]
            p_tgt = target_pts[i]
            p_hold = max_hold[i]
            p_trail = trail_pts[i]
            p_lim = limit_px[i]
            p_stp = stop_px[i]
            p_cap = stop_cap[i]
            p_until = i + max(expiry[i], 1)

    pnl = (x_px[:k] - e_px[:k]) * dirs[:k] * pv * qty - comm * qty
    return e_i[:k], x_i[:k], dirs[:k], e_px[:k], x_px[:k], pnl, mae[:k]


def _opt(a, n, dtype, fill):
    return np.full(n, fill, dtype) if a is None else np.asarray(a, dtype)


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
        _opt(signals.exit_sig, len(bars), np.int8, 0),
        _opt(signals.max_hold, len(bars), np.int64, 0),
        _opt(signals.trail_pts, len(bars), np.float64, 0.0),
        _opt(signals.limit_px, len(bars), np.float64, np.nan),
        _opt(signals.stop_px, len(bars), np.float64, np.nan),
        _opt(signals.stop_cap, len(bars), np.float64, np.nan),
        _opt(signals.expiry, len(bars), np.int64, 1),
        inst.tick_size,
        slippage_ticks * inst.tick_size,
        inst.point_value,
        inst.commission_rt,
        qty,
    )
    return Trades(e_i, x_i, d, e_px, x_px, pnl, mae, bars.day[x_i])
