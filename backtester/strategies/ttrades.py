"""TTrades fractal model (ICT style), Capo 2026-11-01: 4-hour framework, 15-minute execution.

4H key levels: a sweep of the previous 4H candle's high/low, or a tap of the newest unfilled 4H fair value gap.
Reversal: after a 4H tap, a 15M CISD (change in state of delivery) confirms the turn: a 15M CLOSE back through
the open of the first candle in the latest run of same-direction candles. Enter on that close, or on a limit
back at the CISD level. Continuation: after the reversal CISD, a pullback that holds the swept extreme, then
the next 15M CISD in the trade direction. Max 2 trades a day, flat by 3:55 PM.
"""

from types import SimpleNamespace

import numpy as np

from ..data import partner_bars
from ..engine import Signals
from ..indicators import hhmm
from .base import default_force_exit, to_timeframe

ARM_BARS = 16  # a 4H tap stays valid for 16 x 15M bars waiting for the CISD
CONT_BARS = 16  # after a reversal CISD, the pullback + second CISD must come within 16 x 15M bars
RETEST_BARS = 4  # a retest limit at the CISD level stays working for 4 x 15M bars
STOP_PAD = 1.0  # points beyond the extreme
WINDOWS = {
    "ny_am": [(830, 1200)],
    "ny": [(830, 1530)],
    "london_ny_am": [(200, 500), (830, 1200)],
    "killzones": [(200, 500), (830, 1100)],
    "ny_kz": [(830, 1100)],
    "ny_930_1200": [(930, 1200)],
    "ny_930_1530": [(930, 1530)],
    "asia_london": [(2000, 2400), (200, 500)],
    "asia_london_ny": [(2000, 2400), (200, 500), (830, 1100)],
}
CONT_1M = 60  # 1M continuation entry must come within 60 minutes of the 15M CISD
MIN_STOP_1M = 8.0  # points


def _fvgs(h4):
    """Newest unfilled bullish/bearish 4H gap known after each finished 4H candle: (top, bottom), NaN if none."""
    n = len(h4.close)
    bull = np.full((n, 2), np.nan)
    bear = np.full((n, 2), np.nan)
    cur_bull = cur_bear = None
    for i in range(n):
        if i >= 2 and h4.low[i] > h4.high[i - 2]:
            cur_bull = (h4.low[i], h4.high[i - 2])
        if i >= 2 and h4.high[i] < h4.low[i - 2]:
            cur_bear = (h4.low[i - 2], h4.high[i])
        # a 4H close through the far side fills (invalidates) the gap
        if cur_bull is not None and h4.close[i] < cur_bull[1]:
            cur_bull = None
        if cur_bear is not None and h4.close[i] > cur_bear[0]:
            cur_bear = None
        if cur_bull is not None:
            bull[i] = cur_bull
        if cur_bear is not None:
            bear[i] = cur_bear
    return bull, bear


def _daily(bars):
    """Daily candles from the 18:00 session open: (open, high, low, close) per trading day."""
    starts = np.flatnonzero(np.concatenate([[True], bars.day[1:] != bars.day[:-1]]))
    ends = np.concatenate([starts[1:] - 1, [len(bars) - 1]])
    nd = int(bars.day[-1]) + 1
    out = [np.full(nd, np.nan) for _ in range(4)]
    d = bars.day[starts]
    out[0][d] = bars.open[starts]
    out[1][d] = np.maximum.reduceat(bars.high, starts)
    out[2][d] = np.minimum.reduceat(bars.low, starts)
    out[3][d] = bars.close[ends]
    return out


def _easy_bias(dh, dl, dc):
    """TTrades Easy Daily Bias for each day from the previous day vs the day before: +1 bull, -1 bear, 0 none."""
    nd = len(dh)
    b = np.zeros(nd, np.int8)
    for d in range(2, nd):
        ph, pl, pc, qh, ql = dh[d - 1], dl[d - 1], dc[d - 1], dh[d - 2], dl[d - 2]
        if not np.isfinite(ph + pl + pc + qh + ql):
            continue
        above, below = ph > qh, pl < ql
        if above and below:
            continue  # swept both sides: no bias
        if pc > qh:
            b[d] = 1
        elif pc < ql:
            b[d] = -1
        elif above:
            b[d] = -1  # traded above but failed to close above
        elif below:
            b[d] = 1  # traded below but failed to close below
    return b


def _target(p, sp, px, prev_hi, prev_lo, s, pdh=np.nan, pdl=np.nan):
    """Target distance: fixed R multiple, the opposite end of the previous 4H candle, or the previous day's
    high/low (both need at least 1R of room)."""
    if p["target"] in ("htf", "pdh_pdl"):
        hi, lo = (prev_hi, prev_lo) if p["target"] == "htf" else (pdh, pdl)
        dist = (hi - px) if s == 0 else (px - lo)
        return dist if np.isfinite(dist) and dist >= sp else 0.0
    return sp * float(p["target"])


def fractal(bars, p):
    n = len(bars)
    m15, e15 = to_timeframe(bars, 15)
    h4, e4 = to_timeframe(bars, 240)
    bull_gap, bear_gap = _fvgs(h4)
    j4 = np.searchsorted(e4, e15, side="right") - 1  # last finished 4H candle at each 15M close
    wins = [(hhmm(a), hhmm(b)) for a, b in WINDOWS[p["window"]]]
    use_sweep = p["context"] in ("sweep", "either")
    use_gap = p["context"] in ("fvg", "either")
    rev = p["mode"] in ("reversal", "both")
    cont = p["mode"] in ("continuation", "both")
    retest = p["entry"] == "retest"
    e15m = p["entry"] in ("close", "retest", "15m_cisd", "both")
    e1m = p["entry"] in ("1m_cont", "both")
    bias = p.get("bias", "none")
    cisd_stop = p.get("stop", "extreme") == "cisd"
    max_stop = float(p.get("max_stop", 0))
    smt = bool(p.get("smt", False))
    max_day = int(p.get("max_day", 2))  # entry signals allowed per day (the engine still trades one at a time)
    d_o, d_h, d_l, d_c = _daily(bars)
    ebias = _easy_bias(d_h, d_l, d_c) if bias in ("daily", "daily+eq") else None
    es4 = es15 = None
    if smt:
        pb_ = partner_bars(bars)
        if pb_ is not None:
            es15, _ = to_timeframe(pb_, 15)
            es4, _ = to_timeframe(pb_, 240)

    entry = np.zeros(n, np.int8)
    stop = np.zeros(n)
    tgt = np.zeros(n)
    lim = np.full(n, np.nan)
    o, h, l, c = m15.open, m15.high, m15.low, m15.close
    # per side state, index 0 = bullish (long), 1 = bearish (short)
    armed = [-1, -1]  # bar the 4H tap happened (-1 = not armed)
    ext = [np.nan, np.nan]  # extreme since the tap (low for longs, high for shorts)
    level = [np.nan, np.nan]  # CISD level: open of the first candle in the latest down run (longs) / up run (shorts)
    last_dir = 0  # direction of the previous candle: +1 up close, -1 down close, 0 doji
    cstate = [-1, -1]  # bar of the reversal CISD while waiting for a continuation entry
    cext = [np.nan, np.nan]  # reversal extreme that must hold
    pb = [np.nan, np.nan]  # pullback extreme since the reversal CISD
    pb_seen = [False, False]
    arm_j = [-1, -1]  # 4H candle whose high/low was in play when the side armed
    es_ext = [np.nan, np.nan]  # ES extreme since arming, for SMT divergence
    trades_today = 0
    day = -1

    def place(k, s, ref, sp, prev_hi, prev_lo, i=None):
        """Queue an entry at the 15M close (or a retest limit at the CISD level), or at 1M bar i. True if placed."""
        i = e15[k] if i is None else i
        if entry[i] != 0 or sp <= 0 or (max_stop > 0 and sp > max_stop):
            return False
        dd = bars.day[i] - 1
        tp = _target(p, sp, ref, prev_hi, prev_lo, s, d_h[dd] if dd >= 0 else np.nan, d_l[dd] if dd >= 0 else np.nan)
        if tp <= 0:
            return False
        entry[i], stop[i], tgt[i] = (1 if s == 0 else -1), sp, tp
        if retest and i == e15[k]:
            lim[i] = ref
        return True

    def cont_1m(k, s, swing, prev_hi, prev_lo):
        """After the 15M CISD at bar k: a 1M pullback that holds the 15M swing, then a 1M CISD. True if placed."""
        bo, bh, bl, bc = bars.open, bars.high, bars.low, bars.close
        start = e15[k] + 1
        counter = -1 if s == 0 else 1
        lvl, last, seen, ext1 = np.nan, 0, False, np.nan
        for i in range(start, min(n, start + CONT_1M)):
            if bars.day[i] != bars.day[start - 1]:
                return False
            if (bl[i] < swing) if s == 0 else (bh[i] > swing):
                return False
            if seen and np.isfinite(lvl) and ((bc[i] > lvl) if s == 0 else (bc[i] < lvl)):
                if not any(a <= bars.minute[i] < b for a, b in wins):
                    return False
                sp = max((bc[i] - ext1) if s == 0 else (ext1 - bc[i]), 0.0) + STOP_PAD
                return place(k, s, bc[i], max(sp, MIN_STOP_1M), prev_hi, prev_lo, i)
            cur = 1 if bc[i] > bo[i] else (-1 if bc[i] < bo[i] else 0)
            if cur == counter:
                if last != counter:
                    lvl = bo[i]
                seen = True
            if seen:
                x = bl[i] if s == 0 else bh[i]
                ext1 = x if not np.isfinite(ext1) else (min(ext1, x) if s == 0 else max(ext1, x))
            if cur != 0:
                last = cur
        return False

    for k in range(len(c)):
        d = m15.day[k]
        if d != day:
            day = d
            trades_today = 0
            armed = [-1, -1]
            cstate = [-1, -1]
        j = j4[k]
        if j < 0:
            continue
        t = bars.minute[e15[k]]
        in_win = any(a <= t < b for a, b in wins) and trades_today < max_day
        prev_hi, prev_lo = h4.high[j], h4.low[j]

        # 1) 4H context arms a reversal
        tap = [False, False]
        if use_sweep:
            tap[0] |= l[k] < prev_lo
            tap[1] |= h[k] > prev_hi
        if use_gap:
            gt, gb = bull_gap[j]
            tap[0] |= bool(np.isfinite(gt) and l[k] <= gt and h[k] >= gb)
            gt, gb = bear_gap[j]
            tap[1] |= bool(np.isfinite(gt) and h[k] >= gb and l[k] <= gt)
        for s in (0, 1):
            if armed[s] >= 0:
                ext[s] = min(ext[s], l[k]) if s == 0 else max(ext[s], h[k])
                if es15 is not None:
                    es_ext[s] = min(es_ext[s], es15.low[k]) if s == 0 else max(es_ext[s], es15.high[k])
                if not tap[s] and k - armed[s] > ARM_BARS:
                    armed[s] = -1
            if tap[s] and armed[s] < 0:
                armed[s], arm_j[s] = k, j
                ext[s] = l[k] if s == 0 else h[k]
                if es15 is not None:
                    es_ext[s] = es15.low[k] if s == 0 else es15.high[k]

        # 2) CISD: a 15M close through the level of the run before this candle
        cisd = [bool(np.isfinite(level[0]) and c[k] > level[0]), bool(np.isfinite(level[1]) and c[k] < level[1])]
        fired = [False, False]
        allow = [True, True]  # bias filter
        if bias == "daily_open":  # with the session open (18:00) of the current trading day
            allow = [c[k] > d_o[d], c[k] < d_o[d]]
        elif ebias is not None:
            allow = [ebias[d] == 1, ebias[d] == -1]
            if bias == "daily+eq" and d >= 1:  # stay on the right side of the previous day's 50%
                eq = (d_h[d - 1] + d_l[d - 1]) / 2
                allow = [allow[0] and c[k] >= eq, allow[1] and c[k] <= eq]
        elif bias == "4h":  # the last finished 4H candle closed in its upper / lower half
            mid = (h4.high[j] + h4.low[j]) / 2
            allow = [h4.close[j] > mid, h4.close[j] < mid]

        for s in (0, 1):
            # continuation: a pullback that holds the reversal extreme, then another CISD
            if cstate[s] >= 0 and k > cstate[s]:
                broke = (l[k] < cext[s]) if s == 0 else (h[k] > cext[s])
                if broke or k - cstate[s] > CONT_BARS:
                    cstate[s] = -1
                elif cisd[s] and pb_seen[s]:
                    fired[s] = True
                    ref = level[s] if retest else c[k]
                    sx = (l[k] if s == 0 else h[k]) if cisd_stop else pb[s]
                    sp = (ref - sx if s == 0 else sx - ref) + STOP_PAD
                    if cont and in_win and allow[s] and place(k, s, ref, sp, prev_hi, prev_lo):
                        trades_today += 1
                        in_win = trades_today < max_day and in_win
                    cstate[s] = -1
                else:
                    if (c[k] < o[k]) if s == 0 else (c[k] > o[k]):
                        pb_seen[s] = True
                    if pb_seen[s]:
                        pb[s] = min(pb[s], l[k]) if s == 0 else max(pb[s], h[k])
            # reversal: armed by the 4H context, confirmed by the CISD
            if armed[s] >= 0 and cisd[s]:
                fired[s] = True
                ref = level[s] if retest else c[k]
                sx = (l[k] if s == 0 else h[k]) if cisd_stop else ext[s]
                sp = (ref - sx if s == 0 else sx - ref) + STOP_PAD
                div = True
                if es4 is not None:  # SMT: NQ took the 4H level but ES did not
                    jj = arm_j[s]
                    div = bool(es_ext[s] > es4.low[jj]) if s == 0 else bool(es_ext[s] < es4.high[jj])
                ok = rev and in_win and allow[s] and div
                if ok and e15m and place(k, s, ref, sp, prev_hi, prev_lo):
                    trades_today += 1
                if ok and e1m and trades_today < max_day and cont_1m(k, s, ext[s], prev_hi, prev_lo):
                    trades_today += 1
                armed[s] = -1
                cstate[s], cext[s], pb_seen[s] = k, ext[s], False
                pb[s] = np.inf if s == 0 else -np.inf

        # 3) reset a level once it fired, then track the runs: down candles (long CISD) / up candles (short CISD)
        for s in (0, 1):
            if fired[s]:
                level[s] = np.nan
        cur = 1 if c[k] > o[k] else (-1 if c[k] < o[k] else 0)
        if cur == -1 and last_dir != -1:
            level[0] = o[k]
        if cur == 1 and last_dir != 1:
            level[1] = o[k]
        if cur != 0:
            last_dir = cur

    if not retest:
        return Signals(entry, stop, tgt, default_force_exit(bars))
    return Signals(entry, stop, tgt, default_force_exit(bars), exit_sig=np.zeros(n, np.int8),
                   max_hold=np.zeros(n, np.int64), trail_pts=np.zeros(n), limit_px=lim, stop_px=np.full(n, np.nan),
                   stop_cap=np.full(n, np.nan), expiry=np.full(n, RETEST_BARS * 15, np.int64))


def _c2_setups(h4, poi):
    """4H Candle 2 closures at a point of interest, per 4H candle i (as Candle 2).

    Bullish C2: sweeps the previous candle's low and closes back above it, with the sweep into a POI: an unbroken
    4H swing low, or the newest unfilled bullish 4H FVG (poi="any" also accepts a bearish Candle 1). Bearish mirrors.
    Returns bull, bear boolean arrays.
    """
    n = len(h4.close)
    hi, lo, op, cl = h4.high, h4.low, h4.open, h4.close
    bull_gap, bear_gap = _fvgs(h4)
    bull = np.zeros(n, bool)
    bear = np.zeros(n, bool)
    sw_lo = sw_hi = np.nan  # latest confirmed, still unbroken 4H swing low / high
    for i in range(2, n):
        # swing at i-2 is confirmed once candle i-1 closed
        m = i - 2
        if m >= 1 and lo[m] < lo[m - 1] and lo[m] < lo[m + 1]:
            sw_lo = lo[m]
        if m >= 1 and hi[m] > hi[m - 1] and hi[m] > hi[m + 1]:
            sw_hi = hi[m]
        if np.isfinite(sw_lo) and lo[i - 1] < sw_lo:
            sw_lo = np.nan  # broken before Candle 2
        if np.isfinite(sw_hi) and hi[i - 1] > sw_hi:
            sw_hi = np.nan
        if lo[i] < lo[i - 1] and cl[i] > lo[i - 1]:
            gt, gb = bull_gap[i - 1]
            at = (np.isfinite(sw_lo) and lo[i] < sw_lo) or (np.isfinite(gt) and lo[i] <= gt)
            if poi == "any":
                at = at or cl[i - 1] < op[i - 1]
            bull[i] = at
        if hi[i] > hi[i - 1] and cl[i] < hi[i - 1]:
            gt, gb = bear_gap[i - 1]
            at = (np.isfinite(sw_hi) and hi[i] > sw_hi) or (np.isfinite(gb) and hi[i] >= gb)
            if poi == "any":
                at = at or cl[i - 1] > op[i - 1]
            bear[i] = at
    return bull, bear


def _cont_1m(bars, start, s, swing, wins):
    """1M continuation from 1M bar `start`: a 1M pullback that holds `swing`, then a 1M CISD within CONT_1M minutes.
    Returns (bar, entry price, stop distance) or None."""
    bo, bh, bl, bc = bars.open, bars.high, bars.low, bars.close
    n = len(bc)
    counter = -1 if s == 0 else 1
    lvl, last, seen, ext1 = np.nan, 0, False, np.nan
    for i in range(start, min(n, start + CONT_1M)):
        if bars.day[i] != bars.day[start - 1]:
            return None
        if (bl[i] < swing) if s == 0 else (bh[i] > swing):
            return None
        if seen and np.isfinite(lvl) and ((bc[i] > lvl) if s == 0 else (bc[i] < lvl)):
            if not any(a <= bars.minute[i] < b for a, b in wins):
                return None
            sp = max((bc[i] - ext1) if s == 0 else (ext1 - bc[i]), 0.0) + STOP_PAD
            return i, bc[i], max(sp, MIN_STOP_1M)
        cur = 1 if bc[i] > bo[i] else (-1 if bc[i] < bo[i] else 0)
        if cur == counter:
            if last != counter:
                lvl = bo[i]
            seen = True
        if seen:
            x = bl[i] if s == 0 else bh[i]
            ext1 = x if not np.isfinite(ext1) else (min(ext1, x) if s == 0 else max(ext1, x))
        if cur != 0:
            last = cur
    return None


def c2c3(bars, p):
    """TTrades Fractal Model the TTrades way: 4H Candle 2 closure at a POI, then trade Candle 3 on a lower-timeframe
    CISD while price holds the EQ of Candle 2's wick."""
    n = len(bars)
    h4, e4 = to_timeframe(bars, 240)
    mx, ex = to_timeframe(bars, p["exec_tf"])
    j_of = np.searchsorted(e4, ex, side="left")  # 4H candle each exec bar belongs to (Candle 3 candidate)
    bull, bear = _c2_setups(h4, p["poi"])
    hi, lo, op, cl = h4.high, h4.low, h4.open, h4.close
    wins = [(hhmm(a), hhmm(b)) for a, b in WINDOWS[p["window"]]]
    _, d_h, d_l, d_c = _daily(bars)
    ebias = _easy_bias(d_h, d_l, d_c) if p["bias"] == "daily" else None
    small_wick = p["wick"] == "small"
    one_min = p["entry"] == "1m_cont"

    entry = np.zeros(n, np.int8)
    stop = np.zeros(n)
    tgt = np.zeros(n)
    o, h, l, c = mx.open, mx.high, mx.low, mx.close
    cur_j = -1
    valid = [False, False]
    weq = [np.nan, np.nan]
    c2x = [np.nan, np.nan]  # Candle 2 extreme (stop option) and Candle 2 opposite end (target option)
    c2o = [np.nan, np.nan]
    eq1 = np.nan  # EQ of Candle 1's range
    ext = [np.nan, np.nan]  # Candle 3 extreme so far
    level = [np.nan, np.nan]
    last_dir = 0
    trades_today, day = 0, -1
    for k in range(len(c)):
        d = mx.day[k]
        if d != day:
            day, trades_today = d, 0
        j = j_of[k]
        if j != cur_j:  # a new 4H candle starts: it is Candle 3 if the one before was a Candle 2
            cur_j = j
            i = j - 1
            valid = [False, False]
            level = [np.nan, np.nan]
            last_dir = 0
            if i >= 2 and j < len(cl):
                rng = hi[i] - lo[i]
                for s, ok in ((0, bull[i]), (1, bear[i])):
                    if not ok or rng <= 0:
                        continue
                    wick = (min(op[i], cl[i]) - lo[i]) if s == 0 else (hi[i] - max(op[i], cl[i]))
                    if small_wick and wick > 0.5 * rng:
                        continue
                    valid[s] = True
                    weq[s] = (lo[i] + wick / 2) if s == 0 else (hi[i] - wick / 2)
                    c2x[s] = lo[i] if s == 0 else hi[i]
                    c2o[s] = hi[i] if s == 0 else lo[i]
                    ext[s] = np.nan
                eq1 = (hi[i - 1] + lo[i - 1]) / 2
        t = bars.minute[ex[k]]
        in_win = any(a <= t < b for a, b in wins) and trades_today < 2
        for s in (0, 1):
            if not valid[s]:
                continue
            ext[s] = (l[k] if not np.isfinite(ext[s]) else min(ext[s], l[k])) if s == 0 else \
                (h[k] if not np.isfinite(ext[s]) else max(ext[s], h[k]))
            if (c[k] < weq[s]) if s == 0 else (c[k] > weq[s]):
                valid[s] = False  # closed through the wick EQ: Candle 3 failed
                continue
            fire = np.isfinite(level[s]) and ((c[k] > level[s]) if s == 0 else (c[k] < level[s]))
            if not fire:
                continue
            valid[s] = False  # one entry per Candle 3 per side
            if not in_win or (ebias is not None and ebias[d] != (1 if s == 0 else -1)):
                continue
            if one_min:
                got = _cont_1m(bars, ex[k] + 1, s, ext[s], wins)
                if got is None:
                    continue
                bi, px, sp = got
            else:
                bi, px = ex[k], c[k]
                sx = c2x[s] if p["stop"] == "c2" else ext[s]
                sp = (px - sx if s == 0 else sx - px) + STOP_PAD
            if sp <= 0 or entry[bi] != 0:
                continue
            if p["target"] in ("c2", "eq"):
                lvl_t = c2o[s] if p["target"] == "c2" else eq1
                tp = (lvl_t - px) if s == 0 else (px - lvl_t)
                tp = tp if np.isfinite(tp) and tp >= sp else 0.0
            else:
                tp = sp * float(p["target"])
            if tp <= 0:
                continue
            entry[bi], stop[bi], tgt[bi] = (1 if s == 0 else -1), sp, tp
            trades_today += 1
        cur = 1 if c[k] > o[k] else (-1 if c[k] < o[k] else 0)
        if cur == -1 and last_dir != -1:
            level[0] = o[k]
        if cur == 1 and last_dir != 1:
            level[1] = o[k]
        if cur != 0:
            last_dir = cur
    return Signals(entry, stop, tgt, default_force_exit(bars))


TTRADES = [
    SimpleNamespace(
        NAME="nq_ttrades_fractal", generate=fractal, MINE=True, SOURCE="Claude (Capo's TTrades idea)",
        DESCRIPTION="TTrades fractal model (ICT): 4H framework, 15M execution. Price sweeps the previous 4H candle's "
                    "high/low or taps an unfilled 4H fair value gap, then a 15M CISD (a close back through the open of the "
                    "candle run that made the extreme) confirms the reversal. Enter on the CISD close or a retest of the "
                    "CISD level. Continuation: a pullback that holds the extreme, then another 15M CISD. Stop beyond the "
                    "extreme / pullback, target 1.5-3R or the other end of the 4H candle. Max 2 trades a day.",
        PARAM_GRID={"context": ["sweep", "fvg", "either"], "mode": ["reversal", "continuation", "both"],
                    "entry": ["close", "retest"], "target": [1.5, 2, 3, "htf"],
                    "window": ["ny_am", "ny", "london_ny_am"]},
    ),
    SimpleNamespace(
        NAME="nq_ttrades_v2", generate=fractal, MINE=True, SOURCE="Claude (Capo's TTrades idea)",
        DESCRIPTION="TTrades fractal model v2: the same 4H sweep/FVG + 15M CISD model, with the extra TTrades/ICT filters "
                    "as options: bias (only with the daily open, or with the last 4H candle's close), stop at the CISD "
                    "candle instead of the swept extreme, a 40-point max stop, kill zones (London 2-5 AM, NY 8:30-11 AM) "
                    "and SMT divergence with ES (NQ sweeps the 4H level, ES does not).",
        PARAM_GRID={"context": ["sweep", "either"], "mode": ["reversal", "both"], "entry": ["close"],
                    "target": [2, 3, "htf"], "bias": ["none", "daily_open", "4h"], "stop": ["extreme", "cisd"],
                    "max_stop": [0, 40], "smt": [False, True],
                    "window": ["killzones", "london_ny_am", "asia_london", "asia_london_ny"]},
    ),
    SimpleNamespace(
        NAME="nq_ttrades_v3", generate=fractal, MINE=True, SOURCE="Claude (Capo's TTrades idea)",
        DESCRIPTION="TTrades fractal model v3, NY session only: 4H sweep/FVG + 15M CISD, filtered by TTrades' Easy Daily "
                    "Bias (yesterday vs the day before; optional: stay on the right side of yesterday's 50%). Entry on the "
                    "15M CISD close, or the 1M continuation: a 1M pullback that holds the 15M swing, then a 1M CISD within "
                    "60 minutes (stop beyond the 1M pullback, min 8 points). Target 2-3R or yesterday's high/low.",
        PARAM_GRID={"context": ["sweep", "either"], "mode": ["reversal"], "entry": ["15m_cisd", "1m_cont", "both"],
                    "target": [2, 3, "pdh_pdl"], "bias": ["none", "daily", "daily+eq"], "stop": ["extreme", "cisd"],
                    "smt": [False, True], "window": ["ny_kz", "ny_930_1200", "ny_930_1530"]},
    ),
    SimpleNamespace(
        NAME="nq_ttrades_v4", generate=c2c3, MINE=True, SOURCE="Claude (Capo's TTrades idea)",
        DESCRIPTION="TTrades Fractal Model the TTrades way, NY only: a 4H Candle 2 sweeps the previous 4H high/low at a "
                    "point of interest (unbroken 4H swing or 4H FVG; optionally an opposing Candle 1) and closes back "
                    "inside. During Candle 3, enter on a 15M or 5M CISD (or a 1M CISD continuation after it) while price "
                    "holds the EQ of Candle 2's wick. Stop beyond the Candle 3 swing or Candle 2's extreme. Target 2-3R, "
                    "Candle 2's opposite end, or Candle 1's EQ. Optional: small-wick filter, Easy Daily Bias.",
        PARAM_GRID={"exec_tf": [15, 5], "entry": ["cisd", "1m_cont"], "target": [2, 3, "c2", "eq"],
                    "stop": ["swing", "c2"], "poi": ["strict", "any"], "wick": ["any", "small"],
                    "bias": ["none", "daily"], "window": ["ny_kz", "ny_930_1200", "ny_930_1530"]},
    ),
]
