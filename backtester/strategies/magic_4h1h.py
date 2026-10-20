import numpy as np

from . import _magic as mm

MINE = True  # one of Capo's own strategies: picked by default on the Run page
NAME = "magic_4h1h"
DESCRIPTION = (
    "Magic Tradez 4H + 1H model. A 4H candle (C2) sweeps the previous 4H candle's low/high with a small wick, "
    "then a 1H candle inside that 4H window closes strong back inside the range (1H C2 closure). Enter at the "
    "next 1H open (C3). Stop beyond the sweep. Optional SMT (ES vs NQ: only one of them swept) and PSP (ES and NQ "
    "1H candles close opposite colors). The bubble indicator is private, so the C2 closure stands in for it."
)
PARAM_GRID = {
    "hours": ["ny", "london_ny", "all"],  # when the 1H closure may happen
    "max_wick": [0.25, 0.5, 1.0],  # how deep the sweep may go, as a share of the previous 4H candle's range
    "first_hour": [True, False],  # True = only the first 1H candle of the 4H window counts; False = first 3
    "smt": ["none", "4h", "4h+1h"],
    "psp": [False, True],
    "target": ["range", "1R", "2R", "3R"],  # range = other end of the previous 4H candle
}


def generate(bars, p):
    need_partner = p["smt"] != "none" or p["psp"]
    h1, ph1, ends = mm.hourly(bars, need_partner)
    if need_partner and ph1 is None:
        return None  # needs both ES and NQ data
    open_min = (h1.minute - 60) % 1440  # each 1H candle's open time
    g = h1.day.astype(np.int64) * 10 + ((open_min - mm.SESSION_START) % 1440) // 240  # 4H window id
    new_g = np.concatenate([[True], g[1:] != g[:-1]])
    gid = np.cumsum(new_g) - 1  # 0, 1, 2... per 4H window
    pos = np.arange(len(g)) - np.flatnonzero(new_g)[gid]  # 0 = first 1H candle of its 4H window
    ok_time = mm.hour_ok(h1.minute, p["hours"]) & (pos < (1 if p["first_hour"] else 3)) & (gid > 0)

    sides = []
    partner = mm.flip(ph1) if ph1 is not None else [None, None]
    for (o, h, l, c), pc in zip(mm.flip(h1), partner):
        firsts = np.flatnonzero(new_g)
        g_hi = np.maximum.reduceat(h, firsts)
        g_lo = np.minimum.reduceat(l, firsts)
        prev_hi = np.r_[np.nan, g_hi[:-1]][gid]  # previous 4H candle (C1)
        prev_lo = np.r_[np.nan, g_lo[:-1]][gid]
        low_so_far = mm.group_cummin(l, gid)  # lowest point of this 4H candle so far (C2)
        depth = prev_lo - low_so_far
        prev_l = np.r_[np.nan, l[:-1]]
        sig = (
            ok_time
            & (depth > 0)
            & (depth <= p["max_wick"] * (prev_hi - prev_lo))
            & (c > prev_lo)  # closed back inside C1's range
            & (c > o)  # strong bullish close
            & (l < prev_l)
            & (c > prev_l)  # 1H two-candle closure
        )
        if pc is not None:
            po, ph, pl, pcl = pc
            if p["smt"] != "none":
                p_prev_lo = np.r_[np.nan, np.minimum.reduceat(pl, firsts)[:-1]][gid]
                sig &= mm.group_cummin(pl, gid) >= p_prev_lo  # the other index did NOT take its 4H low
                if p["smt"] == "4h+1h":
                    sig &= pl >= np.r_[np.nan, pl[:-1]]  # nor its previous 1H low
            if p["psp"]:
                sig &= pcl < po  # the other index's 1H candle closed the opposite color
        sig = _first_per_window(sig, gid)
        stop = c - low_so_far
        sides.append((sig, stop, mm.r_target(p["target"], stop, prev_hi - c)))
    return mm.finish(bars, ends, h1, sides[0], sides[1], "both")


def _first_per_window(sig, gid):
    idx = np.flatnonzero(sig)
    keep = np.ones(len(idx), bool)
    keep[1:] = gid[idx[1:]] != gid[idx[:-1]]
    out = np.zeros_like(sig)
    out[idx[keep]] = True
    return out
