import numpy as np

from . import _magic as mm

MINE = True  # one of Capo's own strategies: picked by default on the Run page
NAME = "magic_1h"
DESCRIPTION = (
    "Magic Tradez hourly model. A 1H candle (C2) sweeps liquidity and closes back strong: either a prior 1H swing "
    "low/high (ERL) or into an open 1H fair value gap (IRL). Enter at the next 1H open (C3). "
    "Stop beyond the protected swing (the C2 wick) or, refined, at the C2 body when that body is in the bottom half "
    "of the move (TTrades stop rules). Optional SMT (the other index did not sweep) and PSP (ES and NQ candles close "
    "opposite colors). The bubble indicator is private, so the C2 closure stands in for it."
)
PARAM_GRID = {
    "setup": ["erl", "irl"],
    "lookback": [3, 6, 12],  # ERL: swing low of the last N hours. IRL: FVG formed in the last N hours
    "hours": ["ny", "london_ny", "all"],
    "smt": [False, True],
    "psp": [False, True],
    "stop": ["swing", "body"],  # swing = beyond the protected swing (C2 wick); body = refined, at the C2 body
    "target": ["range", "1R", "2R", "3R"],  # range = highest high of the last N hours
}


def generate(bars, p):
    need_partner = p["smt"] or p["psp"]
    h1, ph1, ends = mm.hourly(bars, need_partner)
    if need_partner and ph1 is None:
        return None  # needs both ES and NQ data
    n = p["lookback"]
    ok_time = mm.hour_ok(h1.minute, p["hours"])
    partner = mm.flip(ph1) if ph1 is not None else [None, None]
    sides = []
    for (o, h, l, c), pc in zip(mm.flip(h1), partner):
        prev_l = np.r_[np.nan, l[:-1]]
        closure = (l < prev_l) & (c > prev_l) & (c > o)  # C2: takes the previous low, closes back strong
        swing_lo = _rolling(l, n, np.min)
        swing_hi = _rolling(h, n, np.max)
        if p["setup"] == "erl":
            level = swing_lo
            sig = closure & (l < swing_lo) & (c > swing_lo)
        else:
            bot, top = mm.open_fvgs(h, l, c, n)
            level = bot
            sig = closure & (l <= top) & (l >= bot) & (c > top)
        sig &= ok_time
        if pc is not None:
            po, ph, pl, pcl = pc
            if p["smt"]:
                sig &= pl >= np.r_[np.nan, pl[:-1]]  # the other index held its previous 1H low
            if p["psp"]:
                sig &= pcl < po
        stop = mm.stop_distance(p["stop"], o, c, l)
        sides.append((sig & np.isfinite(level), stop, mm.r_target(p["target"], stop, swing_hi - c)))
    return mm.finish(bars, ends, h1, sides[0], sides[1], "both")


def _rolling(x, n, fn):
    """fn over the previous n candles (not the current one)."""
    out = np.full(len(x), np.nan)
    w = np.lib.stride_tricks.sliding_window_view(x, n)
    out[n:] = fn(w[:-1], axis=1)
    return out

