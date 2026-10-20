import numpy as np

from ..indicators import atr, hhmm
from .base import RTH_OPEN, build, default_force_exit, first_per_day

NAME = "time_momentum"
DESCRIPTION = "Time-of-day momentum: at a set time, trade in (or against) the direction of the day's move so far, exit at a set time."
PARAM_GRID = {
    "entry_time": [1000, 1030, 1100, 1330, 1430],
    "hold_minutes": [30, 60, 120],
    "mode": ["with", "against"],
    "stop_atr": [2.0, 4.0],
    "min_move_atr": [0.0, 3.0],  # only trade if the day already moved this many ATRs
}


def generate(bars, p):
    n = len(bars)
    at_entry = bars.minute == hhmm(p["entry_time"])
    # day's opening price at 9:30
    is_open = bars.minute == RTH_OPEN
    day_open = np.full(bars.day[-1] + 1, np.nan)
    day_open[bars.day[is_open]] = bars.open[is_open]
    move = bars.close - day_open[bars.day]
    a = atr(bars.high, bars.low, bars.close, 14)
    big = np.abs(move) >= p["min_move_atr"] * a
    up = at_entry & big & (move > 0)
    dn = at_entry & big & (move < 0)
    if p["mode"] == "against":
        up, dn = dn, up
    exit_min = hhmm(p["entry_time"]) + p["hold_minutes"]
    force = default_force_exit(bars) | (bars.minute >= exit_min)
    first = first_per_day(up | dn, bars.day)
    return build(bars, up & first, dn & first, a * p["stop_atr"], np.zeros(n), force)
