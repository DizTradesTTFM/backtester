import numpy as np

from .base import RTH_OPEN, build, default_force_exit, entry_window, first_per_day, to_timeframe

NAME = "orb"
DESCRIPTION = "Opening range breakout: trade the first break of the first N minutes' high/low."
PARAM_GRID = {
    "or_minutes": [5, 15, 30],
    "stop_mult": [0.5, 1.0],  # stop = this x opening range size
    "rr": [1.0, 1.5, 2.0, 3.0],  # target = rr x stop
    "entry_end": [1100, 1300],
    "side": ["both", "long"],
    "confirm_tf": [1, 5],  # breakout must close outside the range on this chart (minutes)
}


def generate(bars, p):
    n = len(bars)
    or_hi = np.full(n, np.nan)
    or_lo = np.full(n, np.nan)
    in_or = (bars.minute >= RTH_OPEN) & (bars.minute < RTH_OPEN + p["or_minutes"])
    after = bars.minute >= RTH_OPEN + p["or_minutes"]

    # opening range per day, then broadcast to the bars after it
    days = bars.day[in_or]
    if len(days):
        hi = np.full(bars.day[-1] + 1, np.nan)
        lo = np.full(bars.day[-1] + 1, np.nan)
        np.fmax.at(hi, days, bars.high[in_or])
        np.fmin.at(lo, days, bars.low[in_or])
        or_hi[after] = hi[bars.day[after]]
        or_lo[after] = lo[bars.day[after]]

    window = entry_window(bars, 930, p["entry_end"]) & after
    rng = or_hi - or_lo
    long_m = window & (bars.close > or_hi)
    short_m = window & (bars.close < or_lo)
    if p["confirm_tf"] > 1:
        _, ends = to_timeframe(bars, p["confirm_tf"])
        on_close = np.zeros(n, bool)
        on_close[ends] = True
        long_m &= on_close
        short_m &= on_close
    first = first_per_day(long_m | short_m, bars.day)
    stop = rng * p["stop_mult"]
    return build(bars, long_m & first, short_m & first, stop, stop * p["rr"], default_force_exit(bars), p["side"])
