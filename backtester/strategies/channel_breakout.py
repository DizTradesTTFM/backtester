from ..indicators import atr, rolling_max_prev, rolling_min_prev
from .base import build, default_force_exit, entry_window

NAME = "channel_breakout"
DESCRIPTION = "Channel breakout: buy a close above the highest high of the last N bars, short below the lowest low."
PARAM_GRID = {
    "lookback": [20, 45, 90],
    "stop_atr": [1.0, 2.0, 3.0],
    "rr": [1.0, 2.0, 3.0],
    "start": [945, 1030],
    "end": [1200, 1500],
}


def generate(bars, p):
    hi = rolling_max_prev(bars.high, p["lookback"])
    lo = rolling_min_prev(bars.low, p["lookback"])
    window = entry_window(bars, p["start"], p["end"])
    stop = atr(bars.high, bars.low, bars.close, 14) * p["stop_atr"]
    return build(bars, window & (bars.close > hi), window & (bars.close < lo), stop, stop * p["rr"], default_force_exit(bars))
