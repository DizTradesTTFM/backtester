import numpy as np

from ..indicators import atr, rsi
from .base import build, default_force_exit, entry_window

NAME = "rsi_reversal"
DESCRIPTION = "RSI reversal: buy when RSI crosses back up from oversold, short when it crosses down from overbought."
PARAM_GRID = {
    "length": [7, 14],
    "level": [20, 25, 30],  # oversold level; overbought is 100 - level
    "stop_atr": [1.0, 2.0, 3.0],
    "rr": [1.0, 1.5, 2.0],
    "start": [945, 1030],
}


def generate(bars, p):
    r = rsi(bars.close, p["length"])
    lo, hi = p["level"], 100 - p["level"]
    long_m = np.zeros(len(bars), bool)
    short_m = np.zeros(len(bars), bool)
    long_m[1:] = (r[:-1] < lo) & (r[1:] >= lo)
    short_m[1:] = (r[:-1] > hi) & (r[1:] <= hi)
    window = entry_window(bars, p["start"], 1530)
    stop = atr(bars.high, bars.low, bars.close, 14) * p["stop_atr"]
    return build(bars, window & long_m, window & short_m, stop, stop * p["rr"], default_force_exit(bars))
