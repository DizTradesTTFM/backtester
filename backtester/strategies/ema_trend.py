import numpy as np

from ..indicators import atr, ema
from .base import build, default_force_exit, entry_window, expand, to_timeframe

NAME = "ema_trend"
DESCRIPTION = "Moving average crossover: go with the trend when the fast EMA crosses the slow EMA."
PARAM_GRID = {
    "tf": [5, 15, 30],  # chart timeframe in minutes
    "fast": [5, 9, 20],
    "slow": [21, 50, 100],
    "stop_atr": [1.0, 2.0, 3.0],
    "rr": [1.0, 2.0, 3.0],
    "side": ["both", "long"],
}


def generate(bars, p):
    if p["fast"] >= p["slow"]:
        return None  # invalid combo, skipped
    big, ends = to_timeframe(bars, p["tf"])
    return expand(bars, ends, _signals(big, p))


def _signals(bars, p):
    f = ema(bars.close, p["fast"])
    s = ema(bars.close, p["slow"])
    above = f > s
    cross_up = np.zeros(len(bars), bool)
    cross_dn = np.zeros(len(bars), bool)
    cross_up[1:] = above[1:] & ~above[:-1]
    cross_dn[1:] = ~above[1:] & above[:-1]
    window = entry_window(bars, 945, 1530)
    stop = atr(bars.high, bars.low, bars.close, 14) * p["stop_atr"]
    return build(bars, window & cross_up, window & cross_dn, stop, stop * p["rr"], default_force_exit(bars), p["side"])
