from ..indicators import atr, session_vwap
from .base import build, default_force_exit, entry_window

NAME = "vwap_fade"
DESCRIPTION = "Fade stretches away from VWAP: short when price is far above it, long when far below."
PARAM_GRID = {
    "band": [1.5, 2.0, 2.5, 3.0],  # how many standard deviations from VWAP
    "stop_atr": [1.0, 2.0, 3.0],
    "rr": [1.0, 1.5, 2.0],
    "start": [1000, 1030],
    "atr_len": [14],
}


def generate(bars, p):
    vwap, sd = session_vwap(bars.high, bars.low, bars.close, bars.volume, bars.day)
    a = atr(bars.high, bars.low, bars.close, p["atr_len"])
    window = entry_window(bars, p["start"], 1530) & (sd > 0)
    long_m = window & (bars.close < vwap - p["band"] * sd)
    short_m = window & (bars.close > vwap + p["band"] * sd)
    stop = a * p["stop_atr"]
    return build(bars, long_m, short_m, stop, stop * p["rr"], default_force_exit(bars))
