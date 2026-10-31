import numpy as np

from ..indicators import atr
from .base import build, default_force_exit, entry_window, expand, to_timeframe

MINE = True  # one of Capo's picks: selected by default on the Run page
SOURCE = "QuantJourney example_orders_01_market_sma_cross.py"
NAME = "qj_o01_sma_cross"
DESCRIPTION = (
    "QuantJourney orders #1, Market SMA Cross. Buy when SMA(20) crosses above SMA(50), sell when it crosses back below. "
    "Long only, like the original. Ported from daily ETF bars to ES/NQ intraday charts. The original has no stop, "
    "but fixed-dollar risk needs one, so a protective stop of N x ATR is added. Flat by 15:55."
)
PARAM_GRID = {
    "tf": [5, 15, 30, 60],  # chart timeframe in minutes (the original used daily bars)
    "stop_atr": [1.5, 2.0, 3.0],  # protective stop distance in ATRs (not in the original)
}
FAST, SLOW = 20, 50


def sma(x, n):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def generate(bars, p):
    big, ends = to_timeframe(bars, p["tf"])
    above = sma(big.close, FAST) > sma(big.close, SLOW)
    cross_up = np.zeros(len(big), bool)
    cross_dn = np.zeros(len(big), bool)
    cross_up[1:] = above[1:] & ~above[:-1]
    cross_dn[1:] = ~above[1:] & above[:-1]
    stop = atr(big.high, big.low, big.close, 14) * p["stop_atr"]
    window = entry_window(big, 930, 1530)
    sig = build(big, window & cross_up, np.zeros(len(big), bool), stop, 0.0, np.zeros(len(big), bool), "long")
    exit_on_cross = np.zeros(len(bars), bool)
    exit_on_cross[ends[cross_dn]] = True  # the reverse cross closes the trade
    return expand(bars, ends, sig, default_force_exit(bars) | exit_on_cross)
