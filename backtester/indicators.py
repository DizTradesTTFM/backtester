"""Indicators used by strategies. Session-aware ones reset each trading day."""

import numpy as np
from numba import njit


@njit(cache=True)
def ema(x, n):
    out = np.empty_like(x)
    a = 2.0 / (n + 1.0)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


@njit(cache=True)
def atr(h, l, c, n):
    tr = np.empty_like(c)
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    out = np.empty_like(c)
    out[0] = tr[0]
    for i in range(1, len(c)):
        out[i] = out[i - 1] + (tr[i] - out[i - 1]) / n
    return out


@njit(cache=True)
def rsi(c, n):
    out = np.full(len(c), 50.0)
    up = 0.0
    dn = 0.0
    for i in range(1, len(c)):
        ch = c[i] - c[i - 1]
        up += ((ch if ch > 0 else 0.0) - up) / n
        dn += ((-ch if ch < 0 else 0.0) - dn) / n
        if i >= n:
            out[i] = 100.0 if dn == 0 else 100.0 - 100.0 / (1.0 + up / dn)
    return out


@njit(cache=True)
def session_vwap(h, l, c, v, day):
    """VWAP and the running standard deviation of price around it, reset each day."""
    vwap = np.empty_like(c)
    sd = np.empty_like(c)
    pv = 0.0
    vol = 0.0
    p2v = 0.0
    for i in range(len(c)):
        if i == 0 or day[i] != day[i - 1]:
            pv = 0.0
            vol = 0.0
            p2v = 0.0
        tp = (h[i] + l[i] + c[i]) / 3.0
        w = max(v[i], 1.0)
        pv += tp * w
        p2v += tp * tp * w
        vol += w
        vwap[i] = pv / vol
        sd[i] = np.sqrt(max(p2v / vol - vwap[i] ** 2, 0.0))
    return vwap, sd


@njit(cache=True)
def rolling_max_prev(x, n):
    """Highest value of the previous n bars (excluding the current bar). NaN until n bars exist."""
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        out[i] = x[i - n : i].max()
    return out


@njit(cache=True)
def rolling_min_prev(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        out[i] = x[i - n : i].min()
    return out


def last_bar_of_day(day):
    out = np.zeros(len(day), dtype=np.bool_)
    out[:-1] = day[1:] != day[:-1]
    out[-1] = True
    return out


def hhmm(t: int) -> int:
    """930 -> minutes since midnight (570)."""
    return (t // 100) * 60 + t % 100
