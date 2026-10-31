"""Ports of the QuantJourney example strategies (github.com/QuantJourneyOrg/quantjourney-bt, strategies/).

The originals trade ETFs (SPY/QQQ/IWM, sector ETFs) mostly on DAILY bars and size by % of capital. Here they trade
ES/NQ on intraday charts and every trade risks a fixed dollar amount, so a few things had to change:

- Chart: daily-bar originals run on 15/30/60-minute charts; intraday originals keep their own timeframe.
- Percent levels: on daily ETFs "1%" is roughly one day's range. On an intraday chart we use `pct_atr` ATRs
  for each 1% of the original (e.g. a 5% stop with pct_atr 0.5 = 2.5 ATR). Intraday originals keep real percents.
- Stops: fixed-dollar risk needs a stop. Originals without one get a protective stop of `stop_atr` ATRs.
- Flat by 15:55 New York, entries 9:30-15:30 (the ETFs only trade then).
- Skipped: monthly/quarterly rotations, cross-sectional ranking, risk parity, vol targeting, FX, pairs (two legs),
  walk-forward tools, and exact duplicates here (orders 08 = 06, 11 = 09, 18 = 01, weights 03 = orders 02),
  because stop-limit fill risk is not modeled.
"""

from types import SimpleNamespace

import numpy as np

from ..engine import Signals
from ..indicators import atr, ema, rsi
from .base import default_force_exit, entry_window, to_timeframe

DAILY_TF = [15, 30, 60]
STOPS = [1.5, 2.0, 3.0]
PCT = [0.5, 1.0]


# ---- indicators -----------------------------------------------------------------
def sma(x, n):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rolling_std(x, n):
    m = sma(x, n)
    m2 = sma(x * x, n)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def prev_max(x, n):
    """Highest of the previous n values (not including the current one)."""
    out = np.full(len(x), np.nan)
    if len(x) > n:
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        out[n:] = w[:-1].max(axis=1)
    return out


def prev_min(x, n):
    return -prev_max(-x, n)


def crosses(a, b):
    """(cross up, cross down) of series a against b."""
    above = a > b
    valid = np.isfinite(a) & np.isfinite(b)
    up = np.zeros(len(a), bool)
    dn = np.zeros(len(a), bool)
    up[1:] = above[1:] & ~above[:-1] & valid[1:] & valid[:-1]
    dn[1:] = ~above[1:] & above[:-1] & valid[1:] & valid[:-1]
    return up, dn


def state(enter, leave):
    """1 while in a position: switch on at `enter`, off at `leave` (enter wins on the same bar)."""
    out = np.zeros(len(enter), bool)
    on = False
    for i in range(len(enter)):
        if enter[i]:
            on = True
        elif leave[i]:
            on = False
        out[i] = on
    return out


# ---- turning chart-bar decisions into engine signals -----------------------------
def make(bars, big, ends, tf, long, stop, short=None, *, target=None, exit_long=None, exit_short=None, hold=0,
         trail=None, limit_px=None, stop_px=None, cap=None, expiry=1, window=(930, 1530), force=None):
    """Everything is per chart bar; hold/expiry are counted in chart bars. Entries only 9:30-15:30."""
    n, m = len(bars), len(big)
    windows = [window] if isinstance(window[0], int) else window
    win = np.zeros(m, bool)
    for w in windows:
        win |= entry_window(big, *w)
    short = np.zeros(m, bool) if short is None else short
    e = np.zeros(m, np.int8)
    e[long & win] = 1
    e[short & win & ~long] = -1
    stop = np.broadcast_to(np.asarray(stop, float), (m,))
    e[~np.isfinite(stop) | (stop <= 0)] = 0  # every trade needs a defined risk

    def put(vals, dtype, fill):
        out = np.full(n, fill, dtype)
        if vals is not None:
            out[ends] = np.broadcast_to(np.asarray(vals, dtype), (m,))
        return out

    entry = put(e, np.int8, 0)
    xs = np.zeros(m, np.int8)
    if exit_long is not None:
        xs[exit_long] = 1
    if exit_short is not None:
        xs[exit_short & (xs == 0)] = -1
    tgt = put(target, float, 0.0)
    tgt[~np.isfinite(tgt)] = 0.0
    return Signals(
        entry, put(stop, float, 0.0), tgt, default_force_exit(bars) if force is None else force,
        exit_sig=put(xs, np.int8, 0),
        max_hold=put(hold * tf, np.int64, 0),
        trail_pts=put(trail, float, 0.0),
        limit_px=put(limit_px, float, np.nan),
        stop_px=put(stop_px, float, np.nan),
        stop_cap=put(cap, float, np.nan),
        expiry=put(max(expiry, 1) * tf, np.int64, 1),
    )


def chart(bars, tf):
    big, ends = to_timeframe(bars, tf)
    return big, ends, atr(big.high, big.low, big.close, 14)


# ---- the strategies ---------------------------------------------------------------
def o02(bars, p):
    big, ends, a = chart(bars, p["tf"])
    r = rsi(big.close, 14)
    return make(bars, big, ends, p["tf"], r < 35, a * p["stop_atr"], exit_long=r > 60)


def o03(bars, p):
    big, ends, a = chart(bars, p["tf"])
    u = a * p["pct_atr"]
    r = rsi(big.close, 14)
    return make(bars, big, ends, p["tf"], r < 40, a * p["stop_atr"], target=3.0 * u + 1.5 * u,
                limit_px=big.close - 1.5 * u, expiry=3)


def o04(bars, p):
    big, ends, a = chart(bars, p["tf"])
    u = a * p["pct_atr"]
    trend = sma(big.close, 50)
    up = big.close > trend
    return make(bars, big, ends, p["tf"], up, a * p["stop_atr"], target=4.0 * u + 1.0 * u,
                limit_px=big.close - 1.0 * u, expiry=2, exit_long=big.close < trend)


def o05(bars, p):
    big, ends, a = chart(bars, p["tf"])
    u = a * p["pct_atr"]
    level = prev_max(big.close, 20) + 0.2 * u
    return make(bars, big, ends, p["tf"], np.isfinite(level), a * p["stop_atr"], stop_px=level, expiry=5, hold=10)


def o06(bars, p):
    big, ends, a = chart(bars, p["tf"])
    up, dn = crosses(sma(big.close, 20), sma(big.close, 50))
    return make(bars, big, ends, p["tf"], up, 5.0 * a * p["pct_atr"], exit_long=dn)


def o07(bars, p):
    big, ends, a = chart(bars, p["tf"])
    u = a * p["pct_atr"]
    level = prev_max(big.close, 20) + 0.2 * u
    return make(bars, big, ends, p["tf"], np.isfinite(level), a * p["stop_atr"], stop_px=level,
                cap=level + 0.4 * u, expiry=5, hold=12)


def o09(bars, p):
    big, ends, a = chart(bars, p["tf"])
    up, dn = crosses(sma(big.close, 20), sma(big.close, 50))
    t = 4.0 * a * p["pct_atr"]
    return make(bars, big, ends, p["tf"], up, t, trail=t, exit_long=dn)


def o10(bars, p):
    big, ends, a = chart(bars, p["tf"])
    r = rsi(big.close, 14)
    t = 5.0 * a * p["pct_atr"]
    return make(bars, big, ends, p["tf"], r < 38, t, trail=t, exit_long=r > 65)


def o12(bars, p):
    big, ends, a = chart(bars, p["tf"])
    u = a * p["pct_atr"]
    up, dn = crosses(sma(big.close, 20), sma(big.close, 50))
    return make(bars, big, ends, p["tf"], up, 3.0 * u, target=6.0 * u, exit_long=dn)


def o13(bars, p):
    big, ends, a = chart(bars, p["tf"])
    u = a * p["pct_atr"]
    r = rsi(big.close, 14)
    return make(bars, big, ends, p["tf"], r < 35, 2.0 * u, target=4.0 * u, exit_long=r > 70)


def o14(bars, p):
    big, ends, a = chart(bars, p["tf"])
    u = a * p["pct_atr"]
    every5 = (np.arange(len(big)) % 5) == 4
    return make(bars, big, ends, p["tf"], every5, a * p["stop_atr"], limit_px=big.close - 2.0 * u,
                stop_px=big.close + 2.0 * u, expiry=5, hold=10)


def o15(bars, p):
    big, ends, a = chart(bars, 5)
    r = rsi(big.close, 14)
    return make(bars, big, ends, 5, r < 30, 0.004 * big.close, target=0.006 * big.close)


def o16(bars, p):
    big, ends, a = chart(bars, 30)
    level = prev_max(big.high, 12) * 1.001
    return make(bars, big, ends, 30, np.isfinite(level), a * p["stop_atr"], stop_px=level, expiry=3, hold=8)


def o20(bars, p):
    big, ends, a = chart(bars, p["tf"])
    hi, lo = prev_max(big.high, 55), prev_min(big.low, 55)
    up, dn = big.close > hi, big.close < lo
    a20 = atr(big.high, big.low, big.close, 20)
    return make(bars, big, ends, p["tf"], up, a20 * p["stop_atr"], short=dn, exit_long=dn, exit_short=up)


def w01(bars, p):
    big, ends, a = chart(bars, p["tf"])
    up, dn = crosses(sma(big.close, 50), sma(big.close, 200))
    return make(bars, big, ends, p["tf"], up, a * p["stop_atr"], exit_long=dn)


def w07(bars, p):
    big, ends, a = chart(bars, 15)
    r = rsi(big.close, 14)
    return make(bars, big, ends, 15, r < 35, a * p["stop_atr"], exit_long=r > 55)


def w08(bars, p):
    big, ends, a = chart(bars, p["tf"])
    up, dn = crosses(ema(big.close, 9), ema(big.close, 21))
    return make(bars, big, ends, p["tf"], up, a * p["stop_atr"], exit_long=dn)


def w09(bars, p):
    big, ends, a = chart(bars, 60)
    up, dn = crosses(sma(big.close, 10), sma(big.close, 30))
    return make(bars, big, ends, 60, up, a * p["stop_atr"], exit_long=dn)


def w21(bars, p):
    big, ends, a = chart(bars, p["tf"])
    mid = sma(big.close, 20)
    lower = mid - 2 * rolling_std(big.close, 20)
    return make(bars, big, ends, p["tf"], big.close < lower, a * p["stop_atr"], exit_long=big.close > mid)


def w22(bars, p):
    big, ends, a = chart(bars, p["tf"])
    macd = ema(big.close, 12) - ema(big.close, 26)
    up, dn = crosses(macd, ema(macd, 9))
    return make(bars, big, ends, p["tf"], up, a * p["stop_atr"], exit_long=dn)


def w25(bars, p):
    big, ends, a = chart(bars, p["tf"])
    up, dn = crosses(sma(big.close, 63), sma(big.close, 252))
    return make(bars, big, ends, p["tf"], up, a * p["stop_atr"], short=dn, exit_long=dn, exit_short=up)


def _s(name, fn, desc, grid):
    return SimpleNamespace(NAME=name, DESCRIPTION=desc, PARAM_GRID=grid, generate=fn, MINE=True,
                           SOURCE="QuantJourney")


D = {"tf": DAILY_TF, "stop_atr": STOPS}
DP = {"tf": DAILY_TF, "pct_atr": PCT}
DPS = {"tf": DAILY_TF, "pct_atr": PCT, "stop_atr": STOPS}
PORTS = [
    _s("qj_o02_rsi_reversion", o02, "QuantJourney orders #2: buy when RSI(14) < 35, sell when RSI > 60. Long only. "
       "Protective ATR stop added.", D),
    _s("qj_o03_limit_rsi_dip", o03, "QuantJourney orders #3: when RSI(14) < 40, place a buy limit 1.5% below the close "
       "(good for 3 bars); take profit 3% above entry. Protective ATR stop added.", DPS),
    _s("qj_o04_trend_pullback", o04, "QuantJourney orders #4: above SMA(50), place a buy limit 1% below the close (2 bars); "
       "take profit 4% above entry; exit if the close drops under SMA(50). Protective ATR stop added.", DPS),
    _s("qj_o05_stop_breakout", o05, "QuantJourney orders #5: buy-stop 0.2% above the highest close of the last 20 bars "
       "(good for 5 bars); hold 10 bars then exit. Protective ATR stop added.", DPS),
    _s("qj_o06_sma_cross_stop", o06, "QuantJourney orders #6: buy on SMA(20) crossing above SMA(50) with a 5% stop; sell on "
       "the reverse cross. (Orders #8, the same with a stop-limit, behaves identically here.)", DP),
    _s("qj_o07_stop_limit_breakout", o07, "QuantJourney orders #7: like #5 but a stop-limit: buy-stop 0.2% above the 20-bar "
       "closing high, refused if it fills more than 0.4% above that; hold 12 bars. Protective ATR stop added.", DPS),
    _s("qj_o09_sma_cross_trail", o09, "QuantJourney orders #9: buy on SMA(20/50) cross up, 4% trailing stop, sell on the "
       "reverse cross. (Orders #11, a trailing stop-limit, behaves identically here.)", DP),
    _s("qj_o10_rsi_trail", o10, "QuantJourney orders #10: buy when RSI(14) < 38, 5% trailing stop, sell when RSI > 65.", DP),
    _s("qj_o12_bracket_trend", o12, "QuantJourney orders #12: buy on SMA(20/50) cross up with a bracket: take profit +6%, "
       "stop -3%; sell on the reverse cross.", DP),
    _s("qj_o13_bracket_rsi", o13, "QuantJourney orders #13: buy when RSI(14) < 35 with a bracket: +4% take profit, -2% stop; "
       "sell if RSI > 70.", DP),
    _s("qj_o14_oco_dip_breakout", o14, "QuantJourney orders #14: every 5 bars while flat, place two buy orders: a limit 2% "
       "below and a stop 2% above the close (first one to fill cancels the other, 5 bars); hold 10 bars. "
       "Protective ATR stop added.", DPS),
    _s("qj_o15_5m_bracket_rsi", o15, "QuantJourney orders #15 (native 5-minute): buy when RSI(14) < 30 with a tight bracket: "
       "+0.6% take profit, -0.4% stop.", {"variant": ["original"]}),
    _s("qj_o16_30m_stop_breakout", o16, "QuantJourney orders #16 (native 30-minute): buy-stop 0.1% above the 12-bar high "
       "(3 bars); hold 8 bars then exit. Protective ATR stop added.", {"stop_atr": STOPS}),
    _s("qj_o20_donchian", o20, "QuantJourney orders #20: Donchian breakout: long above the 55-bar high, short below the "
       "55-bar low, always flips to the new side. Protective stop of N x ATR(20) added.", D),
    _s("qj_w01_sma_50_200", w01, "QuantJourney weights #1: long while SMA(50) is above SMA(200). Protective ATR stop added.",
       D),
    _s("qj_w07_15m_rsi", w07, "QuantJourney weights #7 (native 15-minute): buy when RSI(14) < 35, hold until RSI > 55. "
       "Protective ATR stop added.", {"stop_atr": STOPS}),
    _s("qj_w08_ema_scalp", w08, "QuantJourney weights #8: long while EMA(9) is above EMA(21). Native 1-minute; also tried "
       "on 5-minute. Protective ATR stop added.", {"tf": [1, 5], "stop_atr": STOPS}),
    _s("qj_w09_1h_sma_trend", w09, "QuantJourney weights #9 (native 1-hour): long while SMA(10) is above SMA(30). "
       "Protective ATR stop added.", {"stop_atr": STOPS}),
    _s("qj_w21_bollinger", w21, "QuantJourney weights #21: buy a close below the lower Bollinger Band (20, 2), exit when "
       "price closes back above the middle line. Protective ATR stop added.", D),
    _s("qj_w22_macd", w22, "QuantJourney weights #22: long while MACD(12/26) is above its 9 signal line. "
       "Protective ATR stop added.", D),
    _s("qj_w25_futures_trend", w25, "QuantJourney weights #25: long when SMA(63) is above SMA(252), short when below, "
       "always in. Protective ATR stop added.", D),
]
