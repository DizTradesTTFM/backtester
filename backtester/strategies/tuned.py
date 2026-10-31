"""Tuned versions of Capo's closest QuantJourney strategies (asked for 2026-10-31: tweak them to pass prop evals).

Same entry idea as the original, plus a few levers the original didn't have:
- target_r: take profit at N x the stop distance (0 = original timed / signal exit)
- hours: "all" = 9:30-15:30 entries, "open" = only the first two hours (9:30-11:30)
- side: "long" = original long-only, "trend" = long above SMA(50), mirrored short below it
- hold_x: hold up to 1x or 2x the original number of bars
The 70/30 split and the luck bar still apply, and they count every extra combo tried.
"""

from types import SimpleNamespace

import numpy as np

from ..indicators import atr
from .quantjourney import chart, make, prev_max, prev_min, sma

WINDOWS = {"all": (930, 1530), "open": (930, 1130)}
U = 0.5  # pct_atr: 1% of the original = 0.5 ATR


def _common(bars, p):
    big, ends, a = chart(bars, p["tf"])
    up_trend = big.close > sma(big.close, 50)
    longs_ok = np.ones(len(big), bool) if p["side"] == "long" else up_trend
    shorts_ok = np.zeros(len(big), bool) if p["side"] == "long" else ~up_trend & np.isfinite(sma(big.close, 50))
    return big, ends, a, longs_ok, shorts_ok


def _finish(bars, big, ends, p, long, short, stop, **kw):
    tgt = stop * p["target_r"] if p["target_r"] > 0 else None
    return make(bars, big, ends, p["tf"], long, stop, short=short, target=tgt, window=WINDOWS[p["hours"]], **kw)


def _breakout(bars, p, hold, capped):
    big, ends, a, lk, sk = _common(bars, p)
    u = a * U
    hi = prev_max(big.close, 20) + 0.2 * u
    lo = prev_min(big.close, 20) - 0.2 * u
    long = np.isfinite(hi) & lk
    short = np.isfinite(lo) & sk & ~long
    px = np.where(long, hi, lo)
    cap = np.where(long, hi + 0.4 * u, lo - 0.4 * u) if capped else None
    return _finish(bars, big, ends, p, long, short, a * p["stop_atr"], stop_px=px, cap=cap, expiry=5,
                   hold=hold * p["hold_x"])


def o05t(bars, p):
    return _breakout(bars, p, 10, False)


def o07t(bars, p):
    return _breakout(bars, p, 12, True)


def o14t(bars, p):
    big, ends, a, lk, sk = _common(bars, p)
    u = a * U
    every5 = (np.arange(len(big)) % 5) == 4
    long = every5 & lk
    short = every5 & sk & ~long
    # long: buy limit below + buy stop above; short (mirror): sell limit above + sell stop below
    lim = np.where(long, big.close - 2.0 * u, big.close + 2.0 * u)
    stp = np.where(long, big.close + 2.0 * u, big.close - 2.0 * u)
    return _finish(bars, big, ends, p, long, short, a * p["stop_atr"], limit_px=lim, stop_px=stp, expiry=5,
                   hold=10 * p["hold_x"])


def o20t(bars, p):
    big, ends, a, lk, sk = _common(bars, {**p, "side": "trend" if p["side"] == "trend" else "long"})
    n = p["lookback"]
    hi, lo = prev_max(big.high, n), prev_min(big.low, n)
    up, dn = big.close > hi, big.close < lo
    if p["side"] == "trend":
        up, dn = up & lk, dn & sk
    a20 = atr(big.high, big.low, big.close, 20)
    return _finish(bars, big, ends, p, up, dn & ~up, a20 * p["stop_atr"], exit_long=big.close < lo,
                   exit_short=big.close > hi)


BASE = {"tf": [15, 30], "stop_atr": [1.0, 1.5, 2.0], "target_r": [0, 2, 3], "hours": ["all", "open"]}


def _s(name, fn, desc, extra):
    return SimpleNamespace(NAME=name, DESCRIPTION=desc, PARAM_GRID={**BASE, **extra}, generate=fn, MINE=True,
                           SOURCE="QuantJourney (tuned)")


TUNED = [
    _s("qj_o05_tuned_stop_breakout", o05t, "Tuned Orders #5: stop breakout above the 20-bar closing high, plus optional "
       "take profit, morning-only entries, and a trend filter that mirrors it short below SMA(50).",
       {"side": ["long", "trend"], "hold_x": [1, 2]}),
    _s("qj_o07_tuned_stop_limit_breakout", o07t, "Tuned Orders #7: stop-limit breakout (no chasing more than 0.4% past "
       "the level), plus optional take profit, morning-only entries, and a trend filter that mirrors it short.",
       {"side": ["long", "trend"], "hold_x": [1, 2]}),
    _s("qj_o14_tuned_oco_dip_breakout", o14t, "Tuned Orders #14: OCO dip-or-breakout orders every 5 bars, plus optional "
       "take profit, morning-only entries, and a trend filter that mirrors it short.",
       {"side": ["long", "trend"], "hold_x": [1, 2]}),
    _s("qj_o20_tuned_donchian", o20t, "Tuned Orders #20: Donchian breakout with a 20 or 55-bar channel, optional take "
       "profit, morning-only entries, and a trend filter (only trade with SMA(50)) to cut the blow-ups.",
       {"side": ["both", "trend"], "lookback": [20, 55]}),
]
