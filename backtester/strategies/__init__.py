"""Strategy library. To add a strategy, drop a new module in this folder and list it in _MODULES."""

import itertools

from . import channel_breakout, ema_trend, orb, prop_nq, qj_o01_sma_cross, quantjourney, rsi_reversal, time_momentum, ttrades, tuned, vwap_fade

_MODULES = [orb, vwap_fade, ema_trend, rsi_reversal, channel_breakout, time_momentum, qj_o01_sma_cross, *quantjourney.PORTS, *tuned.TUNED, *prop_nq.PROP, *ttrades.TTRADES]
STRATEGIES = {m.NAME: m for m in _MODULES}
# Capo's strategies (the QuantJourney ports). The built-in examples stay in code but are not run by default.
MINE = [n for n in STRATEGIES if getattr(STRATEGIES[n], "MINE", False)]

_SMALL = {"rsi", "sma", "ema", "macd", "oco", "nq", "orb"}


def title(name: str) -> str:
    """qj_o02_rsi_reversion -> 'RSI Reversion' (short display name)."""
    parts = name.split("_")
    if parts[0] == "qj" and len(parts) > 2:
        parts = parts[2:]
    return " ".join(p.upper() if p in _SMALL else (p if p[0].isdigit() else p.capitalize()) for p in parts)


def tag(name: str) -> str:
    """qj_o02_... -> 'Orders #2', qj_w07_... -> 'Weights #7'."""
    parts = name.split("_")
    if parts[0] == "qj" and len(parts) > 1 and parts[1][1:].isdigit():
        return ("Orders" if parts[1][0] == "o" else "Weights") + f" #{int(parts[1][1:])}"
    if parts[0] == "nq":
        return "Prop (new)"
    return "Built-in"


def get_strategy(name: str):
    try:
        return STRATEGIES[name]
    except KeyError:
        raise ValueError(f"Unknown strategy {name!r}. Known: {sorted(STRATEGIES)}")


def param_combos(module) -> list[dict]:
    keys = list(module.PARAM_GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(module.PARAM_GRID[k] for k in keys))]
