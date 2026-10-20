"""Strategy library. To add a strategy, drop a new module in this folder and list it in _MODULES."""

import itertools

from . import channel_breakout, ema_trend, orb, rsi_reversal, time_momentum, vwap_fade

_MODULES = [orb, vwap_fade, ema_trend, rsi_reversal, channel_breakout, time_momentum]
STRATEGIES = {m.NAME: m for m in _MODULES}


def get_strategy(name: str):
    try:
        return STRATEGIES[name]
    except KeyError:
        raise ValueError(f"Unknown strategy {name!r}. Known: {sorted(STRATEGIES)}")


def param_combos(module) -> list[dict]:
    keys = list(module.PARAM_GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(module.PARAM_GRID[k] for k in keys))]
