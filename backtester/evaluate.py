"""Run one strategy/settings combo and score it: in-sample, out-of-sample, stability, prop rules, luck check.

Every strategy here is flat by the end of each day, so one backtest over the full history
can be split by day into the in-sample period (used for ranking) and the out-of-sample
period (never used for ranking; it is the honest test).
"""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .engine import Trades, run_backtest
from .instruments import Instrument
from .metrics import compute_metrics
from .prop import PropRules, evaluate_prop


@dataclass(frozen=True)
class EvalConfig:
    oos_fraction: float = 0.3  # last 30% of days are out-of-sample
    n_chunks: int = 6  # in-sample history is cut into this many periods for the stability check
    slippage_ticks: float = 1.0
    qty: int = 1
    min_trades: int = 100  # fewer in-sample trades than this = not enough evidence


def subset(trades: Trades, lo_day: int, hi_day: int) -> Trades:
    m = (trades.day >= lo_day) & (trades.day < hi_day)
    return Trades(*(getattr(trades, f)[m] for f in Trades.__dataclass_fields__))


def shift_days(trades: Trades, by: int) -> Trades:
    t = Trades(*(getattr(trades, f) for f in Trades.__dataclass_fields__))
    t.day = trades.day - by
    return t


def luck_bar(n_trials: int, n_days: int) -> float:
    """Annualized Sharpe the BEST of n_trials pure-luck strategies would be expected to reach.

    Based on the expected maximum of n_trials random draws (Bailey & Lopez de Prado).
    A strategy whose in-sample Sharpe is below this is indistinguishable from luck.
    """
    if n_trials < 2 or n_days < 2:
        return 0.0
    g = 0.5772156649
    nd = NormalDist()
    z = (1 - g) * nd.inv_cdf(1 - 1 / n_trials) + g * nd.inv_cdf(1 - 1 / (n_trials * math.e))
    return z * math.sqrt(252 / n_days)


def evaluate(bars, signals, inst: Instrument, rules: PropRules, cfg: EvalConfig, n_trials: int) -> dict:
    trades = run_backtest(bars, signals, inst, cfg.slippage_ticks, cfg.qty)
    n_days = int(bars.day[-1]) + 1
    split = int(n_days * (1 - cfg.oos_fraction))

    ins = subset(trades, 0, split)
    oos = shift_days(subset(trades, split, n_days), split)
    m_in = compute_metrics(ins, split)
    m_out = compute_metrics(oos, n_days - split)

    edges = np.linspace(0, split, cfg.n_chunks + 1).astype(int)
    chunk_pnl = [subset(ins, edges[k], edges[k + 1]).pnl.sum() for k in range(cfg.n_chunks)]
    stability = float(np.mean([p > 0 for p in chunk_pnl]))

    p_in = evaluate_prop(ins, split, rules)
    p_out = evaluate_prop(oos, n_days - split, rules)
    bar = luck_bar(n_trials, split)

    row = {f"is_{k}": v for k, v in m_in.items()}
    row.update({f"oos_{k}": v for k, v in m_out.items()})
    row.update({f"is_{k}": v for k, v in p_in.items()})
    row.update({f"oos_{k}": v for k, v in p_out.items()})
    row["stability"] = stability
    row["luck_bar"] = bar
    row["verdict"], row["reason"] = verdict(row, cfg, bar)
    return row


def verdict(r: dict, cfg: EvalConfig, bar: float) -> tuple[str, str]:
    """PASS only if the strategy clears every filter. Reasons are in plain English."""
    if r["is_trades"] < cfg.min_trades:
        return "FAIL", f"too few trades ({r['is_trades']})"
    if r["is_net_profit"] <= 0:
        return "FAIL", "loses money in-sample"
    if not (r["is_sharpe"] > bar):
        return "FAIL", f"in-sample Sharpe {r['is_sharpe']:.2f} below luck bar {bar:.2f}"
    if r["stability"] < 4 / 6:
        return "FAIL", f"only profitable in {r['stability']:.0%} of periods"
    if r["oos_trades"] < 20:
        return "FAIL", "too few out-of-sample trades"
    if r["oos_net_profit"] <= 0 or r["oos_profit_factor"] < 1.1:
        return "FAIL", "falls apart on unseen data"
    if not (r["oos_prop_pass_rate"] >= 0.3):
        return "WATCH", "real edge, but weak prop pass rate"
    return "PASS", "survived every filter"
