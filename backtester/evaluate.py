"""Run one strategy/settings combo and score it: in-sample, out-of-sample, stability, prop rules, luck check.

Every strategy here is flat by the end of each day, so one backtest over the full history
can be split by day into the in-sample period (used for ranking) and the out-of-sample
period (never used for ranking; it is the honest test).
"""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numba import njit

from .engine import Trades, run_backtest
from .instruments import Instrument
from .metrics import compute_metrics
from .prop import PropRules, evaluate_prop


# Bump when scoring changes, so the next run starts a fresh batch instead of resuming an old-style one.
SCORING_VERSION = 3


@dataclass(frozen=True)
class EvalConfig:
    oos_fraction: float = 0.3  # last 30% of days are out-of-sample
    n_chunks: int = 6  # in-sample history is cut into this many periods for the stability check
    slippage_ticks: float = 1.0
    qty: int = 1
    min_trades: int = 100  # fewer in-sample trades than this = not enough evidence
    sizes: tuple = (1, 2, 3, 4, 5, 6, 8, 10)  # contract counts tried for the prop evaluation
    loss_stops: tuple = (0, 300, 500, 750)  # stop trading for the day after losing this much ($, 0 = off)
    profit_locks: tuple = (0, 500, 1000, 1500)  # stop trading for the day after making this much ($, 0 = off)


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


def scaled(trades: Trades, qty: int) -> Trades:
    t = Trades(*(getattr(trades, f) for f in Trades.__dataclass_fields__))
    t.pnl = trades.pnl * qty
    t.mae = trades.mae * qty
    return t


@njit(cache=True)
def _daily_limit_mask(day, pnl, loss_stop, profit_lock):
    keep = np.ones(len(pnl), np.bool_)
    cur = -1
    acc = 0.0
    for j in range(len(pnl)):
        if day[j] != cur:
            cur = day[j]
            acc = 0.0
        if (loss_stop > 0 and acc <= -loss_stop) or (profit_lock > 0 and acc >= profit_lock):
            keep[j] = False
        else:
            acc += pnl[j]
    return keep


def daily_limits(trades: Trades, loss_stop: float, profit_lock: float) -> Trades:
    """Skip the rest of a day's trades once the day is down loss_stop or up profit_lock dollars."""
    if loss_stop <= 0 and profit_lock <= 0:
        return trades
    m = _daily_limit_mask(trades.day, trades.pnl, float(loss_stop), float(profit_lock))
    return Trades(*(getattr(trades, f)[m] for f in Trades.__dataclass_fields__))


def best_prop_setup(ins: Trades, oos: Trades, split: int, oos_days: int, rules: PropRules, cfg, micro: bool) -> dict:
    """Pick contract count + daily loss stop + daily profit lock with the best TRAINING pass rate,
    then report unseen results with that exact setup.

    The setup is chosen on training data only, so the unseen pass rate stays honest.
    """
    limit = rules.max_minis * (10 if micro else 1)
    sizes = [q for q in cfg.sizes if q <= limit] or [1]
    def score(p):  # higher pass rate first, then fewer days to pass
        rate = p["prop_pass_rate"] if p["prop_pass_rate"] == p["prop_pass_rate"] else 0.0
        days = p["prop_days_to_pass"] if p["prop_days_to_pass"] == p["prop_days_to_pass"] else 1e9
        return (round(rate, 6), -days)

    best, best_p = (sizes[0], 0, 0), None
    for q in sizes:
        sized = scaled(ins, q)
        for ls in cfg.loss_stops:
            for pl in cfg.profit_locks:
                p = evaluate_prop(daily_limits(sized, ls, pl), split, rules)
                if best_p is None or score(p) > score(best_p):
                    best, best_p = (q, ls, pl), p
    q, ls, pl = best
    out = {f"is_{k}": v for k, v in best_p.items()}
    out.update({f"oos_{k}": v for k, v in evaluate_prop(daily_limits(scaled(oos, q), ls, pl), oos_days, rules).items()})
    out.update(contracts=q, daily_loss_stop=ls, daily_profit_lock=pl)
    return out


def side_split(trades: Trades, prefix: str) -> dict:
    return {
        f"{prefix}_long_profit": float(trades.pnl[trades.direction > 0].sum()),
        f"{prefix}_short_profit": float(trades.pnl[trades.direction < 0].sum()),
    }


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

    bar = luck_bar(n_trials, split)

    row = {f"is_{k}": v for k, v in m_in.items()}
    row.update({f"oos_{k}": v for k, v in m_out.items()})
    row.update(best_prop_setup(ins, oos, split, n_days - split, rules, cfg, bool(inst.data_symbol)))
    row.update(side_split(ins, "is"))
    row.update(side_split(oos, "oos"))
    row["stability"] = stability
    row["luck_bar"] = bar
    row["verdict"], row["reason"] = verdict(row, cfg, bar)
    return row


def verdict(r: dict, cfg: EvalConfig, bar: float, robustness: float | None = None) -> tuple[str, str]:
    """PASS only if the strategy clears every filter. Reasons are in plain English.

    robustness = share of neighboring settings (one notch different) that were also profitable in training.
    It is only known once the whole batch has finished, so the runner re-scores everything at the end.
    """
    if r["is_trades"] < cfg.min_trades:
        return "FAIL", f"too few trades ({r['is_trades']})"
    if r["is_net_profit"] <= 0:
        return "FAIL", "loses money in training"
    if r["stability"] < 4 / 6:
        return "FAIL", f"only profitable in {r['stability']:.0%} of periods"
    if r["oos_trades"] < 20:
        return "FAIL", "too few unseen trades"
    if r["oos_net_profit"] <= 0 or not (r["oos_profit_factor"] >= 1.1):
        return "FAIL", "falls apart on unseen data"
    near_miss = False
    if not (r["is_sharpe"] > bar):
        # Neighboring settings are tested too, so the luck bar overstates luck when a whole area of settings works.
        if robustness is not None and robustness >= 0.6 and r["is_sharpe"] >= 0.85 * bar:
            near_miss = True
        else:
            return "FAIL", f"training Sharpe {r['is_sharpe']:.2f} below luck bar {bar:.2f}"
    if not (r["oos_prop_pass_rate"] >= 0.3):
        return "WATCH", "edge looks real, but weak prop pass rate"
    if near_miss:
        return "WATCH", "just under the luck bar, but neighboring settings also work"
    return "PASS", "survived every filter"
