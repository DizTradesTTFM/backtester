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
from .instruments import Instrument, get_instrument
from .metrics import compute_metrics
from .prop import PropRules, evaluate_prop


# Bump when scoring changes, so the next run starts a fresh batch instead of resuming an old-style one.
SCORING_VERSION = 6


@dataclass(frozen=True)
class EvalConfig:
    oos_fraction: float = 0.3  # last 30% of days are out-of-sample
    n_chunks: int = 6  # in-sample history is cut into this many periods for the stability check
    slippage_ticks: float = 1.0
    qty: int = 1
    min_trades: int = 100  # fewer in-sample trades than this = not enough evidence
    risk: float = 300.0  # dollars risked per trade for the headline stats (prop firms use their own risk)


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


def fixed_risk(trades: Trades, stop_pts: np.ndarray, inst: Instrument, risk: float, max_micros: int) -> Trades:
    """Re-size every trade so it risks the same dollars: as many micro contracts as fit inside `risk` at the stop.

    Trades whose stop is too wide for even one micro are skipped (you would not take them at that risk).
    """
    micro = get_instrument("M" + inst.data)
    pts = (trades.exit_px - trades.entry_px) * trades.direction
    worst_pts = (trades.mae / inst.point_value + inst.commission_rt / inst.point_value)  # <= 0, slippage included
    qty = np.minimum(np.floor(risk / (stop_pts * micro.point_value)), max_micros)
    qty = np.where(np.isfinite(qty), qty, 0)
    t = Trades(*(getattr(trades, f) for f in Trades.__dataclass_fields__))
    t.pnl = (pts * micro.point_value - micro.commission_rt) * qty
    t.mae = np.minimum(worst_pts * micro.point_value * qty, 0.0) - micro.commission_rt * qty
    keep = qty > 0
    return Trades(*(getattr(t, f)[keep] for f in Trades.__dataclass_fields__))


def trade_stops(trades: Trades, signals) -> np.ndarray:
    """Stop distance (points) each trade was entered with. Entries fill on the bar after the signal."""
    return np.asarray(signals.stop_pts, np.float64)[trades.entry_i - 1]


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


def firm_results(ins: Trades, oos: Trades, ins_stops, oos_stops, split, oos_days, inst, firms: dict) -> dict:
    """Run the same trades through every firm's evaluation, each at that firm's fixed risk per trade.

    The headline firm is the one with the best TRAINING pass rate, so its unseen pass rate stays honest.
    """
    out, best, best_key = {}, None, None
    for key, rules in firms.items():
        cap = rules.max_minis * 10
        p_in = evaluate_prop(fixed_risk(ins, ins_stops, inst, rules.risk, cap), split, rules)
        p_out = evaluate_prop(fixed_risk(oos, oos_stops, inst, rules.risk, cap), oos_days, rules)
        out.update({f"f_{key}_is_pass": p_in["prop_pass_rate"], f"f_{key}_oos_pass": p_out["prop_pass_rate"],
                    f"f_{key}_oos_fail": p_out["prop_fail_rate"], f"f_{key}_oos_days": p_out["prop_days_to_pass"]})
        score = p_in["prop_pass_rate"] if p_in["prop_pass_rate"] == p_in["prop_pass_rate"] else -1.0
        if best is None or score > best[0]:
            best, best_key = (score, p_in, p_out), key
    _, p_in, p_out = best
    out.update({f"is_{k}": v for k, v in p_in.items()})
    out.update({f"oos_{k}": v for k, v in p_out.items()})
    out["best_firm"] = best_key
    return out


def side_split(trades: Trades, prefix: str) -> dict:
    return {
        f"{prefix}_long_profit": float(trades.pnl[trades.direction > 0].sum()),
        f"{prefix}_short_profit": float(trades.pnl[trades.direction < 0].sum()),
    }


def evaluate(bars, signals, inst: Instrument, firms: dict, cfg: EvalConfig, n_trials: int) -> dict:
    """firms: {key: PropRules}. Headline stats are at a fixed cfg.risk dollars per trade, sized in micros."""
    raw = run_backtest(bars, signals, inst, cfg.slippage_ticks, 1)
    n_days = int(bars.day[-1]) + 1
    split = int(n_days * (1 - cfg.oos_fraction))
    stops = trade_stops(raw, signals)
    raw_in, raw_out = raw.day < split, raw.day >= split
    pick = lambda t, m: Trades(*(getattr(t, f)[m] for f in Trades.__dataclass_fields__))
    r_ins, r_oos = pick(raw, raw_in), shift_days(pick(raw, raw_out), split)
    trades = fixed_risk(raw, stops, inst, cfg.risk, 10**6)

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
    row.update(firm_results(r_ins, r_oos, stops[raw_in], stops[raw_out], split, n_days - split, inst, firms))
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
