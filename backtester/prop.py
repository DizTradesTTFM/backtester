"""Prop firm evaluation simulator.

Takes a strategy's trades and replays them as if you had started an evaluation
on many different start days. Reports how often it passes, fails, or runs out of time.

Rule numbers below are EXAMPLES. Check the firm's current rules before trusting results.
"""

from dataclasses import asdict, dataclass

import numpy as np
from numba import njit

from .engine import Trades


@dataclass(frozen=True)
class PropRules:
    name: str
    profit_target: float  # dollars of profit needed to pass
    max_loss: float  # trailing max drawdown, dollars
    trailing: str  # "eod" = trails end-of-day balance highs, "intraday" = trails open equity highs
    daily_loss_limit: float  # 0 = no daily limit
    max_days: int  # trading days allowed before we call it a timeout
    min_days: int = 1  # minimum trading days before a pass counts

    def to_dict(self):
        return asdict(self)


PROP_PRESETS = {
    # Approximate 50K evaluation-style rules. Verify against the firm's site before relying on them.
    "topstep_50k": PropRules("Topstep 50K (verify)", 3000.0, 2000.0, "eod", 1000.0, 60, 2),
    "generic_50k_intraday": PropRules("Generic 50K intraday trailing", 3000.0, 2500.0, "intraday", 0.0, 60, 1),
}


@njit(cache=True)
def _simulate_evals(t_day, pnl, mae, n_days, target, max_loss, intraday_trail, dll, max_days, min_days, step):
    n_starts = (n_days + step - 1) // step
    outcome = np.zeros(n_starts, np.int8)  # 1 pass, -1 fail, 0 timeout/ran out of data
    days_used = np.zeros(n_starts, np.int64)
    m = len(pnl)
    s_idx = 0
    j0 = 0
    for s in range(0, n_days, step):
        while j0 < m and t_day[j0] < s:
            j0 += 1
        bal = 0.0
        peak = 0.0
        floor = -max_loss
        cur_day = s
        day_pnl = 0.0
        traded_days = 0
        last_traded = -1
        res = 0
        used = 0
        j = j0
        while j < m and t_day[j] < s + max_days:
            d = t_day[j]
            if d != cur_day:
                # end of the previous day(s): trail the floor on closing balance, never above start
                if not intraday_trail:
                    peak = max(peak, bal)
                    floor = min(peak - max_loss, 0.0)
                cur_day = d
                day_pnl = 0.0
            if d != last_traded:
                traded_days += 1
                last_traded = d
            if bal + mae[j] <= floor or (dll > 0 and day_pnl + mae[j] <= -dll):
                res = -1
                used = d - s + 1
                break
            bal += pnl[j]
            day_pnl += pnl[j]
            if intraday_trail:
                peak = max(peak, bal)
                floor = min(peak - max_loss, 0.0)
            if bal <= floor or (dll > 0 and day_pnl <= -dll):
                res = -1
                used = d - s + 1
                break
            if bal >= target and traded_days >= min_days:
                res = 1
                used = d - s + 1
                break
            j += 1
        if res == 0:
            used = max_days
            if s + max_days > n_days:
                res = 2  # not enough data left to judge; excluded from stats
        outcome[s_idx] = res
        days_used[s_idx] = used
        s_idx += 1
    return outcome[:s_idx], days_used[:s_idx]


def evaluate_prop(trades: Trades, n_days: int, rules: PropRules, step: int = 5) -> dict:
    """Start an evaluation every `step` trading days and replay. Returns pass/fail stats."""
    out, used = _simulate_evals(
        trades.day.astype(np.int64),
        trades.pnl,
        trades.mae,
        int(n_days),
        rules.profit_target,
        rules.max_loss,
        rules.trailing == "intraday",
        rules.daily_loss_limit,
        rules.max_days,
        rules.min_days,
        step,
    )
    valid = out != 2
    total = int(valid.sum())
    if total == 0:
        return {"prop_attempts": 0, "prop_pass_rate": np.nan, "prop_fail_rate": np.nan, "prop_days_to_pass": np.nan}
    passed = out == 1
    return {
        "prop_attempts": total,
        "prop_pass_rate": float(passed.sum() / total),
        "prop_fail_rate": float((out == -1).sum() / total),
        "prop_days_to_pass": float(np.median(used[passed])) if passed.any() else np.nan,
    }
