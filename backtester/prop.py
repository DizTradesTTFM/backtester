"""Prop firm evaluation simulator.

Takes a strategy's trades and replays them as if you had started an evaluation
on many different start days. Reports how often it passes, fails, or runs out of time.

Firm rules change often. Check the firm's current rules before trusting results.
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
    trailing: str  # "eod" = trails end-of-day balance highs, "intraday" = trails balance highs during the day
    daily_loss_limit: float  # 0 = no daily limit
    max_days: int  # trading days allowed before we call it a timeout
    min_days: int = 1  # minimum trading days before a pass counts
    max_minis: int = 5  # position limit in full-size contracts (10 micros = 1 mini)
    risk: float = 300.0  # dollars risked on every trade (stop distance x contracts)
    lock_at: float = 0.0  # the trailing floor stops rising here (dollars above the starting balance)
    dll_fails: bool = False  # True = hitting the daily loss limit fails the account; False = done for the day
    consistency: float = 0.0  # best day must be at most this share of total profit to pass (0 = no rule)

    def to_dict(self):
        return asdict(self)


# Rules as published by the firms / review sites in October 2026. Firms change these often: verify before buying.
# Apex: 30 calendar days ~ 21 trading days. Lucid and Topstep have no time limit; 60 trading days is our cut-off.
PROP_PRESETS = {
    "topstep_50k": PropRules("Topstep 50K", 3000, 2000, "eod", 1000, 60, 2, 5, 300, 0, False, 0.5),
    "lucid_flex_25k": PropRules("Lucid Flex 25K", 1250, 1000, "eod", 0, 60, 1, 2, 150, 100, False, 0.5),
    "lucid_pro_25k": PropRules("Lucid Pro 25K", 1250, 1000, "eod", 0, 60, 1, 2, 150, 100, False, 0.0),
    "lucid_flex_50k": PropRules("Lucid Flex 50K", 3000, 2000, "eod", 0, 60, 1, 4, 300, 100, False, 0.5),
    "lucid_pro_50k": PropRules("Lucid Pro 50K", 3000, 2000, "eod", 1200, 60, 1, 4, 300, 100, False, 0.0),
    "apex_25k_eod": PropRules("Apex 25K EOD", 1500, 1000, "eod", 500, 21, 1, 4, 150, 1500, False, 0.0),
    "apex_25k_intraday": PropRules("Apex 25K Intraday", 1500, 1000, "intraday", 0, 21, 1, 4, 150, 1500, False, 0.0),
    "apex_50k_eod": PropRules("Apex 50K EOD", 3000, 2000, "eod", 1000, 21, 1, 6, 300, 3000, False, 0.0),
    "apex_50k_intraday": PropRules("Apex 50K Intraday", 3000, 2000, "intraday", 0, 21, 1, 6, 300, 3000, False, 0.0),
}


@njit(cache=True)
def _simulate_evals(t_day, pnl, mae, n_days, target, max_loss, intraday_trail, dll, max_days, min_days, step,
                    lock_at, dll_fails, consistency):
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
        best_day = 0.0
        day_done = False
        traded_days = 0
        last_traded = -1
        res = 0
        used = 0
        j = j0
        while j < m and t_day[j] < s + max_days:
            d = t_day[j]
            if d != cur_day:
                # end of the previous day(s): trail the floor on closing balance, up to the lock level
                best_day = max(best_day, day_pnl)
                if not intraday_trail:
                    peak = max(peak, bal)
                    floor = min(peak - max_loss, lock_at)
                cur_day = d
                day_pnl = 0.0
                day_done = False
            if day_done:  # daily loss limit hit earlier today: no more trades
                j += 1
                continue
            if d != last_traded:
                traded_days += 1
                last_traded = d
            if bal + mae[j] <= floor:
                res = -1
                used = d - s + 1
                break
            if dll > 0 and day_pnl + mae[j] <= -dll:
                if dll_fails:
                    res = -1
                    used = d - s + 1
                    break
                # the firm flattens you at the daily limit and you are done for the day
                bal += -dll - day_pnl
                day_pnl = -dll
                day_done = True
                j += 1
                continue
            bal += pnl[j]
            day_pnl += pnl[j]
            if intraday_trail:
                peak = max(peak, bal)
                floor = min(peak - max_loss, lock_at)
            if bal <= floor:
                res = -1
                used = d - s + 1
                break
            if bal >= target and traded_days >= min_days:
                if consistency <= 0 or max(best_day, day_pnl) <= consistency * bal:
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
        float(rules.lock_at),
        rules.dll_fails,
        float(rules.consistency),
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
