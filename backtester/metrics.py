"""Performance stats for a list of trades."""

import numpy as np

from .engine import Trades


def daily_pnl(trades: Trades, n_days: int) -> np.ndarray:
    out = np.zeros(n_days)
    np.add.at(out, trades.day, trades.pnl)
    return out


def compute_metrics(trades: Trades, n_days: int) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "net_profit": 0.0, "win_rate": np.nan, "profit_factor": np.nan, "avg_trade": np.nan,
                "max_drawdown": 0.0, "sharpe": np.nan, "trades_per_day": 0.0, "pct_days_profitable": np.nan}
    pnl = trades.pnl
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    equity = np.cumsum(pnl)
    dd = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:] - equity
    daily = daily_pnl(trades, n_days)
    traded = daily[daily != 0]
    sd = daily.std()
    return {
        "trades": int(n),
        "net_profit": float(equity[-1]),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else np.inf,
        "avg_trade": float(pnl.mean()),
        "max_drawdown": float(dd.max()),
        "sharpe": float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
        "trades_per_day": float(n / max(n_days, 1)),
        "pct_days_profitable": float((traded > 0).mean()) if len(traded) else np.nan,
    }
