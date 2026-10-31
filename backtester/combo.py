"""Run one strategy on ES and NQ at the same time in ONE prop account, and compare with each market alone.

    python -m backtester.combo qj_o14_fast_oco_dip_breakout            # tries every setting combo, prints the top 10
    python -m backtester.combo qj_o14_tuned_oco_dip_breakout --params '{"tf": 30, ...}'

Each trade still risks the firm's fixed dollars. Pass rates are on the last 30% of the data window (new data);
the firm is picked on the first 70%. Note: the firm's contract limit is applied per trade, not across both markets.
"""

import argparse
import json

import numpy as np

from .data import load_bars
from .engine import Trades, run_backtest
from .evaluate import fixed_risk, shift_days, subset, trade_stops
from .instruments import get_instrument
from .prop import DEFAULT_FIRMS, PROP_PRESETS, evaluate_prop
from .strategies import get_strategy, param_combos

FIELDS = list(Trades.__dataclass_fields__)


def _raw(sym, bars, strat, params, slip):
    inst = get_instrument(sym)
    sig = strat.generate(bars, params)
    if sig is None:
        return None
    raw = run_backtest(bars, sig, inst, slip, 1)
    return inst, raw, trade_stops(raw, sig), bars.ts[raw.exit_i].asi8


def _merge(parts, date_index):
    """parts: [(Trades, dates, exit_ns)] -> one Trades on a shared day index, in time order."""
    cols = {f: np.concatenate([getattr(t, f) for t, _, _ in parts]) for f in FIELDS}
    cols["day"] = np.concatenate([np.searchsorted(date_index, d) for _, d, _ in parts]).astype(np.int64)
    order = np.lexsort((np.concatenate([x for _, _, x in parts]), cols["day"]))
    return Trades(**{f: v[order] for f, v in cols.items()})


def score(raws, bars_by, firms, split_frac=0.7):
    dates = np.unique(np.concatenate([bars_by[s].day_dates for s in raws]))
    n = len(dates)
    split = int(n * split_frac)
    out = {}
    for key in firms:
        r = PROP_PRESETS[key]
        parts = []
        for sym, (inst, raw, stops, ts) in raws.items():
            t = fixed_risk(raw, stops, inst, r.risk, r.max_minis * 10)
            keep = np.isin(raw.exit_i, t.exit_i) if len(t) != len(raw) else np.ones(len(raw), bool)
            parts.append((t, bars_by[sym].day_dates[t.day], ts[keep]))
        allt = _merge(parts, dates)
        ins, oos = subset(allt, 0, split), shift_days(subset(allt, split, n), split)
        p_in, p_out = evaluate_prop(ins, split, r), evaluate_prop(oos, n - split, r)
        out[key] = (p_in, p_out, float(ins.pnl.sum()), float(oos.pnl.sum()), len(oos) / max(1, n - split) * 21)
    evals = [k for k in firms if PROP_PRESETS[k].stage == "eval"] or list(firms)
    best = max(evals, key=lambda k: np.nan_to_num(out[k][0]["prop_pass_rate"], nan=-1))
    return best, out[best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy")
    ap.add_argument("--params", default=None, help="JSON settings; default = try every combo")
    ap.add_argument("--years", type=float, default=3)
    ap.add_argument("--slippage", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    strat = get_strategy(args.strategy)
    combos = [json.loads(args.params)] if args.params else param_combos(strat)
    bars_by = {s: load_bars(s, years=args.years) for s in ("ES", "NQ")}
    rows = []
    for p in combos:
        raws = {s: _raw(s, bars_by[s], strat, p, args.slippage) for s in bars_by}
        if any(v is None for v in raws.values()):
            continue
        for label, sub in (("ES+NQ", raws), ("ES", {"ES": raws["ES"]}), ("NQ", {"NQ": raws["NQ"]})):
            firm, (pi, po, is_pnl, oos_pnl, tpm) = score(sub, bars_by, DEFAULT_FIRMS)
            rows.append((label, p, firm, po, is_pnl, oos_pnl, tpm))
    # setups that made money on BOTH the older and the new data first (the rest are likely luck)
    rows.sort(key=lambda r: (not (r[4] > 0 and r[5] > 0), -np.nan_to_num(r[3]["prop_pass_rate"], nan=-1)))
    print(f"{'market':6} {'firm':34} {'pass':>5} {'blow':>5} {'time':>5} {'days':>5} {'tr/mo':>6} {'old $':>8} {'new $':>8}  settings")
    for label, p, firm, po, isp, oosp, tpm in rows[: args.top]:
        pr, fr = po["prop_pass_rate"], po["prop_fail_rate"]
        print(f"{label:6} {PROP_PRESETS[firm].name:34} {pr:5.0%} {fr:5.0%} {max(0, 1 - pr - fr):5.0%} "
              f"{po['prop_days_to_pass']:5.0f} {tpm:6.0f} {isp:8.0f} {oosp:8.0f}  {json.dumps(p)}")


if __name__ == "__main__":
    main()
