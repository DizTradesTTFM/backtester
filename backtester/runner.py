"""Overnight batch runner.

Builds a queue of every (instrument, strategy, settings) combo, runs them on all CPU cores,
and saves each result to results/results.db as soon as it finishes. If the PC restarts,
running the same command again skips everything already done and continues.

    python -m backtester.runner --run nightly
"""

import argparse
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from .data import load_bars
from .evaluate import EvalConfig, evaluate
from .instruments import get_instrument
from .prop import PROP_PRESETS
from .strategies import STRATEGIES, get_strategy, param_combos

RESULTS_DB = Path(__file__).resolve().parent.parent / "results" / "results.db"

_BARS = {}
_CTX = {}


def connect(db_path: Path = RESULTS_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """CREATE TABLE IF NOT EXISTS runs (
            run TEXT PRIMARY KEY, started TEXT, finished TEXT, config TEXT, n_jobs INTEGER)"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS results (
            run TEXT, instrument TEXT, strategy TEXT, params TEXT, metrics TEXT, verdict TEXT, reason TEXT,
            PRIMARY KEY (run, instrument, strategy, params))"""
    )
    return con


def _init_worker(data_paths, ctx):
    for sym, path in data_paths.items():
        _BARS[sym] = load_bars(sym, path)
    _CTX.update(ctx)


def _run_job(job):
    inst_sym, strat_name, params = job
    inst = get_instrument(inst_sym)
    bars = _BARS[inst.data]
    sig = get_strategy(strat_name).generate(bars, params)
    if sig is None:
        return job, None
    row = evaluate(bars, sig, inst, _CTX["rules"], _CTX["cfg"], _CTX["n_trials"])
    return job, row


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def build_jobs(instruments, strategies):
    jobs = []
    for sym in instruments:
        for name in strategies:
            for p in param_combos(get_strategy(name)):
                jobs.append((sym.upper(), name, p))
    return jobs


def run(run_name, instruments, strategies, prop="topstep_50k", workers=None, cfg=EvalConfig(), data_paths=None, db_path=RESULTS_DB):
    con = connect(db_path)
    rules = PROP_PRESETS[prop]
    jobs = build_jobs(instruments, strategies)

    # the luck bar depends on how many combos were tried per instrument
    n_trials = max(1, len(jobs) // max(1, len(instruments)))

    done = {
        (r[0], r[1], r[2]) for r in con.execute("SELECT instrument, strategy, params FROM results WHERE run=?", (run_name,))
    }
    todo = [j for j in jobs if (j[0], j[1], json.dumps(j[2], sort_keys=True)) not in done]

    config = {"instruments": instruments, "strategies": strategies, "prop": rules.to_dict(), "eval": cfg.__dict__}
    con.execute(
        "INSERT OR IGNORE INTO runs (run, started, config, n_jobs) VALUES (?, ?, ?, ?)",
        (run_name, datetime.now().isoformat(timespec="seconds"), json.dumps(config), len(jobs)),
    )
    con.commit()
    print(f"Run '{run_name}': {len(jobs)} backtests total, {len(done)} already done, {len(todo)} to go.")
    if not todo:
        return

    needed = sorted({get_instrument(s).data for s in instruments})
    data_paths = data_paths or {s: None for s in needed}
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    ctx = {"rules": rules, "cfg": cfg, "n_trials": n_trials}

    t0 = time.time()
    finished = 0
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(data_paths, ctx)) as pool:
        futures = [pool.submit(_run_job, j) for j in todo]
        for fut in as_completed(futures):
            (sym, name, p), row = fut.result()
            finished += 1
            if row is not None:
                con.execute(
                    "INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (run_name, sym, name, json.dumps(p, sort_keys=True),
                     json.dumps({k: _clean(v) for k, v in row.items()}), row["verdict"], row["reason"]),
                )
            if finished % 25 == 0 or finished == len(todo):
                con.commit()
                rate = finished / (time.time() - t0)
                left = (len(todo) - finished) / rate if rate else 0
                print(f"  {finished}/{len(todo)} done, {rate:.1f}/sec, ~{left / 60:.0f} min left", flush=True)

    con.execute("UPDATE runs SET finished=? WHERE run=?", (datetime.now().isoformat(timespec="seconds"), run_name))
    con.commit()
    n_pass = con.execute("SELECT COUNT(*) FROM results WHERE run=? AND verdict='PASS'", (run_name,)).fetchone()[0]
    print(f"Done. {n_pass} strategies passed every filter. Open the dashboard to see them.")


def main():
    ap = argparse.ArgumentParser(description="Run thousands of backtests and save the results.")
    ap.add_argument("--run", default="main", help="name for this batch. Re-using a name resumes it and skips finished backtests")
    ap.add_argument("--instruments", nargs="+", default=["ES", "NQ", "MES", "MNQ"])
    ap.add_argument("--strategies", nargs="+", default=["all"])
    ap.add_argument("--prop", default="topstep_50k", choices=sorted(PROP_PRESETS))
    ap.add_argument("--workers", type=int, default=None, help="CPU cores to use (default: all but one)")
    ap.add_argument("--slippage", type=float, default=1.0, help="slippage in ticks per side")
    args = ap.parse_args()

    strategies = sorted(STRATEGIES) if args.strategies == ["all"] else args.strategies
    run(args.run, args.instruments, strategies, args.prop, args.workers, EvalConfig(slippage_ticks=args.slippage))


if __name__ == "__main__":
    main()
