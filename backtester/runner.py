"""Overnight batch runner.

Builds a queue of every (instrument, strategy, settings) combo, runs them on all CPU cores,
and saves each result to results/results.db as soon as it finishes. If the PC restarts,
running the same command again skips everything already done and continues.

    python -m backtester.runner --run nightly
"""

import argparse
import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from .data import DATA_DIR, load_bars
from .evaluate import SCORING_VERSION, EvalConfig, evaluate, verdict
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


def default_run_name(instruments) -> str:
    """Same data files -> same name, so re-running resumes. New data -> a fresh batch."""
    h = hashlib.sha1()
    for sym in sorted({get_instrument(s).data for s in instruments}):
        f = DATA_DIR / f"{sym}_1m.parquet"
        if f.exists():
            st = f.stat()
            h.update(f"{sym}{st.st_size}{st.st_mtime_ns}".encode())
    kind = "SAMPLE" if (DATA_DIR / "SAMPLE_DATA").exists() else "real"
    return f"{kind}_{h.hexdigest()[:6]}_v{SCORING_VERSION}"


def finalize(con, run_name, cfg: EvalConfig):
    """Once a batch is done: score how many neighboring settings also worked, then re-judge every result."""
    rows = con.execute("SELECT instrument, strategy, params, metrics FROM results WHERE run=?", (run_name,)).fetchall()
    by_key = {
        (i, s, p): {k: (np.nan if v is None else v) for k, v in json.loads(m).items()} for i, s, p, m in rows
    }
    updates = []
    for (inst, strat, params), m in by_key.items():
        p = json.loads(params)
        neighbors = []
        for k, vals in get_strategy(strat).PARAM_GRID.items():
            if p.get(k) not in vals:
                continue
            i = vals.index(p[k])
            for j in (i - 1, i + 1):
                if 0 <= j < len(vals):
                    nm = by_key.get((inst, strat, json.dumps({**p, k: vals[j]}, sort_keys=True)))
                    if nm is not None:
                        neighbors.append(nm["is_net_profit"] > 0)
        rob = float(np.mean(neighbors)) if neighbors else None
        v, why = verdict(m, cfg, m["luck_bar"], rob)
        m.update(robustness=rob, verdict=v, reason=why)
        updates.append((json.dumps({k: _clean(x) for k, x in m.items()}), v, why, run_name, inst, strat, params))
    con.executemany(
        "UPDATE results SET metrics=?, verdict=?, reason=? WHERE run=? AND instrument=? AND strategy=? AND params=?",
        updates,
    )
    con.commit()


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
    con.execute("UPDATE runs SET config=?, n_jobs=? WHERE run=?", (json.dumps(config), len(jobs), run_name))
    con.commit()
    print(f"Run '{run_name}': {len(jobs)} backtests total, {len(done)} already done, {len(todo)} to go.")
    if todo:
        _run_todo(con, run_name, todo, instruments, rules, cfg, n_trials, workers, data_paths)
    finalize(con, run_name, cfg)
    con.execute("UPDATE runs SET finished=? WHERE run=?", (datetime.now().isoformat(timespec="seconds"), run_name))
    con.commit()
    counts = dict(con.execute("SELECT verdict, COUNT(*) FROM results WHERE run=? GROUP BY verdict", (run_name,)).fetchall())
    print(f"Done. {counts.get('PASS', 0)} passed every filter, {counts.get('WATCH', 0)} on the watch list. See the Results page.")


def _run_todo(con, run_name, todo, instruments, rules, cfg, n_trials, workers, data_paths):
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
    con.commit()


def main():
    ap = argparse.ArgumentParser(description="Run thousands of backtests and save the results.")
    ap.add_argument("--run", default=None, help="batch name (default: based on the data files). Re-using a name resumes it")
    ap.add_argument("--instruments", nargs="+", default=["ES", "NQ", "MES", "MNQ"])
    ap.add_argument("--strategies", nargs="+", default=["all"])
    ap.add_argument("--prop", default="topstep_50k", choices=sorted(PROP_PRESETS))
    ap.add_argument("--workers", type=int, default=None, help="CPU cores to use (default: all but one)")
    ap.add_argument("--slippage", type=float, default=1.0, help="slippage in ticks per side")
    args = ap.parse_args()

    strategies = sorted(STRATEGIES) if args.strategies == ["all"] else args.strategies
    name = args.run or default_run_name(args.instruments) + ("" if args.prop == "topstep_50k" else f"_{args.prop}")
    if not args.run and set(strategies) != set(STRATEGIES):
        name += "_" + "+".join(sorted(strategies))  # its own batch, so the luck bar matches what was tested
    run(name, args.instruments, strategies, args.prop, args.workers, EvalConfig(slippage_ticks=args.slippage))


if __name__ == "__main__":
    main()
