import numpy as np
import pandas as pd
import pytest

from backtester.data import bars_from_frame, make_sample_bars
from backtester.engine import Signals, Trades, run_backtest
from backtester.evaluate import EvalConfig, evaluate, luck_bar
from backtester.instruments import Instrument, get_instrument
from backtester.prop import PropRules, evaluate_prop
from backtester.strategies import STRATEGIES, param_combos

INST = Instrument("T", 1.0, 10.0, 2.0)  # 1 point tick, $10/point, $2 round trip


def make_bars(rows):
    """rows: list of (open, high, low, close), one minute apart on one RTH day."""
    ts = pd.date_range("2024-01-02 09:30", periods=len(rows), freq="1min", tz="America/New_York")
    o, h, l, c = zip(*rows)
    df = pd.DataFrame({"ts": ts.tz_convert("UTC"), "open": o, "high": h, "low": l, "close": c, "volume": 100})
    return bars_from_frame(df, "T")


def signals(n, entries, stop=5.0, target=10.0, force=None):
    e = np.zeros(n, np.int8)
    for i, d in entries.items():
        e[i] = d
    fx = np.zeros(n, bool)
    for i in force or []:
        fx[i] = True
    return Signals(e, np.full(n, stop), np.full(n, target), fx)


def test_long_hits_target_needs_trade_through():
    bars = make_bars([(100, 100, 100, 100), (100, 101, 99, 100), (100, 110, 99, 110), (110, 111, 109, 111)])
    t = run_backtest(bars, signals(4, {0: 1}), INST, slippage_ticks=1)
    # fill at 100 + 1 slip = 101, target 111; bar 2 high 110 does not trade through, bar 3 high 111 = touch only
    assert len(t) == 0

    bars = make_bars([(100, 100, 100, 100), (100, 101, 99, 100), (100, 112, 99, 112)])
    t = run_backtest(bars, signals(3, {0: 1}), INST, slippage_ticks=1)
    assert t.exit_px[0] == 111 and t.pnl[0] == pytest.approx(10 * 10 - 2)


def test_stop_first_when_both_hit_and_gap_through_stop():
    bars = make_bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 130, 80, 100)])
    t = run_backtest(bars, signals(3, {0: 1}, stop=5, target=5), INST, slippage_ticks=0)
    assert t.exit_px[0] == 95  # stop assumed first

    bars = make_bars([(100, 100, 100, 100), (100, 100, 100, 100), (90, 91, 88, 89)])
    t = run_backtest(bars, signals(3, {0: 1}, stop=5, target=50), INST, slippage_ticks=1)
    # filled at 101, stop at 96, gap open 90 -> filled at 90 - 1 slip
    assert t.exit_px[0] == 89
    assert t.mae[0] == pytest.approx((88 - 101) * 10 - 2)


def test_short_and_forced_exit():
    bars = make_bars([(100, 100, 100, 100), (100, 100, 99, 99), (99, 99, 97, 97)])
    t = run_backtest(bars, signals(3, {0: -1}, stop=50, target=50, force=[2]), INST, slippage_ticks=1)
    # short fill 100 - 1 = 99, forced out at close 97 + 1 = 98 -> +1 point
    assert t.direction[0] == -1 and t.exit_px[0] == 98 and t.pnl[0] == pytest.approx(10 - 2)


def test_no_entry_on_last_bar():
    bars = make_bars([(100, 100, 100, 100)] * 3)
    assert len(run_backtest(bars, signals(3, {2: 1}), INST)) == 0


def trades_from(days, pnl, mae=None):
    days = np.asarray(days, np.int64)
    pnl = np.asarray(pnl, float)
    mae = np.minimum(np.asarray(mae if mae is not None else np.minimum(pnl, 0), float), 0)
    z = np.zeros(len(days))
    return Trades(z, z, z, z, z, pnl, mae, days)


def test_prop_pass_fail_and_trailing():
    rules = PropRules("t", profit_target=1000, max_loss=500, trailing="eod", daily_loss_limit=0, max_days=10, min_days=1)
    # +600, +600 -> passes on day 1
    r = evaluate_prop(trades_from([0, 1], [600, 600]), 20, rules, step=100)
    assert r["prop_pass_rate"] == 1.0 and r["prop_days_to_pass"] == 2
    # +400 then -600: EOD trail moved floor to -100, so balance -200 fails
    r = evaluate_prop(trades_from([0, 1], [400, -600]), 20, rules, step=100)
    assert r["prop_fail_rate"] == 1.0
    # same but the loss happened intraday via MAE even though trade closed green
    r = evaluate_prop(trades_from([0, 1, 2], [0, 100, 1000], [0, -600, 0]), 20, rules, step=100)
    assert r["prop_fail_rate"] == 1.0


def test_prop_daily_loss_limit():
    rules = PropRules("t", 1000, 5000, "eod", daily_loss_limit=300, max_days=10, dll_fails=True)
    r = evaluate_prop(trades_from([0, 0], [-200, -200]), 20, rules, step=100)
    assert r["prop_fail_rate"] == 1.0


def test_luck_bar_grows_with_trials():
    assert luck_bar(1000, 500) > luck_bar(10, 500) > 0


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_strategy_runs_on_sample(name):
    bars = bars_from_frame(make_sample_bars(days=40), "ES")
    mod = STRATEGIES[name]
    p = param_combos(mod)[0]
    sig = mod.generate(bars, p)
    assert sig is not None
    assert len(sig.entry) == len(bars)
    row = evaluate(bars, sig, get_instrument("ES"), {"t": PropRules("t", 3000, 2000, "eod", 1000, 60)}, EvalConfig(), 100)
    assert row["verdict"] in {"PASS", "WATCH", "FAIL"}


def test_trades_are_flat_by_close():
    bars = bars_from_frame(make_sample_bars(days=20), "ES")
    mod = STRATEGIES["ema_trend"]
    t = run_backtest(bars, mod.generate(bars, param_combos(mod)[0]), get_instrument("ES"))
    assert len(t) > 0
    assert (bars.day[t.entry_i] == bars.day[t.exit_i]).all()
    assert (bars.minute[t.exit_i] <= 15 * 60 + 59).all()


def test_timeframe_signals_fire_on_bigger_bar_close():
    from backtester.strategies.base import to_timeframe

    bars = bars_from_frame(make_sample_bars(days=3), "ES")
    big, ends = to_timeframe(bars, 15)
    assert len(big) == 3 * 26  # 390 minutes / 15
    assert big.high[0] == bars.high[:15].max() and big.close[0] == bars.close[14]
    assert ends[0] == 14 and big.minute[0] == 9 * 60 + 45
    mod = STRATEGIES["ema_trend"]
    sig = mod.generate(bars, {**param_combos(mod)[0], "tf": 15})
    fired = np.flatnonzero(sig.entry)
    assert len(fired) and np.isin(fired, ends).all()


def test_fixed_risk_sizes_in_micros_and_skips_too_wide_stops():
    from backtester.evaluate import fixed_risk

    es = get_instrument("ES")
    # two ES trades: +4 points with a 3-point stop, -10 points with a 40-point stop
    t = Trades(np.array([1, 5]), np.array([3, 8]), np.array([1, -1]), np.array([100.0, 200.0]),
               np.array([104.0, 210.0]), np.array([195.5, -504.5]), np.array([-54.5, -504.5]), np.array([0, 1]))
    out = fixed_risk(t, np.array([3.0, 40.0]), es, risk=300, max_micros=50)
    # MES = $5/point: 300 / (3 * 5) = 20 micros; 40-point stop = $200 per micro -> 1 micro
    assert out.pnl[0] == pytest.approx((4 * 5 - 1.5) * 20)
    assert out.pnl[1] == pytest.approx((-10 * 5 - 1.5) * 1)
    assert len(fixed_risk(t, np.array([3.0, 80.0]), es, risk=300, max_micros=50)) == 1  # $400 per micro > $300
    assert fixed_risk(t, np.array([3.0, 40.0]), es, risk=300, max_micros=8).pnl[0] == pytest.approx((4 * 5 - 1.5) * 8)


def test_prop_consistency_lock_and_daily_pause():
    # consistency 50%: one +1000 day can't pass alone; passes once another day brings the best day to <= half
    rules = PropRules("t", 1000, 2000, "eod", 0, 10, consistency=0.5)
    assert evaluate_prop(trades_from([0], [1000]), 20, rules, step=100)["prop_pass_rate"] == 0
    r = evaluate_prop(trades_from([0, 1], [1000, 1000]), 20, rules, step=100)
    assert r["prop_pass_rate"] == 1.0 and r["prop_days_to_pass"] == 2
    # lock: floor stops rising at +100, so after +2000 the account survives a -1800 drop (floor would be +500 without it)
    rules = PropRules("t", 5000, 1000, "eod", 0, 10, lock_at=100)
    assert evaluate_prop(trades_from([0, 1, 2], [900, 1100, -1800]), 20, rules, step=100)["prop_fail_rate"] == 0
    # daily loss limit that pauses instead of failing: -600 trade is cut at -500, next day's trades still count
    rules = PropRules("t", 1000, 2000, "eod", 500, 10, dll_fails=False)
    r = evaluate_prop(trades_from([0, 0, 1], [-600, 900, 1600]), 20, rules, step=100)
    assert r["prop_pass_rate"] == 1.0


def test_runner_end_to_end_with_robustness(tmp_path):
    from backtester.runner import connect, run

    data = tmp_path / "ES_1m.parquet"
    make_sample_bars(days=60).to_parquet(data)
    db = tmp_path / "r.db"
    run("t", ["ES"], ["orb"], ["topstep_50k", "lucid_flex_25k"], workers=1, data_paths={"ES": data}, db_path=db)
    rows = connect(db).execute("SELECT metrics, verdict FROM results").fetchall()
    assert len(rows) == len(param_combos(STRATEGIES["orb"]))
    import json

    m = json.loads(rows[0][0])
    assert "robustness" in m and "f_lucid_flex_25k_oos_pass" in m and m["best_firm"] in ("topstep_50k", "lucid_flex_25k")


def test_daily_limits_skip_rest_of_day():
    from backtester.evaluate import daily_limits

    t = trades_from([0, 0, 0, 1, 1], [-200, -200, 500, 600, 100])
    kept = daily_limits(t, loss_stop=300, profit_lock=500)
    # day 0: -200, -200 (now -400 <= -300) -> third skipped; day 1: +600 >= 500 -> second skipped
    assert list(kept.pnl) == [-200, -200, 600]


def test_background_job_start_watch_stop(tmp_path, monkeypatch):
    import time

    from backtester import jobs

    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)
    jobs.start("t", ["-c", "import time; print('hello', flush=True); time.sleep(30)"])
    for _ in range(50):
        if "hello" in jobs.log_tail("t"):
            break
        time.sleep(0.1)
    assert jobs.running("t") and "hello" in jobs.log_tail("t")
    jobs.stop("t")
    time.sleep(0.3)
    assert not jobs.running("t")


def test_download_plan_only_fetches_new_days(tmp_path, monkeypatch):
    import scripts.download_data as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    assert dl.plan(["ES"], end="2024-03-01", update=True) == [("ES", dl.FIRST_DAY, "2024-03-01")]
    pd.DataFrame({"ts": pd.to_datetime(["2024-02-10 15:00"], utc=True)}).to_parquet(tmp_path / "ES_1m.parquet")
    assert dl.plan(["ES"], end="2024-03-01", update=True) == [("ES", "2024-02-11", "2024-03-01")]
    assert dl.plan(["ES"], end="2024-02-11", update=True) == []


def hourly_bars(candles, start="2024-01-02 06:00"):
    """candles: (open, high, low, close) per hour, expanded into 60 one-minute bars each."""
    rows = []
    for o, h, l, c in candles:
        path = np.r_[np.full(15, o), np.full(15, l), np.full(15, h), np.full(15, c)]
        rows += [(p, p, p, p) for p in path]
    ts = pd.date_range(start, periods=len(rows), freq="1min", tz="America/New_York")
    o, h, l, c = map(np.asarray, zip(*rows))
    df = pd.DataFrame({"ts": ts.tz_convert("UTC"), "open": o, "high": h, "low": l, "close": c, "volume": 1})
    return bars_from_frame(df, "ES"), df


def test_engine_limit_stop_entries_trail_hold_and_signal_exit():
    flat = (100, 100, 100, 100)
    # limit buy at 98: bar 1 low 99 no fill, bar 2 low 97 fills at 98 (no slippage)
    bars = make_bars([flat, (100, 101, 99, 100), (99, 99, 97, 98), (98, 99, 98, 99), flat])
    s = signals(5, {0: 1}, stop=50, target=0)
    s.limit_px = np.full(5, 98.0)
    s.expiry = np.full(5, 3)
    s.force_exit = np.zeros(5, bool)
    s.force_exit[4] = True
    t = run_backtest(bars, s, INST, slippage_ticks=1)
    assert t.entry_i[0] == 2 and t.entry_px[0] == 98
    # same order but it expires after 1 bar -> no trade
    s.expiry = np.full(5, 1)
    assert len(run_backtest(bars, s, INST, slippage_ticks=1)) == 0

    # buy-stop at 102: fills when the high reaches it, plus slippage; held max 1 bar then exits at the close
    bars = make_bars([flat, (101, 103, 101, 102), (102, 104, 102, 104), (104, 105, 104, 105), flat])
    s = signals(5, {0: 1}, stop=50, target=0)
    s.stop_px = np.full(5, 102.0)
    s.expiry = np.full(5, 5)
    s.max_hold = np.full(5, 1)
    t = run_backtest(bars, s, INST, slippage_ticks=1)
    assert t.entry_i[0] == 1 and t.entry_px[0] == 103 and t.exit_i[0] == 2 and t.exit_px[0] == 103

    # trailing stop 3 points: highs 105 then 110 -> stop 107; bar 4 low 106 stops out at 107 - slip
    bars = make_bars([flat, (100, 105, 100, 105), (105, 110, 105, 110), (110, 110, 109, 109), (109, 109, 106, 106)])
    s = signals(5, {0: 1}, stop=3, target=0)
    s.trail_pts = np.full(5, 3.0)
    t = run_backtest(bars, s, INST, slippage_ticks=1)
    assert t.exit_i[0] == 4 and t.exit_px[0] == 106

    # signal exit closes longs at the close, and a new entry on that same bar is still allowed
    bars = make_bars([flat] * 6)
    s = signals(6, {0: 1, 2: 1}, stop=50, target=0, force=[5])
    s.exit_sig = np.zeros(6, np.int8)
    s.exit_sig[2] = 1
    t = run_backtest(bars, s, INST, slippage_ticks=0)
    assert list(t.exit_i[:1]) == [2] and t.entry_i[1] == 3


def test_funded_win_days_rule():
    from backtester.prop import PropRules, evaluate_prop
    from backtester.engine import Trades
    import numpy as np
    n = 10
    pnl = np.full(n, 200.0)
    t = Trades(*[np.zeros(n) for _ in Trades.__dataclass_fields__])
    t = Trades(**{**t.__dict__, "day": np.arange(n), "pnl": pnl, "mae": np.zeros(n)})
    r = PropRules("x", 1, 2000, "eod", 1000, 60, 1, 5, 300, 0, False, 0.0, "funded", 5, 150)
    out = evaluate_prop(t, 70, r, step=100)
    assert out["prop_pass_rate"] == 1.0 and out["prop_days_to_pass"] == 5
    r2 = PropRules("x", 1, 2000, "eod", 1000, 60, 1, 5, 300, 0, False, 0.0, "funded", 5, 250)
    assert evaluate_prop(t, 70, r2, step=100)["prop_pass_rate"] == 0.0
