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
    rules = PropRules("t", 1000, 5000, "eod", daily_loss_limit=300, max_days=10)
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
    row = evaluate(bars, sig, get_instrument("ES"), PropRules("t", 3000, 2000, "eod", 1000, 60), EvalConfig(), 100)
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


def test_best_prop_size_picks_on_training_and_respects_limit():
    from backtester.evaluate import best_prop_size

    rules = PropRules("t", profit_target=1000, max_loss=500, trailing="eod", daily_loss_limit=0, max_days=10, max_minis=2)
    # $60/day steady winner: 1 contract times out, bigger sizes pass faster; limit caps minis at 2
    ins = trades_from(range(40), [60] * 40)
    oos = trades_from(range(20), [60] * 20)
    out = best_prop_size(ins, oos, 40, 20, rules, (1, 2, 3, 4), micro=False)
    assert out["contracts"] == 2 and out["oos_prop_pass_rate"] == 1.0
    out = best_prop_size(ins, oos, 40, 20, rules, (1, 2, 3, 4), micro=True)
    assert out["contracts"] == 4


def test_runner_end_to_end_with_robustness(tmp_path):
    from backtester.runner import connect, run

    data = tmp_path / "ES_1m.parquet"
    make_sample_bars(days=60).to_parquet(data)
    db = tmp_path / "r.db"
    run("t", ["ES"], ["orb"], workers=1, data_paths={"ES": data}, db_path=db)
    rows = connect(db).execute("SELECT metrics, verdict FROM results").fetchall()
    assert len(rows) == len(param_combos(STRATEGIES["orb"]))
    import json

    m = json.loads(rows[0][0])
    assert "robustness" in m and "contracts" in m and "oos_long_profit" in m
