"""Run page: start or stop a batch of backtests and watch progress."""

import re
import subprocess
import sys

import streamlit as st

from backtester import jobs
from backtester.data import DATA_DIR
from backtester.instruments import INSTRUMENTS
from backtester.prop import PROP_PRESETS
from backtester.strategies import STRATEGIES, param_combos
from app_pages import ui

JOB = "backtest"
TASK = "Backtester nightly"

ui.header("Run", "Pick → Start → Sleep", [("Your strategies", ui.BLUE), ("Prop rules", ui.AMBER), ("Every night", ui.GREEN)])

missing = sorted({i.data for i in INSTRUMENTS.values() if not (DATA_DIR / f"{i.data}_1m.parquet").exists()})
if missing:
    st.warning(f"No price data for {', '.join(missing)} yet. Go to **Data** first.")


@st.fragment(run_every=2)
def status():
    log = jobs.log_tail(JOB, 200)
    total = re.findall(r"(\d+) backtests total, (\d+) already done, (\d+) to go", log)
    prog = re.findall(r"(\d+)/(\d+) done, ([\d.]+)/sec, ~(\d+) min left", log)
    if jobs.running(JOB):
        if prog:
            done, todo, rate, left = prog[-1]
            if done == todo:
                st.progress(1.0, text="All backtests done. Scoring the results (checking neighbors, luck bar)...")
            else:
                text = f"{int(done):,} of {int(todo):,} backtests · {rate}/sec · about {left} min left"
                st.progress(int(done) / int(todo), text=text)
        else:
            st.progress(0.0, text="Starting up (loading data, warming up the engine)...")
        if st.button("Stop", type="secondary"):
            jobs.stop(JOB)
            st.rerun()
    elif log:
        if "Done." in log:
            st.success(log.strip().splitlines()[-1])
        elif "Traceback" in log or "Error" in log:
            st.error("The last run crashed. Send Claude a screenshot of this:")
            st.code(jobs.log_tail(JOB, 25))
        elif total:
            st.info("The last run was stopped before it finished. Press Start to pick up where it left off.")
    now = jobs.running(JOB)
    if st.session_state.get("was_running", now) != now:
        st.session_state["was_running"] = now
        st.rerun(scope="app")  # refresh the Start button when a run starts or ends
    st.session_state["was_running"] = now
    return now


with ui.card("Status", "Live", ui.GREEN):
    busy = status()

with ui.card("What to test", "01 · Setup", ui.BLUE):
    c1, c2 = st.columns(2)
    insts = c1.multiselect("Contracts", list(INSTRUMENTS), default=list(INSTRUMENTS))
    mine = [s for s in sorted(STRATEGIES) if getattr(STRATEGIES[s], "MINE", False)]
    strats = c2.multiselect("Strategies", sorted(STRATEGIES), default=mine or sorted(STRATEGIES))
    c3, c4 = st.columns(2)
    presets = sorted(PROP_PRESETS)
    prop = c3.selectbox("Prop rules", presets, index=presets.index("topstep_50k"), format_func=lambda k: PROP_PRESETS[k].name)
    slip = c4.number_input("Slippage (ticks per side)", 0.0, 5.0, 1.0, 0.5)

    n = len(insts) * sum(len(param_combos(STRATEGIES[s])) for s in strats)
    st.caption(f"{n:,} backtests. Finished ones are skipped, so pressing Start again after a stop resumes.")

    if st.button("Start", type="primary", disabled=busy or not insts or not strats or bool(missing)):
        jobs.start(
            JOB,
            ["-m", "backtester.runner", "--instruments", *insts, "--strategies", *strats, "--prop", prop, "--slippage", str(slip)],
        )
        st.rerun()

with ui.card("Run every night", "02 · Schedule", ui.AMBER):
    if sys.platform != "win32":
        st.caption("Only available on Windows.")
    else:
        scheduled = subprocess.run(["schtasks", "/Query", "/TN", TASK], capture_output=True).returncode == 0
        on = st.toggle("Every night at 11 PM (all contracts, all strategies)", value=scheduled)
        if on != scheduled:
            if on:
                bat = str(jobs.ROOT / "run_scheduled.bat")
                subprocess.run(["schtasks", "/Create", "/F", "/SC", "DAILY", "/ST", "23:00", "/TN", TASK, "/TR", f'"{bat}"'], capture_output=True)
            else:
                subprocess.run(["schtasks", "/Delete", "/TN", TASK, "/F"], capture_output=True)
            st.rerun()
        st.caption("Leave the PC on and set Sleep to Never (Settings → System → Power). Useful once there are more strategies to test.")
