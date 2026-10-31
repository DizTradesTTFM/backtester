"""Run page: tick the strategies and prop firm accounts to test, press Start."""

import re
import subprocess
import sys

import streamlit as st

from backtester import jobs
from backtester.data import DATA_DIR
from backtester.instruments import INSTRUMENTS
from backtester.prop import BOTS_ALLOWED, DEFAULT_FIRMS, PROP_PRESETS
from backtester.strategies import MINE, STRATEGIES, param_combos, tag, title
from app_pages import ui

JOB = "backtest"
TASK = "Backtester nightly"
MARKETS = {"ES": "S&P 500 (MES)", "NQ": "Nasdaq (MNQ)"}

ui.header("Run", "Tick what to test. Press Start.")

missing = sorted({i.data for i in INSTRUMENTS.values() if not (DATA_DIR / f"{i.data}_1m.parquet").exists()})
if missing:
    st.warning(f"No price data for {', '.join(missing)} yet. Go to **Data** first.")


def set_all(prefix, keys, value):
    for k in keys:
        st.session_state[f"{prefix}{k}"] = value


@st.fragment(run_every=2)
def status():
    log = jobs.log_tail(JOB, 200)
    prog = re.findall(r"(\d+)/(\d+) done, ([\d.]+)/sec, ~(\d+) min left", log)
    total = re.findall(r"(\d+) backtests total", log)
    if jobs.running(JOB):
        if prog:
            done, todo, rate, left = prog[-1]
            text = ("Scoring results..." if done == todo
                    else f"{int(done):,} / {int(todo):,} backtests · about {left} min left")
            st.progress(int(done) / max(1, int(todo)), text=text)
        else:
            st.progress(0.0, text="Starting up...")
        if st.button("Stop", type="secondary"):
            jobs.stop(JOB)
            st.rerun()
    elif log:
        if "Done." in log:
            st.success("Last run finished. See **Results**.")
        elif "Traceback" in log or "Error" in log:
            st.error("The last run crashed. Tell Claude.")
            with st.expander("Error details"):
                st.code(jobs.log_tail(JOB, 25))
        elif total:
            st.info("Last run was stopped. Press Start to resume.")
    now = jobs.running(JOB)
    if st.session_state.get("was_running", now) != now:
        st.session_state["was_running"] = now
        st.rerun(scope="app")
    st.session_state["was_running"] = now
    return now


busy = status()

# ---- 01 strategies -----------------------------------------------------------
strat_keys = sorted(MINE, key=lambda n: (tag(n).split()[0], int(tag(n).split("#")[1]) if "#" in tag(n) else 0))
for n in strat_keys:
    st.session_state.setdefault(f"s_{n}", True)
with ui.card("Strategies", f"{len(strat_keys)} available", key="pick_strats", number="01"):
    b1, b2, _ = st.columns([1, 1, 6])
    b1.button("All", key="s_all", on_click=set_all, args=("s_", strat_keys, True), width="stretch")
    b2.button("None", key="s_none", on_click=set_all, args=("s_", strat_keys, False), width="stretch")
    cols = st.columns(3)
    for i, n in enumerate(strat_keys):
        with cols[i % 3].container(key=f"tile_s_{n}"):
            st.checkbox(title(n), key=f"s_{n}", help=STRATEGIES[n].DESCRIPTION)
            ui._html(f'<div class="bt-tile-sub">{tag(n)} · {len(param_combos(STRATEGIES[n]))} setups</div>')
strats = [n for n in strat_keys if st.session_state[f"s_{n}"]]

# ---- 02 prop firms -----------------------------------------------------------
firm_keys = list(PROP_PRESETS)
for k in firm_keys:
    st.session_state.setdefault(f"f_{k}", k in DEFAULT_FIRMS)
with ui.card("Prop firms", "every strategy is tested at each account you tick", key="pick_firms", number="02"):
    b1, b2, _ = st.columns([1, 1, 6])
    b1.button("All", key="f_all", on_click=set_all, args=("f_", firm_keys, True), width="stretch")
    b2.button("None", key="f_none", on_click=set_all, args=("f_", firm_keys, False), width="stretch")
    for b, (bname, _) in ui.BRANDS.items():
        keys = [k for k in firm_keys if ui.brand(k) == b]
        if not keys:
            continue
        bots = keys[0] in BOTS_ALLOWED
        pill = (f'<span class="bt-pill" style="color:{ui.GREEN};background:{ui.GREEN}1f">bots OK</span>' if bots
                else f'<span class="bt-pill" style="color:{ui.PINK};background:{ui.PINK}1f">no bots</span>')
        with st.container(key=f"tile_f_{b}"):
            ui._html(f'<div class="bt-brand">{ui.logo(keys[0], 24)}{bname} {pill}</div>')
            cols = st.columns(4)
            for i, k in enumerate(keys):
                r = PROP_PRESETS[k]
                stage = "Funded" if r.stage == "funded" else "Eval"
                cols[i % 4].checkbox(f"{ui.account_label(r)} · {stage}", key=f"f_{k}",
                                     help=f"Target ${r.profit_target:,.0f} · drawdown ${r.max_loss:,.0f} {r.trailing.upper()} · "
                                          f"DLL ${r.daily_loss_limit:,.0f} · ${r.risk:,.0f} risk per trade")
firms = [k for k in firm_keys if st.session_state[f"f_{k}"]]

# ---- 03 start ----------------------------------------------------------------
with ui.card("Start", "", key="pick_start", number="03"):
    c1, c2 = st.columns([3, 1])
    insts = c1.multiselect("Markets", list(MARKETS), default=list(MARKETS), format_func=MARKETS.get)
    slip = c2.number_input("Slippage (ticks)", 0.0, 5.0, 1.0, 0.5)
    n = len(insts) * sum(len(param_combos(STRATEGIES[s])) for s in strats)
    ui.stats([
        ("Strategies", len(strats), "", ui.BLUE),
        ("Accounts", len(firms), "", ui.BLUE),
        ("Backtests", f"{n:,}", "", None),
    ])
    if st.button("Start", type="primary", width="stretch",
                 disabled=busy or not insts or not strats or not firms or bool(missing)):
        jobs.start(JOB, ["-m", "backtester.runner", "--instruments", *insts, "--strategies", *strats,
                         "--firms", *firms, "--slippage", str(slip)])
        st.rerun()

if sys.platform == "win32":
    with st.expander("Run every night"):
        scheduled = subprocess.run(["schtasks", "/Query", "/TN", TASK], capture_output=True).returncode == 0
        on = st.toggle("Every night at 11 PM (all your strategies, default firms)", value=scheduled)
        if on != scheduled:
            if on:
                bat = str(jobs.ROOT / "run_scheduled.bat")
                subprocess.run(["schtasks", "/Create", "/F", "/SC", "DAILY", "/ST", "23:00", "/TN", TASK, "/TR", f'"{bat}"'],
                               capture_output=True)
            else:
                subprocess.run(["schtasks", "/Delete", "/TN", TASK, "/F"], capture_output=True)
            st.rerun()
        st.caption("Leave the PC on with Sleep set to Never.")
