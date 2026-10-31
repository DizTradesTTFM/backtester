"""Results page: the answer first (does anything work?), then strategies compared, best setups, details, raw table."""

import html
import json
import sqlite3

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester.data import load_bars
from backtester.engine import run_backtest
from backtester.evaluate import EvalConfig
from backtester.instruments import get_instrument
from backtester.runner import RESULTS_DB
from backtester.strategies import STRATEGIES
from app_pages import ui


@st.cache_data(ttl=60)
def load_runs():
    if not RESULTS_DB.exists():
        return pd.DataFrame()
    with sqlite3.connect(RESULTS_DB) as con:
        return pd.read_sql("SELECT * FROM runs ORDER BY started DESC", con)


@st.cache_data(ttl=60)
def load_results(run):
    with sqlite3.connect(RESULTS_DB) as con:
        df = pd.read_sql("SELECT * FROM results WHERE run=?", con, params=(run,))
    if df.empty:
        return df
    m = pd.json_normalize(df["metrics"].map(json.loads))
    return pd.concat([df.drop(columns=["metrics"]), m.drop(columns=["verdict", "reason"], errors="ignore")], axis=1)


@st.cache_resource
def bars_for(symbol):
    return load_bars(symbol)


VERDICT_ORDER = {"PASS": 0, "WATCH": 1, "FAIL": 2}


def problem(reason: str) -> str:
    """The verdict reason in a few plain words."""
    r = str(reason)
    if r.startswith("loses money"):
        return "Loses money"
    if r.startswith("falls apart"):
        return "Stops working on new data"
    if r.startswith("only profitable"):
        return "Too inconsistent"
    if r.startswith("too few"):
        return "Too few trades"
    if r.startswith("training Sharpe"):
        return "Could just be luck"
    if "prop pass" in r:
        return "Too weak for prop rules"
    if r.startswith("just under"):
        return "Almost beats the luck check"
    return "Passes every check" if r.startswith("survived") else r


def pct(v):
    return f"{v:.0%}" if pd.notna(v) else "–"


def money(v):
    return "–" if pd.isna(v) else (f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}")


def money_colored(v):
    return money(v) if pd.isna(v) else f'<span style="color:{ui.GREEN if v > 0 else ui.PINK}">{money(v)}</span>'


def label(r):
    return f"{r.instrument} · {r.strategy}"


runs = load_runs()
if runs.empty:
    ui.header("Results", "Nothing tested yet.")
    st.info("Go to **Run** and press Start. Results show up here when it's done.")
    st.stop()

# ---- header + filters ----------------------------------------------------------
ui.header("Results", "What worked, what didn't, and why. Numbers are on data the strategy never saw while being picked.")
c1, c2, c3 = st.columns([2, 1, 1])
run = c1.selectbox("Test batch", runs["run"], format_func=lambda r: f"{runs.set_index('run').loc[r, 'started'][:16].replace('T', ' ')}  ·  {r}")
run_info = runs.set_index("run").loc[run]
config = json.loads(run_info["config"])
all_df = load_results(run)
if all_df.empty:
    st.warning("This batch has no finished backtests yet.")
    st.stop()
strat_pick = c2.selectbox("Strategy", ["All"] + sorted(all_df.strategy.unique()))
inst_pick = c3.selectbox("Contract", ["All"] + sorted(all_df.instrument.unique()))
df = all_df[((all_df.strategy == strat_pick) | (strat_pick == "All")) & ((all_df.instrument == inst_pick) | (inst_pick == "All"))].copy()
df["problem"] = df["reason"].map(problem)
df["rank_key"] = df.verdict.map(VERDICT_ORDER)
ranked = df.sort_values(["rank_key", "oos_prop_pass_rate", "oos_net_profit"], ascending=[True, False, False], na_position="last")

n_pass, n_watch, n_fail = (int((df.verdict == v).sum()) for v in ("PASS", "WATCH", "FAIL"))
best = ranked.iloc[0]

# ---- 1. the answer -----------------------------------------------------------
with ui.card("The bottom line", f"{len(df):,} setups tested", key="answer", number="01"):
    common = df.loc[df.verdict == "FAIL", "problem"].value_counts()
    if n_pass:
        msg = (f"<b style='color:{ui.GREEN}'>{n_pass} setup{'s' if n_pass > 1 else ''} passed every check.</b> "
               f"Best: {label(best)}, which passed the prop evaluation {pct(best.oos_prop_pass_rate)} of the time on new data.")
    elif n_watch:
        msg = (f"<b style='color:{ui.AMBER}'>Nothing is ready yet, but {n_watch} setup{'s are' if n_watch > 1 else ' is'} close.</b> "
               f"Closest: {label(best)}, passing the prop evaluation {pct(best.oos_prop_pass_rate)} of the time on new data.")
    else:
        top = common.index[0] if len(common) else "–"
        share = common.iloc[0] / max(1, n_fail) if len(common) else 0
        msg = (f"<b style='color:{ui.PINK}'>Nothing works yet.</b> "
               f"<span class='m'>Most common problem: {top.lower()} ({share:.0%} of setups).</span>")
    ui._html(f'<div class="bt-answer">{msg}</div>')
    ui.stats([
        ("Ready for a prop account", n_pass, "passed every check", ui.GREEN if n_pass else None),
        ("Close", n_watch, "worth a second look", ui.AMBER if n_watch else None),
        ("Fail", f"{n_fail:,}", "see why below", None),
        ("Made money on new data", pct((df.oos_net_profit > 0).mean()), "share of all setups", None),
    ])
    if len(common):
        bars_html = "".join(
            f'<div class="bt-row"><div class="k">{ui.esc(k)}</div><div class="v">'
            f'<div style="display:flex;align-items:center;gap:10px"><div style="height:8px;border-radius:4px;background:{ui.PINK};'
            f'opacity:.75;width:{max(2, v / common.iloc[0] * 70):.0f}%"></div><span class="bt-muted">{v:,}</span></div></div></div>'
            for k, v in common.items()
        )
        ui._html('<div class="bt-muted" style="font-size:13px;margin-top:4px">Why setups failed</div>' + bars_html)

# ---- 2. strategies compared ----------------------------------------------------
with ui.card("Strategies compared", "best setup of each, on new data", key="compare", number="02"):
    body = []
    for (strat, inst), g in df.groupby(["strategy", "instrument"]):
        b = g.sort_values(["rank_key", "oos_prop_pass_rate", "oos_net_profit"], ascending=[True, False, False]).iloc[0]
        body.append((b.rank_key, -(b.oos_prop_pass_rate if pd.notna(b.oos_prop_pass_rate) else -1), [
            f'<div class="name">{ui.esc(strat)}</div><div class="set">{inst} · {len(g):,} setups</div>',
            ui.pill(b.verdict),
            pct((g.oos_net_profit > 0).mean()),
            pct(b.oos_prop_pass_rate),
            pct(b.oos_prop_fail_rate),
            money_colored(b.oos_net_profit),
            ui.esc(g.loc[g.verdict == "FAIL", "problem"].mode().iat[0] if (g.verdict == "FAIL").any() else "–"),
        ]))
    body.sort(key=lambda x: (x[0], x[1]))
    ui.table(["Strategy", "Best result", "Setups profitable", "Prop pass", "Prop blow-up", "Profit (1 contract)", "Main problem"],
             [r for _, _, r in body], num_cols=(2, 3, 4, 5))
    ui._html('<div class="bt-muted" style="font-size:13px">Prop pass / blow-up: how often the best setup passed or failed a '
             f'simulated {ui.esc(config["prop"]["name"])} evaluation on new data.</div>')

# ---- 3. best setups ------------------------------------------------------------
# settings that made no difference give identical results: show each distinct result once
top = ranked.drop_duplicates(["instrument", "strategy", "oos_net_profit", "oos_trades", "is_net_profit"]).head(5)
with ui.card("Top 5 setups", "ranked by prop pass rate on new data, identical results shown once", key="top", number="03"):
    ui.table(
        ["#", "Setup", "", "Prop pass", "Blow-up", "Profit", "Win rate", "Trades / month"],
        [[
            f'<span class="rank">{i + 1}</span>',
            f'<div class="name">{label(r)}</div><div class="set">{ui.esc(ui.settings_text(json.loads(r.params)))}</div>',
            ui.pill(r.verdict),
            pct(r.oos_prop_pass_rate),
            pct(r.oos_prop_fail_rate),
            money_colored(r.oos_net_profit),
            pct(r.oos_win_rate),
            f"{r.oos_trades_per_day * 21:.0f}" if pd.notna(r.oos_trades_per_day) else "–",
        ] for i, (_, r) in enumerate(top.iterrows())],
        num_cols=(3, 4, 5, 6, 7),
    )

# ---- 4. one setup up close -------------------------------------------------------
with ui.card("Look closer", "equity curve, prop setup and every trade", key="detail", number="04"):
    choice = st.radio("Setup", list(range(len(top))), horizontal=True, label_visibility="collapsed",
                      format_func=lambda i: f"#{i + 1}  {label(top.iloc[i])}")
    row = top.iloc[choice]
    inst = get_instrument(row.instrument)
    params = json.loads(row.params)
    bars = bars_for(inst.data)
    sig = STRATEGIES[row.strategy].generate(bars, params)
    cfg = EvalConfig(**config["eval"])
    trades = run_backtest(bars, sig, inst, cfg.slippage_ticks, cfg.qty)

    n_days = int(bars.day[-1]) + 1
    split_ts = pd.Timestamp(bars.day_dates[int(n_days * (1 - cfg.oos_fraction))])
    when = bars.ts[trades.exit_i].tz_localize(None)
    equity = np.cumsum(trades.pnl)

    setup = "–"
    if pd.notna(row.get("contracts")):
        setup = (f"{row.contracts:.0f} contract{'s' if row.contracts != 1 else ''} · "
                 + (f"stop for the day after losing ${row.daily_loss_stop:,.0f}" if row.daily_loss_stop else "no daily loss stop")
                 + (f" · stop for the day after making ${row.daily_profit_lock:,.0f}" if row.daily_profit_lock else ""))
    ui.rows([
        ("Result", f"{ui.pill(row.verdict)} &nbsp;<span class='bt-muted'>{ui.esc(problem(row.reason))}</span>"),
        ("Settings", ui.esc(ui.settings_text(params))),
        ("Best prop setup", setup + " <span class='bt-muted'>(picked on the older data)</span>"),
    ])
    ui.stats([
        ("Prop pass", pct(row.oos_prop_pass_rate), "new data", ui.GREEN),
        ("Prop blow-up", pct(row.oos_prop_fail_rate), "new data", ui.PINK),
        ("Days to pass", f"{row.oos_prop_days_to_pass:.0f}" if pd.notna(row.oos_prop_days_to_pass) else "–", "median", None),
        ("Win rate", pct(row.oos_win_rate), "new data", None),
        ("Profit", money(row.oos_net_profit), "new data, 1 contract", None),
        ("Worst drawdown", money(row.oos_max_drawdown), "new data, 1 contract", None),
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=when, y=equity, mode="lines", line=dict(color=ui.BLUE, width=2),
                             hovertemplate="%{x|%b %d %Y %H:%M}<br>$%{y:,.0f}<extra></extra>"))
    fig.add_vrect(x0=split_ts, x1=when.max() if len(when) else split_ts, fillcolor=ui.AMBER, opacity=0.06, line_width=0)
    fig.add_vline(x=split_ts, line=dict(color=ui.MUTED, width=1, dash="dot"))
    fig.add_annotation(x=split_ts, y=1, yref="paper", text="  new data (the honest test) →", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(color=ui.AMBER))
    ui.plot_layout(fig)
    fig.update_layout(showlegend=False, hovermode="x unified", yaxis_tickprefix="$", yaxis_title="Profit, 1 contract")
    st.plotly_chart(fig, width="stretch")

    with st.expander(f"All {len(trades):,} trades"):
        st.dataframe(
            pd.DataFrame({
                "Entry": bars.ts[trades.entry_i].tz_localize(None),
                "Exit": when,
                "Side": np.where(trades.direction > 0, "long", "short"),
                "Entry price": trades.entry_px,
                "Exit price": trades.exit_px,
                "Profit $": trades.pnl.round(2),
                "Worst point $": trades.mae.round(2),
            }),
            hide_index=True, width="stretch",
        )

# ---- 5. everything -------------------------------------------------------------
with st.expander(f"Every result ({len(df):,} rows, for digging)"):
    full = ranked.assign(
        Result=ranked.verdict.map(ui.VERDICT_WORDS),
        Setup=ranked.instrument + " · " + ranked.strategy,
        Settings=ranked.params.map(lambda p: ui.settings_text(json.loads(p))),
        Problem=ranked.problem,
    )[["Result", "Setup", "Settings", "oos_prop_pass_rate", "oos_prop_fail_rate", "oos_net_profit", "oos_win_rate",
       "is_net_profit", "contracts", "robustness", "Problem"]]
    st.dataframe(
        full.rename(columns={
            "oos_prop_pass_rate": "Prop pass", "oos_prop_fail_rate": "Blow-up", "oos_net_profit": "Profit (new data)",
            "oos_win_rate": "Win rate", "is_net_profit": "Profit (older data)", "contracts": "Contracts",
            "robustness": "Nearby settings profitable",
        }),
        hide_index=True, width="stretch",
        column_config={
            "Prop pass": st.column_config.NumberColumn(format="percent"),
            "Blow-up": st.column_config.NumberColumn(format="percent"),
            "Win rate": st.column_config.NumberColumn(format="percent"),
            "Nearby settings profitable": st.column_config.NumberColumn(format="percent"),
            "Profit (new data)": st.column_config.NumberColumn(format="dollar"),
            "Profit (older data)": st.column_config.NumberColumn(format="dollar"),
        },
    )
    st.caption(
        f"Older data = the first 70% of history, used to pick settings. New data = the last 30%, never used for picking. "
        f"Prop rules: {config['prop']['name']}, slippage {config['eval']['slippage_ticks']} tick per side."
    )
