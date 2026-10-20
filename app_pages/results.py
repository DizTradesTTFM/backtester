"""Results page: leaderboard, equity curves and trades for each batch."""

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


runs = load_runs()
ui.header("Results", "Test → Filter → Fund", [("✓ Pass", ui.GREEN), ("◎ Watch", ui.AMBER), ("✕ Fail", ui.PINK)])
if runs.empty:
    st.info("No results yet. Go to **Run** and press Start.")
    st.stop()

run = st.selectbox("Batch", runs["run"], format_func=lambda r: f"{r}  ({runs.set_index('run').loc[r, 'started']})")
run_info = runs.set_index("run").loc[run]
config = json.loads(run_info["config"])
df = load_results(run)
if df.empty:
    st.warning("This batch has no finished backtests yet.")
    st.stop()

done = len(df)
ui.stats([
    ("Backtests finished", f"{done:,}", f"of {int(run_info['n_jobs']):,}", None),
    ("Passed every filter", int((df.verdict == "PASS").sum()), "ready to try on a prop account", ui.GREEN),
    ("Watch list", int((df.verdict == "WATCH").sum()), "close, needs a second look", ui.AMBER),
    ("Failed", f"{int((df.verdict == 'FAIL').sum()):,}", "luck, losses or blow-ups", ui.PINK),
])
st.caption(
    f"Prop rules: {config['prop']['name']} · target \\${config['prop']['profit_target']:,.0f}, "
    f"max loss \\${config['prop']['max_loss']:,.0f} ({config['prop']['trailing']} trailing), "
    f"daily limit \\${config['prop']['daily_loss_limit']:,.0f} · slippage {config['eval']['slippage_ticks']} tick/side"
)

with st.expander("Why strategies failed"):
    reasons = df.loc[df.verdict == "FAIL", "reason"].str.replace(r"\s*\(.*|\s[\d.]+.*", "", regex=True)
    st.bar_chart(reasons.value_counts(), horizontal=True, color=ui.BLUE)

# ---- leaderboard -------------------------------------------------------------
cols = {
    "verdict": "Verdict",
    "instrument": "Contract",
    "strategy": "Strategy",
    "params": "Settings",
    "contracts": "Contracts",
    "daily_loss_stop": "Daily loss stop $",
    "daily_profit_lock": "Daily profit lock $",
    "oos_prop_pass_rate": "Prop pass % (unseen)",
    "oos_prop_fail_rate": "Prop blow-up % (unseen)",
    "oos_net_profit": "Profit $ (unseen)",
    "oos_profit_factor": "Profit factor (unseen)",
    "is_net_profit": "Profit $ (training)",
    "is_sharpe": "Sharpe (training)",
    "luck_bar": "Luck bar",
    "robustness": "Neighbors profitable",
    "stability": "Periods profitable",
    "oos_long_profit": "Longs $ (unseen)",
    "oos_short_profit": "Shorts $ (unseen)",
    "is_trades": "Trades",
    "is_max_drawdown": "Max drawdown $",
    "reason": "Why",
}
with ui.card("Leaderboard", "01 · Best first", ui.BLUE):
    f1, f2, f3 = st.columns(3)
    verdicts = f1.multiselect("Verdict", ["PASS", "WATCH", "FAIL"], default=["PASS", "WATCH"])
    insts = f2.multiselect("Contract", sorted(df.instrument.unique()), default=sorted(df.instrument.unique()))
    strats = f3.multiselect("Strategy", sorted(df.strategy.unique()), default=sorted(df.strategy.unique()))
    view = df[df.verdict.isin(verdicts) & df.instrument.isin(insts) & df.strategy.isin(strats)]
    sort_by = ["oos_prop_pass_rate", "is_sharpe"]
    if view.empty and verdicts == ["PASS", "WATCH"]:
        st.info("Nothing survived the filters in this batch. Showing the closest failures (best training Sharpe).")
        view = df[df.instrument.isin(insts) & df.strategy.isin(strats)]
        sort_by = ["is_sharpe"]

    table = (
        view.sort_values(sort_by, ascending=False, na_position="last")
        .reindex(columns=list(cols))
        .head(500)
        .rename(columns=cols)
        .reset_index()
    )
    shown = table.drop(columns="index")
    picked = st.dataframe(
        shown.style.map(lambda v: f"color: {ui.VERDICT_COLORS.get(v, ui.TEXT)}; font-weight: 600", subset=["Verdict"]),
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Prop pass % (unseen)": st.column_config.NumberColumn(format="percent"),
            "Prop blow-up % (unseen)": st.column_config.NumberColumn(format="percent"),
            "Periods profitable": st.column_config.NumberColumn(format="percent"),
            "Neighbors profitable": st.column_config.NumberColumn(format="percent"),
            "Daily loss stop $": st.column_config.NumberColumn(format="$%d", help="0 = off"),
            "Daily profit lock $": st.column_config.NumberColumn(format="$%d", help="0 = off"),
            "Longs $ (unseen)": st.column_config.NumberColumn(format="dollar"),
            "Shorts $ (unseen)": st.column_config.NumberColumn(format="dollar"),
            "Profit $ (unseen)": st.column_config.NumberColumn(format="dollar"),
            "Profit $ (training)": st.column_config.NumberColumn(format="dollar"),
            "Max drawdown $": st.column_config.NumberColumn(format="dollar"),
            "Sharpe (training)": st.column_config.NumberColumn(format="%.2f"),
            "Luck bar": st.column_config.NumberColumn(format="%.2f"),
            "Profit factor (unseen)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "**Training** = the older 70% of data the settings were picked on. **Unseen** = the newest 30%, never used for "
        "picking: that's the honest number. **Luck bar** = the Sharpe the best of this many random strategies would hit "
        "by pure luck; training Sharpe must beat it. **Contracts / Daily loss stop / Daily profit lock** = the prop setup "
        "with the best pass rate in training (stop trading for the day after losing or making that much; 0 = off). "
        "**Neighbors profitable** = share of settings one notch away that also made money (high = not a fluke). "
        "Dollar columns are for 1 contract."
    )

# ---- detail ------------------------------------------------------------------
rows = picked.selection.rows if picked and picked.selection else []
if not rows:
    st.info("Click a row in the leaderboard to see its equity curve and trades.")
    st.stop()

row = view.loc[table.loc[rows[0], "index"]]
inst = get_instrument(row.instrument)
params = json.loads(row.params)
bars = bars_for(inst.data)
sig = STRATEGIES[row.strategy].generate(bars, params)
cfg = EvalConfig(**config["eval"])
trades = run_backtest(bars, sig, inst, cfg.slippage_ticks, cfg.qty)

n_days = int(bars.day[-1]) + 1
split_day = int(n_days * (1 - cfg.oos_fraction))
when = bars.ts[trades.exit_i].tz_localize(None)
equity = np.cumsum(trades.pnl)
split_ts = pd.Timestamp(bars.day_dates[split_day])

color = ui.VERDICT_COLORS.get(row.verdict, ui.BLUE)
with ui.card(f"{row.instrument} · {row.strategy}", f"02 · {row.verdict}", color, key="detail"):
    ui.rows([
        ("What it does", STRATEGIES[row.strategy].DESCRIPTION),
        ("Settings", " · ".join(f"<span class='mono'>{k}</span> {v}" for k, v in params.items())),
        ("Verdict", f"<span style='color:{color}'>{row.verdict}</span> <span class='bt-muted'>{html.escape(str(row.reason))}</span>"),
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=when, y=equity, mode="lines", line=dict(color=ui.BLUE, width=2), name="Equity",
                             hovertemplate="%{x|%b %d %Y %H:%M}<br>$%{y:,.0f}<extra></extra>"))
    fig.add_vline(x=split_ts, line=dict(color=ui.MUTED, width=1, dash="dot"))
    fig.add_vrect(x0=split_ts, x1=when.max() if len(when) else split_ts, fillcolor=ui.AMBER, opacity=0.06, line_width=0)
    fig.add_annotation(x=split_ts, y=1, yref="paper", text="unseen data →", showarrow=False, xanchor="left",
                       yanchor="bottom", font=dict(color=ui.AMBER, family="JetBrains Mono"))
    ui.plot_layout(fig)
    fig.update_layout(showlegend=False, hovermode="x unified", yaxis_title="Profit $ (1 contract)", yaxis_tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

    size = f"{row.contracts:.0f} contracts" if pd.notna(row.get("contracts")) else "1 contract"
    if pd.notna(row.get("daily_loss_stop")):
        setup = f"{row.contracts:.0f} contracts · " + (
            f"stop for the day after losing ${row.daily_loss_stop:,.0f}" if row.daily_loss_stop else "no daily loss stop"
        ) + (f" · lock the day after making ${row.daily_profit_lock:,.0f}" if row.daily_profit_lock else " · no profit lock")
        ui.rows([("Prop setup", f"{setup} <span class='bt-muted'>(picked on training data)</span>")])

    pct = lambda v: f"{v:.0%}" if pd.notna(v) else "n/a"
    ui.stats([
        (f"Prop pass · unseen", pct(row.oos_prop_pass_rate), size, ui.GREEN),
        ("Prop blow-up · unseen", pct(row.get("oos_prop_fail_rate")), size, ui.PINK),
        ("Median days to pass", f"{row.oos_prop_days_to_pass:.0f}" if pd.notna(row.oos_prop_days_to_pass) else "n/a", "unseen", None),
        ("Win rate", pct(row.is_win_rate), "training", None),
        ("Avg trade", f"${row.is_avg_trade:,.0f}" if pd.notna(row.is_avg_trade) else "n/a", "training, 1 contract", None),
    ])

    with st.expander(f"All {len(trades):,} trades"):
        st.dataframe(
            pd.DataFrame({
                "entry time": bars.ts[trades.entry_i].tz_localize(None),
                "exit time": when,
                "side": np.where(trades.direction > 0, "long", "short"),
                "entry": trades.entry_px,
                "exit": trades.exit_px,
                "profit $": trades.pnl.round(2),
                "worst point $": trades.mae.round(2),
            }),
            hide_index=True,
            use_container_width=True,
        )
