"""Results dashboard. Start it with dashboard.bat (or: streamlit run dashboard.py)."""

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

st.set_page_config(page_title="Backtester", layout="wide")

LINE = "#2a78d6"
SPLIT = "#8a8f98"


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
st.title("Backtester results")
if runs.empty:
    st.info("No results yet. Run run_overnight.bat first.")
    st.stop()

run = st.selectbox("Batch", runs["run"], format_func=lambda r: f"{r}  ({runs.set_index('run').loc[r, 'started']})")
run_info = runs.set_index("run").loc[run]
config = json.loads(run_info["config"])
df = load_results(run)
if df.empty:
    st.warning("This batch has no finished backtests yet.")
    st.stop()

done = len(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Backtests finished", f"{done:,} / {int(run_info['n_jobs']):,}")
c2.metric("✅ Passed every filter", int((df.verdict == "PASS").sum()))
c3.metric("👀 Watch list", int((df.verdict == "WATCH").sum()))
c4.metric("❌ Failed", int((df.verdict == "FAIL").sum()))
st.caption(
    f"Prop rules: {config['prop']['name']} · target \\${config['prop']['profit_target']:,.0f}, "
    f"max loss \\${config['prop']['max_loss']:,.0f} ({config['prop']['trailing']} trailing), "
    f"daily limit \\${config['prop']['daily_loss_limit']:,.0f} · slippage {config['eval']['slippage_ticks']} tick/side"
)

with st.expander("Why strategies failed"):
    reasons = df.loc[df.verdict == "FAIL", "reason"].str.replace(r"\s*\(.*|\s[\d.]+.*", "", regex=True)
    st.bar_chart(reasons.value_counts(), horizontal=True, color=LINE)

# ---- leaderboard -------------------------------------------------------------
st.subheader("Leaderboard")
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

cols = {
    "verdict": "Verdict",
    "instrument": "Contract",
    "strategy": "Strategy",
    "params": "Settings",
    "contracts": "Contracts",
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
table = (
    view.sort_values(sort_by, ascending=False, na_position="last")
    .reindex(columns=list(cols))
    .head(500)
    .rename(columns=cols)
    .reset_index()
)
picked = st.dataframe(
    table.drop(columns="index"),
    hide_index=True,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Prop pass % (unseen)": st.column_config.NumberColumn(format="percent"),
        "Prop blow-up % (unseen)": st.column_config.NumberColumn(format="percent"),
        "Periods profitable": st.column_config.NumberColumn(format="percent"),
        "Neighbors profitable": st.column_config.NumberColumn(format="percent"),
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
    "by pure luck; training Sharpe must beat it. **Contracts** = position size with the best prop pass rate in training. "
    "**Neighbors profitable** = share of settings one notch away that also made money (high = not a fluke). "
    "Dollar columns are for 1 contract."
)

# ---- detail ------------------------------------------------------------------
rows = picked.selection.rows if picked and picked.selection else []
if not rows:
    st.info("Click a row to see its equity curve and trades.")
    st.stop()

row = view.loc[table.loc[rows[0], "index"]]
inst = get_instrument(row.instrument)
params = json.loads(row.params)
bars = bars_for(inst.data)
sig = STRATEGIES[row.strategy].generate(bars, params)
cfg = EvalConfig(**config["eval"])
trades = run_backtest(bars, sig, inst, cfg.slippage_ticks, cfg.qty)

st.subheader(f"{row.verdict} · {row.instrument} · {row.strategy}")
st.write(STRATEGIES[row.strategy].DESCRIPTION)
st.code(json.dumps(params, indent=2), language="json")

n_days = int(bars.day[-1]) + 1
split_day = int(n_days * (1 - cfg.oos_fraction))
when = bars.ts[trades.exit_i].tz_localize(None)
equity = np.cumsum(trades.pnl)
split_ts = pd.Timestamp(bars.day_dates[split_day])

fig = go.Figure()
fig.add_trace(go.Scatter(x=when, y=equity, mode="lines", line=dict(color=LINE, width=2), name="Equity",
                         hovertemplate="%{x|%b %d %Y %H:%M}<br>$%{y:,.0f}<extra></extra>"))
fig.add_vline(x=split_ts, line=dict(color=SPLIT, width=1, dash="dot"))
fig.add_vrect(x0=split_ts, x1=when.max() if len(when) else split_ts, fillcolor=SPLIT, opacity=0.08, line_width=0)
fig.add_annotation(x=split_ts, y=1, yref="paper", text="unseen data →", showarrow=False, xanchor="left", yanchor="bottom")
fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, hovermode="x unified",
                  yaxis=dict(title="Profit $ (1 contract)", tickprefix="$", gridcolor="rgba(128,128,128,0.15)"),
                  xaxis=dict(showgrid=False))
st.plotly_chart(fig, use_container_width=True)

s1, s2, s3, s4 = st.columns(4)
size = f"{row.contracts:.0f} contracts" if pd.notna(row.get("contracts")) else "1 contract"
s1.metric(f"Prop pass % (unseen, {size})", f"{row.oos_prop_pass_rate:.0%}" if pd.notna(row.oos_prop_pass_rate) else "n/a")
s2.metric("Median days to pass", f"{row.oos_prop_days_to_pass:.0f}" if pd.notna(row.oos_prop_days_to_pass) else "n/a")
s3.metric("Win rate (training)", f"{row.is_win_rate:.0%}" if pd.notna(row.is_win_rate) else "n/a")
s4.metric("Avg trade (training)", f"${row.is_avg_trade:,.0f}" if pd.notna(row.is_avg_trade) else "n/a")

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
