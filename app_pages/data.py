"""Data page: what price data is on this PC, and a button to bring it up to date."""

import databento as db
import pandas as pd
import streamlit as st

from backtester import jobs
from backtester.data import DATA_DIR
from scripts.download_data import plan, price

JOB = "download"
SYMBOLS = ["ES", "NQ"]
KEY_FILE = DATA_DIR / ".databento_key"

st.title("Price data")


@st.cache_data(ttl=30)
def summary(sym, mtime):
    df = pd.read_parquet(DATA_DIR / f"{sym}_1m.parquet", columns=["ts"])
    return len(df), df["ts"].min(), df["ts"].max()


rows = []
for sym in SYMBOLS:
    f = DATA_DIR / f"{sym}_1m.parquet"
    if f.exists():
        n, first, last = summary(sym, f.stat().st_mtime)
        rows.append({"Contract": sym, "1-minute bars": f"{n:,}", "From": f"{first:%b %d, %Y}", "To": f"{last:%b %d, %Y}"})
    else:
        rows.append({"Contract": sym, "1-minute bars": "none", "From": "", "To": ""})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=False)
if (DATA_DIR / "SAMPLE_DATA").exists():
    st.warning("This is FAKE sample data. Results on it mean nothing. Download real data below.")


@st.fragment(run_every=2)
def progress():
    if jobs.running(JOB):
        st.info("Downloading... you can use the rest of the app meanwhile.")
        st.code(jobs.log_tail(JOB, 5))
    else:
        log = jobs.log_tail(JOB, 8)
        if "Done." in log:
            st.success(log.strip().splitlines()[-1])
        elif log:
            st.error("The last download failed:")
            st.code(log)


progress()

st.subheader("Update data")
saved = KEY_FILE.read_text().strip() if KEY_FILE.exists() else ""
key = st.text_input("Databento API key", value=saved, type="password", help="Saved on this PC only, in data/.databento_key")
if st.button("Check price", disabled=not key or jobs.running(JOB)):
    try:
        todo = plan(SYMBOLS, update=True)
        cost = price(db.Historical(key), todo) if todo else 0.0
        st.session_state["dl_quote"] = (todo, cost)
        DATA_DIR.mkdir(exist_ok=True)
        KEY_FILE.write_text(key)
    except Exception as e:
        st.error(f"Databento said: {e}")

quote = st.session_state.get("dl_quote")
if quote:
    todo, cost = quote
    if not todo:
        st.success("Already up to date.")
    else:
        span = ", ".join(f"{s} {a} to {b}" for s, a, b in todo)
        st.write(f"**{span}** costs **\\${cost:,.2f}** on Databento.")
        if st.button(f"Buy and download (\\${cost:,.2f})", type="primary"):
            jobs.start(JOB, ["-m", "scripts.download_data", "--update", "--yes"], env={"DATABENTO_API_KEY": key})
            st.session_state.pop("dl_quote")
            st.rerun()
st.caption("After new data arrives, the next run on the Run page starts a fresh batch automatically.")
