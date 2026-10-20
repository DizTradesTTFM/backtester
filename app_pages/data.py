"""Data page: what price data is on this PC, and a button to bring it up to date."""

import databento as db
import pandas as pd
import streamlit as st

from backtester import jobs
from backtester.data import DATA_DIR
from app_pages import ui
from scripts.download_data import plan, price

JOB = "download"
SYMBOLS = ["ES", "NQ"]
KEY_FILE = DATA_DIR / ".databento_key"

ui.header("Data", "1-minute ES + NQ from Databento", [("Have", ui.GREEN), ("Update", ui.BLUE)])


@st.cache_data(ttl=30)
def summary(sym, mtime):
    df = pd.read_parquet(DATA_DIR / f"{sym}_1m.parquet", columns=["ts"])
    return len(df), df["ts"].min(), df["ts"].max()


with ui.card("What's on this PC", "01 · Have", ui.GREEN, key="have"):
    tiles = []
    for sym in SYMBOLS:
        f = DATA_DIR / f"{sym}_1m.parquet"
        if f.exists():
            n, first, last = summary(sym, f.stat().st_mtime)
            tiles.append((sym, f"{n:,}", f"1-minute bars · {first:%b %d, %Y} to {last:%b %d, %Y}", None))
        else:
            tiles.append((sym, "none", "no data yet", ui.PINK))
    ui.stats(tiles)
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


with ui.card("Update data", "02 · Buy new days", ui.BLUE, key="update"):
    progress()
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
