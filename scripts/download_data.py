"""Download ES and NQ 1-minute bars from Databento into data/.

First time: downloads everything from --start. After that, --update only fetches the days
since the last download and appends them (cheap). Always shows the price before buying
unless --yes is given (the app asks you first, then passes --yes).

Needs your Databento API key: set DATABENTO_API_KEY or paste it when asked.
"""

import argparse
import getpass
import os
import warnings
from datetime import date, timedelta

import databento as db
import pandas as pd

from backtester.data import DATA_DIR

DATASET = "GLBX.MDP3"  # CME Globex
FIRST_DAY = "2019-01-01"


def plan(symbols, start=FIRST_DAY, end=None, update=False):
    """[(symbol, start, end)] for what needs downloading."""
    end = end or (date.today() - timedelta(days=1)).isoformat()
    out = []
    for sym in symbols:
        s = start
        path = DATA_DIR / f"{sym}_1m.parquet"
        if update and path.exists() and not (DATA_DIR / "SAMPLE_DATA").exists():
            last = pd.read_parquet(path, columns=["ts"])["ts"].max()
            s = (pd.Timestamp(last).tz_convert("UTC").normalize() + pd.Timedelta(days=1)).date().isoformat()
        if s < end:
            out.append((sym, s, end))
    return out


def _query(sym, start, end):
    return dict(dataset=DATASET, symbols=[f"{sym}.v.0"], schema="ohlcv-1m", stype_in="continuous", start=start, end=end)


def price(client, todo) -> float:
    return sum(client.metadata.get_cost(**_query(*t)) for t in todo)


def download(client, todo):
    DATA_DIR.mkdir(exist_ok=True)
    # Databento flags a handful of days with patchy data; harmless for 1-minute bars.
    warnings.filterwarnings("ignore", message=".*reduced quality.*")
    for sym, start, end in todo:
        print(f"Downloading {sym} {start} to {end}...", flush=True)
        df = client.timeseries.get_range(**_query(sym, start, end)).to_df().reset_index()
        new = pd.DataFrame({"ts": df["ts_event"], **{c: df[c] for c in ("open", "high", "low", "close", "volume")}})
        path = DATA_DIR / f"{sym}_1m.parquet"
        if path.exists() and not (DATA_DIR / "SAMPLE_DATA").exists() and start != FIRST_DAY:
            new = pd.concat([pd.read_parquet(path), new]).drop_duplicates("ts").sort_values("ts")
        new.to_parquet(path)
        print(f"  {sym}: {len(new):,} bars saved", flush=True)
    (DATA_DIR / "SAMPLE_DATA").unlink(missing_ok=True)
    print("Done. Data is up to date.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["ES", "NQ"])
    ap.add_argument("--start", default=FIRST_DAY)
    ap.add_argument("--update", action="store_true", help="only fetch days after the last download")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = ap.parse_args()

    key = os.environ.get("DATABENTO_API_KEY") or getpass.getpass("Paste your Databento API key (it stays hidden), then press Enter: ").strip()
    client = db.Historical(key)
    todo = plan([s.upper() for s in args.symbols], args.start, update=args.update)
    if not todo:
        print("Done. Data is already up to date.")
        return
    cost = price(client, todo)
    print(f"Price: ${cost:,.2f}")
    if not args.yes and input("Download? Type y and press Enter: ").strip().lower() != "y":
        print("Cancelled. Nothing was charged.")
        return
    download(client, todo)


if __name__ == "__main__":
    main()
