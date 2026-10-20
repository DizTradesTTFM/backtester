"""Download ES and NQ 1-minute bars from Databento and save them to data/.

Shows the exact price BEFORE downloading and asks you to confirm.
Uses the front-month continuous contract (rolls to the next contract by volume).

Needs your Databento API key: paste it when asked, or set DATABENTO_API_KEY.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["ES", "NQ"])
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = ap.parse_args()

    key = os.environ.get("DATABENTO_API_KEY") or getpass.getpass("Paste your Databento API key (it stays hidden), then press Enter: ").strip()
    client = db.Historical(key)
    symbols = [f"{s.upper()}.v.0" for s in args.symbols]
    query = dict(dataset=DATASET, symbols=symbols, schema="ohlcv-1m", stype_in="continuous", start=args.start, end=args.end)

    cost = client.metadata.get_cost(**query)
    print(f"\n{', '.join(args.symbols)} 1-minute bars, {args.start} to {args.end}: ${cost:,.2f}")
    print("(New accounts usually get free credit that covers this. Check your Databento balance.)")
    if not args.yes and input("Download? Type y and press Enter: ").strip().lower() != "y":
        print("Cancelled. Nothing was charged.")
        return

    DATA_DIR.mkdir(exist_ok=True)
    # Databento flags a handful of days with patchy data; harmless for 1-minute bars, so keep the window clean.
    warnings.filterwarnings("ignore", message=".*reduced quality.*")
    for sym, cont in zip(args.symbols, symbols):
        print(f"Downloading {sym}...")
        store = client.timeseries.get_range(**{**query, "symbols": [cont]})
        df = store.to_df().reset_index()
        out = pd.DataFrame(
            {
                "ts": df["ts_event"],
                "open": df["open"],
                "high": df["high"],
                "low": df["low"],
                "close": df["close"],
                "volume": df["volume"],
            }
        )
        path = DATA_DIR / f"{sym.upper()}_1m.parquet"
        out.to_parquet(path)
        print(f"  saved {len(out):,} bars to {path}")
    (DATA_DIR / "SAMPLE_DATA").unlink(missing_ok=True)
    print("\nDone. Real data is in place; fake sample data (if any) was replaced.")


if __name__ == "__main__":
    main()
