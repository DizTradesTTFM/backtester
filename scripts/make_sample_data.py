"""Write fake ES/NQ 1-minute data so you can try the whole system before buying real data.

Results on this data mean NOTHING: it is a random walk, so every 'winner' is luck.
"""

from backtester.data import DATA_DIR, make_sample_bars

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for sym, seed, price in [("ES", 1, 5000.0), ("NQ", 2, 18000.0)]:
        df = make_sample_bars(days=500, seed=seed, start_price=price)
        if sym == "NQ":
            for col in ("open", "high", "low", "close"):
                df[col] = (df[col] - price) * 4 + price
        path = DATA_DIR / f"{sym}_1m.parquet"
        df.to_parquet(path)
        print(f"Wrote {len(df):,} fake bars to {path}")
