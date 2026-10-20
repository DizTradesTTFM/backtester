"""Loading 1-minute bars and preparing the arrays the engine needs.

Bar files are Parquet with columns: ts (UTC timestamp), open, high, low, close, volume.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXCHANGE_TZ = "America/New_York"


@dataclass
class Bars:
    symbol: str
    ts: pd.DatetimeIndex  # exchange-local (New York) timestamps, bar open time
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    minute: np.ndarray  # minutes since midnight, New York time
    day: np.ndarray  # trading-day index (0, 1, 2...). A session starting 18:00 counts as the next day.
    day_dates: np.ndarray  # calendar date of each trading-day index

    def __len__(self) -> int:
        return len(self.close)

    def slice(self, start: int, stop: int) -> "Bars":
        """Bars[start:stop], with day indexes renumbered from 0."""
        day = self.day[start:stop]
        first = day[0] if len(day) else 0
        return Bars(
            self.symbol,
            self.ts[start:stop],
            self.open[start:stop],
            self.high[start:stop],
            self.low[start:stop],
            self.close[start:stop],
            self.volume[start:stop],
            self.minute[start:stop],
            day - first,
            self.day_dates[first : (day[-1] + 1 if len(day) else first)],
        )


def bars_from_frame(df: pd.DataFrame, symbol: str) -> Bars:
    df = df.sort_values("ts").drop_duplicates("ts")
    ts = pd.DatetimeIndex(df["ts"])
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert(EXCHANGE_TZ)

    minute = (ts.hour * 60 + ts.minute).to_numpy().astype(np.int32)
    # CME equity futures sessions open at 18:00 New York time and belong to the next trading day.
    trade_date = (ts + pd.Timedelta(hours=6)).normalize().tz_localize(None)
    day_dates, day = np.unique(trade_date.to_numpy(), return_inverse=True)

    return Bars(
        symbol=symbol,
        ts=ts,
        open=df["open"].to_numpy(np.float64),
        high=df["high"].to_numpy(np.float64),
        low=df["low"].to_numpy(np.float64),
        close=df["close"].to_numpy(np.float64),
        volume=df["volume"].to_numpy(np.float64),
        minute=minute,
        day=day.astype(np.int64),
        day_dates=day_dates,
    )


def load_bars(symbol: str, path: Path | None = None) -> Bars:
    path = Path(path) if path else DATA_DIR / f"{symbol.upper()}_1m.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No data file at {path}. Run download_data.bat (real data) or make_sample_data.bat (fake data for testing)."
        )
    return bars_from_frame(pd.read_parquet(path), symbol.upper())


def make_sample_bars(days: int = 250, seed: int = 0, start_price: float = 5000.0, tick: float = 0.25) -> pd.DataFrame:
    """Random-walk 1-minute bars for testing. Not real market data; any 'edge' found on it is fake."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2023-01-03", periods=days)
    rows = []
    price = start_price
    for d in sessions:
        # 09:30-16:00 New York regular session only, to keep the sample small.
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=390, freq="1min", tz=EXCHANGE_TZ)
        vol = rng.uniform(0.6, 1.6)
        steps = rng.standard_t(4, size=390) * 1.2 * vol
        closes = price + np.cumsum(steps)
        opens = np.concatenate([[price], closes[:-1]])
        wick = np.abs(rng.normal(0, 0.8 * vol, size=(2, 390)))
        highs = np.maximum(opens, closes) + wick[0]
        lows = np.minimum(opens, closes) - wick[1]
        rnd = lambda a: np.round(a / tick) * tick
        rows.append(
            pd.DataFrame(
                {
                    "ts": idx.tz_convert("UTC"),
                    "open": rnd(opens),
                    "high": rnd(highs),
                    "low": rnd(lows),
                    "close": rnd(closes),
                    "volume": rng.integers(200, 5000, size=390),
                }
            )
        )
        price = closes[-1] + rng.normal(0, 5)
    return pd.concat(rows, ignore_index=True)
