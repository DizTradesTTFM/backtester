# Backtester

Tests thousands of intraday futures strategies (ES, NQ, MES, MNQ) on 1-minute data, throws out the ones that only worked by luck, and ranks the survivors by how likely they are to **pass and keep a prop firm account**.

You never edit code. You double-click `.bat` files.

## One-time setup (Windows)

1. **Install Python 3.12 or 3.13** (not newer: the speed library lags new Python releases) from https://www.python.org/downloads/windows/ . On the first install screen, tick **"Add python.exe to PATH"**, then click Install Now.
2. **Install GitHub Desktop** from https://desktop.github.com and sign in as your GitHub account.
3. In GitHub Desktop: **File → Clone repository → backtester → Clone**. Note the folder it saves to (usually `Documents\GitHub\backtester`).
4. Open that folder and double-click **`install.bat`**. Wait until it says **All set**.

## Every time

| Double-click | What it does |
|---|---|
| `make_sample_data.bat` | Writes FAKE data so you can try everything. Results on it mean nothing. |
| `download_data.bat` | Buys and downloads real ES/NQ data from Databento. Shows the price first and asks before charging. |
| `run_overnight.bat` | Runs every backtest using all but one CPU core. Close the window to stop; double-click again to resume where it left off. |
| `dashboard.bat` | Opens the results in your browser. |
| `schedule_nightly.bat` | Makes Windows start the run every night at 11 PM automatically. |

**Getting updates:** open GitHub Desktop and click **Fetch origin**, then **Pull**. Your data and results are kept. If an update adds new packages, run `install.bat` again.

**Leave the PC on overnight:** Settings → System → Power → set "Sleep" to **Never** when plugged in.

## How a strategy is judged

Every combo runs on the full history, split into:

- **Training (older 70%)**: used to rank.
- **Unseen (newest 30%)**: never used for picking. This is the honest test.

A strategy gets **PASS** only if all of these hold:

1. At least 100 training trades.
2. Profitable in training, with a Sharpe above the **luck bar** (the Sharpe the best of N random strategies would hit by pure chance; N = how many combos were tested).
3. Profitable in at least 4 of 6 training periods (not one lucky streak).
4. Still profitable on unseen data (profit factor 1.1 or better).
5. Passes the simulated prop evaluation at least 30% of the time on unseen data. If it clears 1-4 but not this, it gets **WATCH**.

The engine is pessimistic on purpose: 1 tick of slippage each way, commissions charged, stops assumed to hit before targets when both are touched in the same bar, targets need price to trade one tick through, everything flat by 3:55 PM New York time.

**Prop rules** default to an approximation of a Topstep 50K evaluation ($3,000 target, $2,000 end-of-day trailing max loss, $1,000 daily loss limit). Verify against the firm's current rules before trusting pass rates. They live in `backtester/prop.py`.

## Folder map

- `backtester/engine.py`: trade simulator
- `backtester/prop.py`: prop evaluation simulator and rule presets
- `backtester/evaluate.py`: training/unseen split, stability, luck bar, verdicts
- `backtester/strategies/`: strategy templates and the settings each one tries
- `backtester/runner.py`: overnight batch runner (results go to `results/results.db`)
- `dashboard.py`: results dashboard
- `scripts/`: data download and fake sample data
