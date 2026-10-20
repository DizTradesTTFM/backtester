# Backtester

An app that tests thousands of intraday futures strategies (ES, NQ, MES, MNQ) on 1-minute data, throws out the ones that only worked by luck, and ranks the survivors by how likely they are to **pass and keep a prop firm account**.

## Setup (once, Windows)

1. **Install Python 3.12 or 3.13** (not newer) from https://www.python.org/downloads/windows/ . On the first screen tick **"Add python.exe to PATH"**.
2. Get this folder onto your PC (GitHub Desktop clone, or Code → Download ZIP and unzip). Don't put it inside OneDrive.
3. Double-click **`install.bat`**. It sets everything up, puts a **Backtester** icon on your desktop and opens the app.

## Using it

Double-click the **Backtester** icon. It updates itself, then opens in your browser. A small minimized window keeps it running; close that window to close the app.

| Page | What it's for |
|---|---|
| **Results** | Leaderboard, equity curves and trades for every batch |
| **Run** | Pick contracts and strategies, press Start, watch the progress bar. Optional nightly schedule |
| **Strategies** | Every strategy in plain English and the settings it tries |
| **Data** | What price data you have, and a button to buy only the new days from Databento |

To add a strategy, describe it to Claude in words (entry, stop, target, hours). It appears in the app the next time you open it.

## How a strategy is judged

Data is 1-minute bars; most strategies read signals off 5, 15 or 30-minute charts built from it. Fills and stops are still checked minute by minute.

Every combo runs on the full history, split into **training** (older 70%, used to rank) and **unseen** (newest 30%, never used for picking: the honest test). **PASS** needs all of:

1. At least 100 training trades, profitable in at least 4 of 6 training periods.
2. Still profitable on unseen data (profit factor 1.1 or better).
3. Training Sharpe above the **luck bar** (what the best of N random strategies would hit by chance). Within 15% of it counts as **WATCH** if 60%+ of neighboring settings also made money.
4. Passes the simulated prop evaluation 30%+ of the time on unseen data, with the setup that did best in training: contract size (1-10, within the firm's limit), daily loss stop, daily profit lock. Clearing 1-3 but not this gets **WATCH**.

The engine is pessimistic: 1 tick slippage each way, commissions, stop assumed hit before target in the same bar, targets must trade one tick through, flat by 3:55 PM New York.

**Prop rules** default to an approximation of a Topstep 50K evaluation. Verify against the firm's current rules. They live in `backtester/prop.py`.
