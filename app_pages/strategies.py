"""Strategies page: every strategy in the app, in plain English, with how many settings it tries."""

import streamlit as st

from app_pages import ui
from backtester.strategies import STRATEGIES, param_combos

ui.header("Strategies", "Describe it → Claude codes it → it shows up here", [("Yours", ui.BLUE), ("Built-in", ui.MUTED)])
st.caption(
    "To add your own, describe it to Claude in plain words: when you enter, where the stop goes, "
    "where you take profit, and what hours you trade. It shows up here the next time you open the app."
)

mine = [n for n in sorted(STRATEGIES) if getattr(STRATEGIES[n], "MINE", False)]
others = [n for n in sorted(STRATEGIES) if n not in mine]
for i, name in enumerate(mine + others):
    mod = STRATEGIES[name]
    yours = name in mine
    with ui.card(name, ("Yours" if yours else "Built-in") + f" · {len(param_combos(mod)):,} combos",
                 ui.BLUE if yours else ui.MUTED, key=f"strat_{name}"):
        ui.rows([("What it does", mod.DESCRIPTION)]
                + [(k, " · ".join(str(x) for x in v)) for k, v in mod.PARAM_GRID.items()])
