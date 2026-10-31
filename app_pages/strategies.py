"""Strategies page: every strategy in the app, in plain English, with the settings it tries."""

import streamlit as st

from app_pages import ui
from backtester.strategies import STRATEGIES, param_combos

ui.header("Strategies", "To add one, describe it to Claude in plain words: entry, stop, target and hours. "
          "It shows up here the next time you open the app.")

mine = [n for n in sorted(STRATEGIES) if getattr(STRATEGIES[n], "MINE", False)]
others = [n for n in sorted(STRATEGIES) if n not in mine]


def show(name):
    mod = STRATEGIES[name]
    ui.rows([("What it does", ui.esc(mod.DESCRIPTION))]
            + [(k.replace("_", " "), " · ".join(ui.esc(ui._word(k, x)) for x in v)) for k, v in mod.PARAM_GRID.items()])


for name in mine:
    with ui.card(name, f"yours · tests {len(param_combos(STRATEGIES[name])):,} setting combos", key=f"strat_{name}"):
        show(name)

with st.expander(f"Built-in example strategies ({len(others)})"):
    for name in others:
        ui.section(name, f"{len(param_combos(STRATEGIES[name])):,} combos")
        show(name)
