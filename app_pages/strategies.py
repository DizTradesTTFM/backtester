"""Strategies page: every strategy in the app, in plain English, with how many settings it tries."""

import streamlit as st

from backtester.strategies import STRATEGIES, param_combos

st.title("Strategies")
st.write(
    "Every strategy below is tested on every contract you pick on the **Run** page. "
    "To add your own, describe it to Claude in plain words: when you enter, where the stop goes, "
    "where you take profit, and what hours you trade. It shows up here the next time you open the app."
)

for name in sorted(STRATEGIES):
    mod = STRATEGIES[name]
    with st.expander(f"**{name}** · {len(param_combos(mod)):,} setting combos"):
        st.write(mod.DESCRIPTION)
        st.write("Settings it tries:")
        for k, v in mod.PARAM_GRID.items():
            st.write(f"- `{k}`: {', '.join(str(x) for x in v)}")
