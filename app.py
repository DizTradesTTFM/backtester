"""The Backtester app. Open it with the desktop icon (or Backtester.bat)."""

import streamlit as st

st.set_page_config(page_title="Backtester", layout="wide")

st.navigation(
    [
        st.Page("app_pages/results.py", title="Results", icon="📊", default=True),
        st.Page("app_pages/run.py", title="Run", icon="▶️"),
        st.Page("app_pages/strategies.py", title="Strategies", icon="🧠"),
        st.Page("app_pages/data.py", title="Data", icon="💾"),
    ]
).run()
