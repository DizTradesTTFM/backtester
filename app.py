"""The Backtester app. Open it with the desktop icon (or Backtester.bat)."""

import streamlit as st

from app_pages.ui import style

st.set_page_config(page_title="Backtester", page_icon="📈", layout="wide")
style()

st.navigation(
    [
        st.Page("app_pages/results.py", title="Results", default=True),
        st.Page("app_pages/run.py", title="Run"),
        st.Page("app_pages/strategies.py", title="Strategies"),
        st.Page("app_pages/data.py", title="Data"),
    ],
    position="top",
).run()
