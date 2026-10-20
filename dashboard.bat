@echo off
cd /d "%~dp0"
echo Opening the dashboard in your browser. Keep this window open while you use it.
.venv\Scripts\python -m streamlit run dashboard.py --server.headless false --browser.gatherUsageStats false
