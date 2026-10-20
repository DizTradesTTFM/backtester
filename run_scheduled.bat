@echo off
cd /d "%~dp0"
.venv\Scripts\python -m backtester.runner >> results\nightly_log.txt 2>&1
