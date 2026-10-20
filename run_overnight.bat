@echo off
cd /d "%~dp0"
echo Running every backtest. Leave this window open. Closing it stops the run; double-click again to resume.
.venv\Scripts\python -m backtester.runner %*
echo.
echo Finished. Double-click dashboard.bat to see results.
pause
