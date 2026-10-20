@echo off
cd /d "%~dp0"
echo Setting up the backtester. This takes a few minutes the first time.
where python >nul 2>nul || (echo Python is not installed. Install Python 3.13 from python.org and tick "Add Python to PATH". & pause & exit /b 1)
if not exist .venv python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || (echo Install failed. Send a screenshot of this window to Claude. & pause & exit /b 1)
.venv\Scripts\python -m pytest -q || (echo Self-test failed. Send a screenshot of this window to Claude. & pause & exit /b 1)
echo.
echo All set. Next: double-click make_sample_data.bat to try it, or download_data.bat for real data.
pause
