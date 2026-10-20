@echo off
cd /d "%~dp0"
echo This makes Windows run the backtests every night at 11:00 PM.
if not exist results mkdir results
schtasks /Create /F /SC DAILY /ST 23:00 /TN "Backtester nightly" /TR "\"%~dp0run_scheduled.bat\""
echo.
echo Done. Make sure your PC is ON and set to never sleep (Settings, System, Power).
echo Progress is written to results\nightly_log.txt
echo To remove it later: schtasks /Delete /TN "Backtester nightly" /F
pause
