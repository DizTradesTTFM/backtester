@echo off
cd /d "%~dp0"
echo Setting up the Backtester. This takes a few minutes the first time.
where python >nul 2>nul || (echo Python is not installed. Install Python 3.13 from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
if not exist .venv python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip -q
.venv\Scripts\python -m pip install -r requirements.txt -q || (echo Install failed. Send a screenshot of this window to Claude. & pause & exit /b 1)
.venv\Scripts\python -m pytest -q -p no:warnings || (echo Self-test failed. Send a screenshot of this window to Claude. & pause & exit /b 1)
.venv\Scripts\python -m scripts.update
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Backtester.lnk'); $s.TargetPath='%~dp0Backtester.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.Save()"
echo.
echo All set. A Backtester icon is on your desktop. Opening the app now...
call Backtester.bat
timeout /t 5 >nul
