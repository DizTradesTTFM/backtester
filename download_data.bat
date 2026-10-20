@echo off
cd /d "%~dp0"
.venv\Scripts\python -m scripts.download_data
pause
