@echo off
cd /d "%~dp0"
.venv\Scripts\python -m scripts.make_sample_data
pause
