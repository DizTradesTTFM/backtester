@echo off
cd /d "%~dp0"
if not exist .venv (echo Run install.bat first. & pause & exit /b 1)
start "Backtester - keep open, minimize me" /min .venv\Scripts\python -m scripts.launch
