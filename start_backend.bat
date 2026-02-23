@echo off
echo Starting Signal Viewer Backend...
cd /d "%~dp0backend"
call venv\Scripts\activate.bat
python run.py
