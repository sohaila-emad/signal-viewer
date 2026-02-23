@echo off
echo Starting Signal Viewer Frontend...
cd /d "%~dp0frontend"
call npm install
call npm start
