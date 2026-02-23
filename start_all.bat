@echo off
echo ==========================================
echo    Signal Viewer - Starting All Services
echo ==========================================
echo.

echo [1/2] Starting Backend (Port 5000)...
start "Backend" cmd /k "cd /d "%~dp0backend" && call venv\Scripts\activate.bat && python run.py"

timeout /t 5 /nobreak > nul

echo [2/2] Starting Frontend (Port 3000)...
start "Frontend" cmd /k "cd /d "%~dp0frontend" && npm start"

echo.
echo ==========================================
echo    Services Started!
echo ==========================================
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit this window...
pause > nul
