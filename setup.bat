@echo off
echo ==========================================
echo    Signal Viewer - Setup
echo ==========================================
echo.

echo [1/4] Creating virtual environment...
cd /d "%~dp0backend"
python -m venv venv

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing backend dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [4/4] Installing frontend dependencies...
cd /d "%~dp0frontend"
call npm install

echo.
echo ==========================================
echo    Setup Complete!
echo ==========================================
echo.
echo To run the project:
echo   - Run start_all.bat to start both services
echo   - Or run start_backend.bat and start_frontend.bat separately
echo.
echo Default URLs:
echo   - Frontend: http://localhost:3000
echo   - Backend:  http://localhost:5000
echo.
pause
