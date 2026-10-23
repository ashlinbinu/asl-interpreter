@echo off
echo ========================================================
echo   Launching ASL Interpreter (Handspeak AI)
echo ========================================================
echo 1. Launching Backend on http://localhost:8000 ...
start "ASL Backend" cmd /k "cd /d "%~dp0backend" && if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

echo 2. Launching Frontend on http://localhost:5173 ...
start "ASL Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo Both services have been launched in separate windows!
echo Open your browser at: http://localhost:5173
echo.
pause
