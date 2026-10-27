@echo off
echo ========================================================
echo   Launching ASL Interpreter (Handspeak AI)
echo ========================================================
echo 1. Launching Backend on http://127.0.0.1:8000 ...
start "ASL Backend" cmd /k "cd /d "%~dp0backend" && if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo 2. Launching Frontend on http://127.0.0.1:5173 ...
start "ASL Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo Waiting 15 seconds for both servers to start...
timeout /t 15 /nobreak >nul
start "" http://127.0.0.1:5173
echo Opened http://127.0.0.1:5173 in your browser.
echo If it shows an error, wait until the "ASL Backend" window says "Application startup complete", then refresh.
echo Keep the two server windows open; close them to stop the app.
echo.
pause
