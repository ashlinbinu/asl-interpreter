@echo off
echo ========================================================
echo   Launching ASL Interpreter (Sign-ify)
echo ========================================================
echo 1. Launching Backend on http://127.0.0.1:8000 ...
start "ASL Backend" cmd /k "cd /d "%~dp0backend" && if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log"

echo 2. Launching Frontend on http://127.0.0.1:5173 ...
start "ASL Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo Waiting for the backend to answer...
rem Full paths: other tools on PATH (Git Bash, MSYS2) ship their own curl. ping -n 2 waits about 1 s.
set /a tries=0
:wait
set /a tries+=1
"%SystemRoot%\System32\curl.exe" -s -o nul http://127.0.0.1:8000/api/status && goto ready
if %tries% geq 30 goto slow
"%SystemRoot%\System32\ping.exe" -n 2 127.0.0.1 >nul
goto wait

:slow
echo The backend has not answered after about 90 s: check the "ASL Backend" window for an error.
echo The page will keep retrying on its own.

:ready
start "" http://127.0.0.1:5173
echo Opened http://127.0.0.1:5173 in your browser.
echo.
echo Keep the two server windows open; close them to stop the app.
echo Don't click inside them: Windows pauses a program while text is selected in its window (press Esc to resume).
echo.
pause
