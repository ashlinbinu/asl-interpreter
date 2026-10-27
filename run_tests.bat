@echo off
echo ========================================================
echo   Handspeak tests (backend pytest incl. JS/Python parity)
echo ========================================================
cd /d "%~dp0backend"
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)
python -m pytest %*
pause
