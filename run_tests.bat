@echo off
echo ========================================================
echo   Sign-ify tests (backend pytest incl. JS/Python parity)
echo ========================================================
cd /d "%~dp0backend"
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)
python -m pytest %*
pause
