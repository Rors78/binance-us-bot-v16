
@echo off
echo =============================================
echo   Binance US One-Switch Bot v16.1 — Micro Mode
echo =============================================
echo.

python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    pause
    exit /b
)

echo Installing/Updating dependencies...
pip install -r requirements.txt

echo Launching bot...
streamlit run one_switch_app.py

pause
