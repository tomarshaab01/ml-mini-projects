@echo off
REM ============================================================
REM  Sign Language Detector — One-Click Setup (Windows)
REM ============================================================

echo.
echo ======================================================
echo   Sign Language Detector -- Setup Script (Windows)
echo ======================================================
echo.

REM Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo   Download from https://www.python.org/downloads/
    echo   Make sure to check 'Add Python to PATH' during install!
    pause
    exit /b 1
)

FOR /F "tokens=*" %%i IN ('python --version') DO SET PYVER=%%i
echo [OK] Found %PYVER%

REM Create virtual environment
echo.
echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
echo [OK] Virtual environment ready.

REM Upgrade pip
echo.
echo [2/3] Installing dependencies...
python -m pip install --upgrade pip -q
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)
echo [OK] All dependencies installed.

REM Verify
echo.
echo [3/3] Verifying installation...
python -c "import cv2; import mediapipe; import sklearn; print('[OK] All packages verified.')" 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] Verification check had issues. Try running manually.
)

echo.
echo ======================================================
echo   Setup complete!
echo ======================================================
echo.
echo   Next steps:
echo   1. venv\Scripts\activate
echo   2. python data_collection.py
echo   3. python extract_features.py
echo   4. python train_model.py
echo   5. python predict_realtime.py
echo.
pause
