@echo off
REM ============================================================
REM  Handwriting OCR + Grammar Fixer — One-Click Setup (Windows)
REM ============================================================

echo.
echo ======================================================
echo   Handwriting OCR + Grammar Fixer -- Setup (Windows)
echo ======================================================
echo.

REM Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Get it from https://www.python.org/downloads/
    echo         Check 'Add Python to PATH' during install!
    pause
    exit /b 1
)
FOR /F "tokens=*" %%i IN ('python --version') DO SET PYVER=%%i
echo [OK] %PYVER%

REM Check Tesseract
echo.
tesseract --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] Tesseract OCR not found!
    echo.
    echo   Please install Tesseract first:
    echo   https://github.com/UB-Mannheim/tesseract/wiki
    echo.
    echo   After installing, re-run this script.
    echo   Also add Tesseract to PATH or set path in ocr_extract.py
    pause
    exit /b 1
) ELSE (
    echo [OK] Tesseract found.
)

REM Virtual environment
echo.
echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
echo [OK] Virtual environment ready.

REM Install deps
echo.
echo [2/3] Installing Python dependencies...
python -m pip install --upgrade pip -q
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Install failed. Check internet connection.
    pause
    exit /b 1
)
echo [OK] All dependencies installed.

REM Verify
echo.
echo [3/3] Verifying...
python -c "import cv2; import pytesseract; import streamlit; print('[OK] All packages verified.')" 2>nul

echo.
echo ======================================================
echo   Setup complete!
echo ======================================================
echo.
echo   Run options:
echo   1. venv\Scripts\activate
echo   2a. streamlit run app.py          (Web UI)
echo   2b. python pipeline.py --image sample_images\sample1.jpg  (CLI)
echo.
pause
