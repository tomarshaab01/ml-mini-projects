#!/bin/bash
# ============================================================
#  Handwriting OCR + Grammar Fixer — One-Click Setup (Mac/Linux)
# ============================================================

echo ""
echo "======================================================"
echo "  ✍️  Handwriting OCR + Grammar Fixer — Setup"
echo "======================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found."
    echo "  → https://www.python.org/downloads/"
    exit 1
fi
echo "[OK] Python: $(python3 --version)"

# Check Tesseract
echo ""
echo "[CHECK] Tesseract OCR..."
if ! command -v tesseract &> /dev/null; then
    echo "[WARN] Tesseract OCR not found."
    echo "  Install it first:"
    echo "  • Ubuntu/Debian : sudo apt-get install tesseract-ocr"
    echo "  • macOS         : brew install tesseract"
    echo "  • Windows       : https://github.com/UB-Mannheim/tesseract/wiki"
    echo ""
    read -p "  Press ENTER after installing Tesseract, or Ctrl+C to cancel..."
else
    echo "[OK] Tesseract: $(tesseract --version | head -1)"
fi

# Virtual environment
echo ""
echo "[1/3] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "[OK] Virtual environment activated."

# Install Python deps
echo ""
echo "[2/3] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt
echo "[OK] Dependencies installed."

# Verify
echo ""
echo "[3/3] Verifying installation..."
python3 -c "import cv2; import pytesseract; import language_tool_python; import streamlit; print('[OK] All packages verified.')" 2>/dev/null || \
    echo "[WARN] Some packages may need manual check."

echo ""
echo "======================================================"
echo "  ✅  Setup complete!"
echo "======================================================"
echo ""
echo "  Run options:"
echo "  1. source venv/bin/activate"
echo ""
echo "  • Web UI  : streamlit run app.py"
echo "  • CLI     : python pipeline.py --image sample_images/sample1.jpg"
echo ""
