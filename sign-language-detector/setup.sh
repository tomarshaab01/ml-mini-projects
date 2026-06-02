#!/bin/bash
# ============================================================
#  Sign Language Detector — One-Click Setup (Mac/Linux)
# ============================================================

echo ""
echo "======================================================"
echo "  🖐️  Sign Language Detector — Setup Script"
echo "======================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed."
    echo "  → Download from https://www.python.org/downloads/"
    exit 1
fi

PYVER=$(python3 -c 'import sys; print(sys.version_info.major, sys.version_info.minor)')
echo "[OK] Python found: $(python3 --version)"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "[ERROR] pip3 not found. Installing..."
    python3 -m ensurepip --upgrade
fi

# Create virtual environment
echo ""
echo "[1/3] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "[OK] Virtual environment created and activated."

# Install dependencies
echo ""
echo "[2/3] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt
echo "[OK] All dependencies installed."

# Test imports
echo ""
echo "[3/3] Verifying installation..."
python3 -c "import cv2; import mediapipe; import sklearn; print('[OK] All packages verified.')" 2>/dev/null || \
    echo "[WARN] Some packages may need manual verification."

echo ""
echo "======================================================"
echo "  ✅  Setup complete!"
echo "======================================================"
echo ""
echo "  Run the project:"
echo "  1. source venv/bin/activate"
echo "  2. python data_collection.py    # collect training data"
echo "  3. python extract_features.py   # extract landmarks"
echo "  4. python train_model.py         # train model"
echo "  5. python predict_realtime.py    # live detection!"
echo ""
