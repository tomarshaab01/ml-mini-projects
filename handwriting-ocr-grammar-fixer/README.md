# ✍️ Handwriting OCR + Grammar Fixer

> A smart AI pipeline that **reads handwritten text from images** and **auto-corrects grammar** using NLP.  
> Built with Tesseract OCR + OpenCV preprocessing + LanguageTool grammar correction.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-red?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![LanguageTool](https://img.shields.io/badge/LanguageTool-NLP-purple?style=flat-square)

---

## 📌 Overview

This project builds a 3-stage AI pipeline:

```
Handwritten Image → [OpenCV Preprocessing] → [Tesseract OCR] → [Grammar Fixer] → Clean Text
```

1. **Preprocess** — Deskew, denoise, threshold the image for best OCR accuracy  
2. **Extract Text** — Tesseract OCR reads the handwritten content  
3. **Fix Grammar** — LanguageTool NLP corrects spelling, grammar, and punctuation  

---

## 🗂️ Project Structure

```
handwriting-ocr-grammar-fixer/
├── preprocess.py            # Image preprocessing (denoise, deskew, threshold)
├── ocr_extract.py           # Tesseract OCR text extraction
├── grammar_fix.py           # Grammar + spelling correction via LanguageTool
├── pipeline.py              # Full end-to-end pipeline (main script)
├── app.py                   # Streamlit web UI (drag-drop image → corrected text)
├── sample_images/           # Sample handwritten test images
├── outputs/                 # Extracted and corrected text results
├── requirements.txt         # Dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Install Tesseract OCR Engine

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```
**macOS:**
```bash
brew install tesseract
```
**Windows:**  
Download from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Install Python Dependencies

```bash
git clone https://github.com/tomarshaab01/ml-mini-projects.git
cd ml-mini-projects
git checkout handwriting-ocr-grammar-fixer
cd handwriting-ocr-grammar-fixer

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A — Command Line Pipeline
```bash
python pipeline.py --image sample_images/sample1.jpg
```

### Option B — Streamlit Web App
```bash
streamlit run app.py
```
- Drag and drop a handwritten image
- See extracted raw text + grammar-corrected output side by side

---

## 🧠 Pipeline Details

| Stage | Tool | What it does |
|-------|------|--------------|
| Preprocessing | OpenCV | Grayscale → Deskew → Denoise → Adaptive Threshold |
| OCR | Tesseract 5.x + pytesseract | Extracts text from preprocessed image |
| Grammar Fix | language-tool-python | Detects and corrects grammar, spelling, punctuation |
| Output | Plain text + JSON | Raw OCR text + corrected text + list of corrections |

---

## 📦 Dependencies

```
opencv-python
pytesseract
language-tool-python
numpy
pillow
streamlit
```

---

## 🔮 Future Improvements

- [ ] Support Hindi / multilingual handwriting
- [ ] Add PDF input support (multi-page)
- [ ] Fine-tune Tesseract with custom handwriting dataset
- [ ] Integrate GPT for context-aware grammar correction
- [ ] Export corrected output as DOCX/PDF

---

## 👨‍💻 Author

**Bharat Tomar** · B.Tech AI & ML @ AKGEC-AKTU  
[GitHub](https://github.com/tomarshaab01) · [LinkedIn](https://www.linkedin.com/in/bharat-tomar-026a87366)
