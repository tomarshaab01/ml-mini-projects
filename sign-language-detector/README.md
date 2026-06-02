# 🖐️ Sign Language Detector

> Real-time hand gesture recognition using **MediaPipe + OpenCV + scikit-learn**  
> Translates ASL (American Sign Language) alphabet gestures into text via webcam.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-yellow?style=flat-square)

---

## 📌 Overview

This project builds a complete pipeline that:
1. **Captures** hand landmarks in real-time using MediaPipe Hands
2. **Extracts** 21 keypoint coordinates (x, y) as features
3. **Classifies** the gesture using a trained Random Forest model
4. **Displays** the predicted letter on the live webcam feed

---

## 🗂️ Project Structure

```
sign-language-detector/
├── data_collection.py       # Collect training images via webcam
├── extract_features.py      # Extract MediaPipe landmarks from images
├── train_model.py           # Train Random Forest classifier
├── predict_realtime.py      # Live webcam prediction (main app)
├── model/
│   └── sign_model.pkl       # Saved trained model (after training)
├── data/
│   └── raw/                 # Collected gesture images (A-Z folders)
├── features.pkl             # Extracted landmark features
├── requirements.txt         # Dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repo and switch to this branch
git clone https://github.com/tomarshaab01/ml-mini-projects.git
cd ml-mini-projects
git checkout sign-language-detector
cd sign-language-detector

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Collect Training Data
```bash
python data_collection.py
```
- Press keys `A–Z` to set the label
- Press `S` to save a frame
- Collect ~100 images per letter

### Step 2 — Extract Features
```bash
python extract_features.py
```
- Processes all images → extracts 42 landmark features per image
- Saves `features.pkl`

### Step 3 — Train the Model
```bash
python train_model.py
```
- Trains Random Forest on extracted features
- Saves model to `model/sign_model.pkl`
- Prints accuracy report

### Step 4 — Run Live Detection
```bash
python predict_realtime.py
```
- Opens webcam feed
- Detects hand and predicts ASL letter in real-time
- Press `Q` to quit

---

## 🧠 Model Details

| Component | Detail |
|-----------|--------|
| Feature Extraction | MediaPipe Hands — 21 landmarks × (x, y) = 42 features |
| Classifier | Random Forest (100 estimators) |
| Input | Webcam frame (BGR → RGB) |
| Output | Predicted ASL letter + confidence |
| Accuracy | ~95%+ on collected dataset |

---

## 📦 Dependencies

```
opencv-python
mediapipe
scikit-learn
numpy
pickle5
```

---

## 🔮 Future Improvements

- [ ] Add word-level prediction (sequence of letters → word)
- [ ] Support dynamic gestures (motion-based signs)
- [ ] Build a Streamlit web UI
- [ ] Add text-to-speech output
- [ ] Extend to ISL (Indian Sign Language)

---

## 👨‍💻 Author

**Bharat Tomar** · B.Tech AI & ML @ AKGEC-AKTU  
[GitHub](https://github.com/tomarshaab01) · [LinkedIn](https://www.linkedin.com/in/bharat-tomar-026a87366)
