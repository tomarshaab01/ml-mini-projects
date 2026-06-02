"""Step 2: Extract 42 MediaPipe hand landmark features from collected images."""

import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

DATA_DIR = './data/raw'
OUTPUT_FILE = 'features.pkl'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

data = []
labels = []

classes = sorted(os.listdir(DATA_DIR))
print(f"[INFO] Found {len(classes)} classes: {classes}")

for label in tqdm(classes, desc="Extracting features"):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Normalize landmarks relative to bounding box
            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]
            x_min, y_min = min(x_vals), min(y_vals)

            features = []
            for lm in hand_landmarks.landmark:
                features.append(lm.x - x_min)
                features.append(lm.y - y_min)

            data.append(features)
            labels.append(label)

hands.close()

print(f"\n[INFO] Extracted features from {len(data)} images")

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': np.array(data), 'labels': np.array(labels)}, f)

print(f"[SAVED] Features saved to '{OUTPUT_FILE}'")
