"""Step 4: Real-time sign language detection from webcam feed."""

import cv2
import pickle
import numpy as np
import mediapipe as mp

MODEL_FILE = 'model/sign_model.pkl'

# Load model
with open(MODEL_FILE, 'rb') as f:
    obj = pickle.load(f)
model = obj['model']
le = obj['encoder']

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()

print("[INFO] Starting real-time sign language detection...")
print("[INFO] Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    predicted_char = None
    confidence = 0.0

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                               mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2))

        # Extract & normalize features
        x_vals = [lm.x for lm in hand_landmarks.landmark]
        y_vals = [lm.y for lm in hand_landmarks.landmark]
        x_min, y_min = min(x_vals), min(y_vals)

        features = []
        for lm in hand_landmarks.landmark:
            features.append(lm.x - x_min)
            features.append(lm.y - y_min)

        features = np.array(features).reshape(1, -1)
        pred_encoded = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = np.max(proba)
        predicted_char = le.inverse_transform([pred_encoded])[0]

        # Bounding box
        x_px = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_px = [int(lm.y * h) for lm in hand_landmarks.landmark]
        x1, y1 = max(min(x_px) - 20, 0), max(min(y_px) - 20, 0)
        x2, y2 = min(max(x_px) + 20, w), min(max(y_px) + 20, h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prediction label
        label = f"{predicted_char}  ({confidence*100:.1f}%)"
        cv2.rectangle(frame, (x1, y1 - 45), (x1 + len(label) * 18, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    # Title bar
    cv2.putText(frame, "Sign Language Detector | Q=Quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Sign Language Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Detection stopped.")
