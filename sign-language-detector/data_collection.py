"""Step 1: Collect hand gesture images via webcam for each ASL letter."""

import cv2
import os

# --- Config ---
DATA_DIR = './data/raw'
CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z
IMAGES_PER_CLASS = 100

os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()

print("=== Sign Language Data Collector ===")
print("Press the LETTER KEY to set the label (A-Z)")
print("Press 'S' to save current frame")
print("Press 'Q' to quit\n")

current_label = None
save_count = {c: 0 for c in CLASSES}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    # Draw UI
    label_text = f"Label: {current_label}" if current_label else "Label: (press A-Z)"
    count_text = f"Saved: {save_count.get(current_label, 0)}/{IMAGES_PER_CLASS}" if current_label else ""
    cv2.putText(display, label_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(display, count_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    cv2.putText(display, "S=Save  Q=Quit", (10, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Data Collector', display)
    key = cv2.waitKey(1) & 0xFF

    # Set label
    if chr(key).upper() in CLASSES:
        current_label = chr(key).upper()
        label_dir = os.path.join(DATA_DIR, current_label)
        os.makedirs(label_dir, exist_ok=True)
        print(f"[INFO] Label set to: {current_label}")

    # Save frame
    elif key == ord('s') and current_label:
        count = save_count[current_label]
        if count < IMAGES_PER_CLASS:
            img_path = os.path.join(DATA_DIR, current_label, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            save_count[current_label] += 1
            print(f"[SAVED] {current_label}/{count}.jpg")
        else:
            print(f"[INFO] {current_label} already has {IMAGES_PER_CLASS} images.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n[DONE] Data collection complete!")
for c, cnt in save_count.items():
    if cnt > 0:
        print(f"  {c}: {cnt} images")
