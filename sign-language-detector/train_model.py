"""Step 3: Train a Random Forest classifier on extracted landmark features."""

import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

FEATURES_FILE = 'features.pkl'
MODEL_DIR = 'model'
MODEL_FILE = os.path.join(MODEL_DIR, 'sign_model.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)

# Load features
print("[INFO] Loading features...")
with open(FEATURES_FILE, 'rb') as f:
    dataset = pickle.load(f)

X = np.array(dataset['data'])
y = np.array(dataset['labels'])

print(f"[INFO] Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

# Train
print("[INFO] Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")
print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))

# Save model + encoder
with open(MODEL_FILE, 'wb') as f:
    pickle.dump({'model': model, 'encoder': le}, f)

print(f"[SAVED] Model saved to '{MODEL_FILE}'")
