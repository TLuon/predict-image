import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

from feature_extraction import extract_feature
# venv\Scripts\activate 
# python training/train_model.py
# uvicorn app.main:app --reload
DATA_DIR = "data/processed"
MODEL_PATH = "model/svm_model.pkl"

# ===== LOAD hoặc BUILD DATA =====
if os.path.exists("X.npy") and os.path.exists("y.npy"):
    print("Loading cached features...")
    X = np.load("X.npy")
    y = np.load("y.npy")
else:
    print("Extracting features from images...")
    X = []
    y = []

    for label in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, label)

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            feature = extract_feature(img)

            X.append(feature)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("Saving features...")
    np.save("X.npy", X)
    np.save("y.npy", y)

print("Total samples:", len(X))

# ===== CHIA DATA =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===== SCALE DATA (CỰC QUAN TRỌNG) =====
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===== MODEL (SVM) =====
print("Start training...")

model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced',
    probability=True
)

model.fit(X_train, y_train)

print("Training finished!")

# ===== ĐÁNH GIÁ =====
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ===== LƯU MODEL + SCALER =====
os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, "model/scaler.pkl")

print("Model saved at:", MODEL_PATH)