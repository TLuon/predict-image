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


def augment_image(img):
    variants = []
    variants.append(cv2.flip(img, 1))
    variants.append(cv2.flip(img, 0))
    variants.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    variants.append(cv2.rotate(img, cv2.ROTATE_180))
    return variants


# ===== LOAD hoặc BUILD DATA =====
if os.path.exists("X.npy") and os.path.exists("y.npy") and os.path.exists("imgs.npy"):
    print("Loading cached features...")
    X = np.load("X.npy", allow_pickle=True)
    y = np.load("y.npy", allow_pickle=True)
    imgs = np.load("imgs.npy", allow_pickle=True)
else:
    print("Extracting features from images...")
    X = []
    y = []
    imgs = []

    for label in os.listdir(DATA_DIR):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            X.append(extract_feature(img))
            y.append(label)
            imgs.append(img)

    X = np.array(X)
    y = np.array(y)
    imgs = np.array(imgs, dtype=object)

    print("Saving features...")
    np.save("X.npy", X)
    np.save("y.npy", y)
    np.save("imgs.npy", imgs)

print("Total samples:", len(X))

# ===== CHIA DATA =====
X_train, X_test, y_train, y_test, imgs_train, _ = train_test_split(
    X, y, imgs, test_size=0.2, random_state=42, stratify=y
)

# ===== AUGMENT CHỈ TRÊN TRAIN =====
print("Augmenting train set...")
X_train_aug = list(X_train)
y_train_aug = list(y_train)

for img, label in zip(imgs_train, y_train):
    for aug_img in augment_image(img):
        X_train_aug.append(extract_feature(aug_img))
        y_train_aug.append(label)

X_train_aug = np.array(X_train_aug)
y_train_aug = np.array(y_train_aug)
print("Train samples after augment:", len(X_train_aug))
print("Test samples:", len(X_test))

# ===== SCALE =====
scaler = StandardScaler()
X_train_aug = scaler.fit_transform(X_train_aug)
X_test = scaler.transform(X_test)

# ===== TRAIN =====
print("Start training...")
model = SVC(
    kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True
)
model.fit(X_train_aug, y_train_aug)
print("Training finished!")

# ===== ĐÁNH GIÁ =====
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ===== LƯU =====
os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, "model/scaler.pkl")
print("Model saved at:", MODEL_PATH)
