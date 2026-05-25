import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_feature(img):
    img = cv2.resize(img, (128, 128))

    # ===== TIỀN XỬ LÝ: CLAHE =====
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # ===== 1. COLOR HIST (BGR) =====
    hist_bgr = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    hist_bgr = cv2.normalize(hist_bgr, hist_bgr).flatten()  # 512 chiều

    # ===== 2. COLOR HIST (HSV) =====
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()  # 512 chiều

    # ===== 3. MULTI-SCALE LBP =====
    lbp_features = []
    for radius, n_points in [(1, 8), (2, 16), (3, 24), (4, 32)]:
        lbp = local_binary_pattern(gray_clahe, P=n_points, R=radius, method='uniform')
        n_bins = n_points + 2
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        lbp_features.append(lbp_hist)
    lbp_concat = np.concatenate(lbp_features)  # 10+18+26+34 = 88 chiều

    # ===== 4. LBP THEO VÙNG 2x2 =====
    h, w = gray_clahe.shape
    region_features = []
    for i in range(2):
        for j in range(2):
            region = gray_clahe[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
            lbp_r = local_binary_pattern(region, P=8, R=1, method='uniform')
            hist_r, _ = np.histogram(lbp_r.ravel(), bins=10, range=(0, 10), density=True)
            region_features.append(hist_r)
    region_concat = np.concatenate(region_features)  # 4×10 = 40 chiều

    # Tổng: 512 + 512 + 88 + 40 = 1152 chiều
    return np.concatenate([hist_bgr, hist_hsv, lbp_concat, region_concat])