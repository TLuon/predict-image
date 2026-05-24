import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_feature(img):
    # resize vừa phải
    img = cv2.resize(img, (64, 64))

    # ===== 1. COLOR HIST =====
    hist = cv2.calcHist([img], [0, 1, 2], None,
                        [6, 6, 6], [0, 256]*3)
    hist = cv2.normalize(hist, hist).flatten()

    # ===== 2. LBP (quan trọng nhất) =====
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 nâng cấp LBP
    lbp = local_binary_pattern(gray, P=16, R=2, method='uniform')

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 18),
        range=(0, 17)
    )

    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    return np.concatenate([hist, lbp_hist])