from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2

from app.services.image_processing import extract_feature
from app.services.model_loader import load_model
from app.utils.disease_info import disease_info

router = APIRouter()

# 🔥 Load cả model + scaler
model, scaler = load_model()


@router.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        # ===== READ IMAGE =====
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # ===== FEATURE =====
        feature = extract_feature(img)

        # 🔥 SCALE (BẮT BUỘC)
        feature = scaler.transform([feature])   # shape: (1, n_features)

        # ===== PREDICT =====
        probs = model.predict_proba(feature)[0]
        pred_index = np.argmax(probs)

        pred = model.classes_[pred_index]
        conf = float(probs[pred_index])

        # ===== THRESHOLD =====
        if conf < 0.4:   # bạn có thể chỉnh 0.3–0.5
            pred = "Không thể xác định chính xác, vui lòng thử lại với ảnh khác"

        # ===== RESPONSE =====
        return {
            "prediction": pred,
            "confidence": round(conf, 3),
            "description": disease_info.get(pred, "Không có mô tả"),
            "note": "AI chỉ mang tính chất tham khảo, không thay thế bác sĩ"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))