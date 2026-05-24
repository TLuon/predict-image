from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict_image import router

app = FastAPI()

# 🔥 THÊM ĐOẠN NÀY
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # cho phép tất cả (dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)