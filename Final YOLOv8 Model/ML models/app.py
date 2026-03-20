"""
TrustLens AI — YOLOv8 Inference Microservice
Deployed on HuggingFace Spaces (Docker, port 7860)

This is the PRODUCTION version of the Colab notebook's FastAPI server.
- Removed: pyngrok, nest_asyncio, threading, Google Drive paths
- Kept:    exact same calculate_severity() logic, same class weights,
           same detection schema your Colab notebook produces

Classes detected: crack, scratch, stain  (skips 'good')

Endpoint:
  POST /predict
    Body: multipart/form-data  { file: <image> }
    Returns: JSON array of detections
"""

import os
import math
import logging
import tempfile
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TrustLens YOLOv8 Inference",
    description="Damage detection: crack / scratch / stain",
    version="2.0.0",
)

# Allow all origins — the Render backend POSTs images here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model — same path logic as Colab, lazy load on first request
# best_v2.pt must be in the same folder as this app.py
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_v2.pt")

# Same class weights as your Colab notebook
CLASS_WEIGHTS = {'crack': 1.0, 'scratch': 0.6, 'stain': 0.4}

_model = None


def get_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        return None
    try:
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
        logger.info("✅ YOLOv8 model loaded: best_v2.pt")
        return _model
    except Exception as exc:
        logger.error(f"❌ Failed to load model: {exc}")
        return None


# ---------------------------------------------------------------------------
# Response schema
# label / confidence / bbox / area_pct
# matches exactly what backend/services/yolo.py expects
# ---------------------------------------------------------------------------
class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]   # [x1, y1, x2, y2]
    area_pct: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "TrustLens YOLOv8",
        "status": "running",
        "model": "phone_damage_v2",
        "classes": ["crack", "scratch", "stain"],
        "model_file_exists": os.path.exists(MODEL_PATH),
    }


@app.get("/health")
def health():
    model = get_model()
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=List[Detection])
async def predict(file: UploadFile = File(...)):
    """
    Accept an image, run best_v2.pt inference, return detections.

    Detection logic is identical to the Colab notebook's calculate_severity()
    but returns raw detections only — severity scoring stays in the backend
    (backend/services/severity.py) so logic lives in one place.

    Returns empty list [] if no damage is detected.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")

    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable — best_v2.pt not found in container"
        )

    # Save upload to a temp file (same pattern as Colab's /content/temp_{filename})
    suffix = os.path.splitext(file.filename or "img.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # --- Inference (identical to Colab notebook) ---
        results = model(tmp_path, verbose=False)[0]
        img_h, img_w = results.orig_shape
        img_area = img_h * img_w

        detections = []

        if len(results.boxes) == 0:
            return []  # no damage detected

        for box in results.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]

            # Skip 'good' — not a damage class (same as Colab notebook)
            if class_name == "good":
                continue

            confidence = round(float(box.conf), 2)
            x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
            box_area = (x2 - x1) * (y2 - y1)
            area_pct = round((box_area / img_area) * 100, 2)

            detections.append(Detection(
                label=class_name,
                confidence=confidence,
                bbox=[x1, y1, x2, y2],
                area_pct=area_pct,
            ))

        logger.info(f"Detected {len(detections)} damage(s) in {file.filename}")
        return detections

    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}")

    finally:
        os.unlink(tmp_path)  # always clean up temp file