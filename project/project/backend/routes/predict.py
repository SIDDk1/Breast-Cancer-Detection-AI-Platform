"""
Prediction API routes.
POST /predict - Run inference
GET /history - Fetch prediction history
DELETE /history - Clear all history
"""

import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime

from ..services.inference import run_inference
from ..database.crud import insert_prediction, get_all_predictions, delete_all_predictions
from ..utils.helpers import save_image, ndarray_to_base64, generate_filename, build_explanation
from ..utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Directories for saved files
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an ultrasound image and get:
    - Segmentation mask
    - Overlay image
    - Classification label + confidence
    - Stored in MongoDB history
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")

    # Save uploaded file
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    orig_name = Path(file.filename).stem if file.filename else "uploaded"
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in orig_name)
    input_filename = f"input_{timestamp}_{safe_name}.png"
    input_path = UPLOADS_DIR / input_filename

    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    logger.info(f"🔄 Prediction request: {file.filename} → {input_path}")

    try:
        # Run full inference pipeline
        results = run_inference(str(input_path))

        # Save outputs to disk
        mask_filename = f"mask_{timestamp}_{safe_name}.png"
        overlay_filename = f"overlay_{timestamp}_{safe_name}.png"

        save_image(results["mask_array"], str(OUTPUTS_DIR), mask_filename, is_rgb=False)
        save_image(results["overlay_array"], str(OUTPUTS_DIR), overlay_filename, is_rgb=True)

        # Build explanation
        explanation = build_explanation(
            results["label"],
            results["confidence"],
            results["has_lesion"],
            results["mask_coverage"]
        )

        # Encode images as base64 for direct display in frontend
        original_b64 = ndarray_to_base64(results["original_array"], is_rgb=True)
        mask_b64 = ndarray_to_base64(results["mask_array"], is_rgb=False)
        overlay_b64 = ndarray_to_base64(results["overlay_array"], is_rgb=True)

        ts_str = datetime.utcnow().isoformat() + "Z"

        # Store in MongoDB history
        mongo_record = {
            "status": "success",
            "label": results["label"],
            "confidence": float(results["confidence"]),
            "has_lesion": bool(results["has_lesion"]),
            "mask_coverage": float(results["mask_coverage"]),
            "input_image": f"/images/{input_filename}",
            "mask_image": f"/outputs/{mask_filename}",
            "overlay_image": f"/outputs/{overlay_filename}",
            "explanation": explanation,
            "filename": file.filename or "unknown",
            "error": "",
        }
        doc_id = await insert_prediction(mongo_record)

        response = {
            "id": doc_id,
            "status": "success",
            "label": results["label"],
            "confidence": round(float(results["confidence"]), 4),
            "has_lesion": bool(results["has_lesion"]),
            "mask_coverage": round(float(results["mask_coverage"]) * 100, 2),
            "timestamp": ts_str,
            "filename": file.filename or "unknown",
            "original_image": f"data:image/png;base64,{original_b64}",
            "mask_image": f"data:image/png;base64,{mask_b64}",
            "overlay_image": f"data:image/png;base64,{overlay_b64}",
            "input_image_url": f"/images/{input_filename}",
            "mask_image_url": f"/outputs/{mask_filename}",
            "overlay_image_url": f"/outputs/{overlay_filename}",
            "explanation": explanation,
        }

        logger.info(f"✅ Prediction complete: {results['label']} ({results['confidence']:.4f})")
        return JSONResponse(content=response)

    except RuntimeError as e:
        # Model not loaded
        logger.error(f"Model error: {e}")
        err_record = {
            "status": "error",
            "label": "",
            "confidence": 0.0,
            "has_lesion": False,
            "mask_coverage": 0.0,
            "input_image": f"/images/{input_filename}",
            "mask_image": None,
            "overlay_image": None,
            "explanation": str(e),
            "filename": file.filename or "unknown",
            "error": str(e),
        }
        await insert_prediction(err_record)
        raise HTTPException(status_code=503, detail=f"Model not ready: {str(e)}")

    except Exception as e:
        logger.exception(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/history")
async def get_history():
    """
    Fetch all prediction history records, newest first.
    """
    items = await get_all_predictions(limit=100)
    return JSONResponse(content={"count": len(items), "items": items})


@router.delete("/history")
async def clear_history():
    """
    Delete ALL prediction history records from MongoDB.
    """
    deleted = await delete_all_predictions()
    return JSONResponse(content={
        "deleted": deleted,
        "message": f"Successfully deleted {deleted} records"
    })
