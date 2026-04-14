"""
FastAPI main application entry point.
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import torch
from dotenv import load_dotenv

load_dotenv()

from backend.database.db import connect_db, disconnect_db, get_db
from backend.services.inference import load_models, get_device
from backend.routes.predict import router as predict_router
from backend.routes.report  import router as report_router
from backend.utils.logger import get_logger

logger = get_logger("main")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("🚀 Starting Breast Cancer AI SaaS Backend")
    
    # Connect MongoDB
    mongo_ok = await connect_db()
    if not mongo_ok:
        logger.warning("⚠️ MongoDB unavailable – history features will be disabled")

    # Load AI models
    model_ok = load_models(str(WEIGHTS_DIR))
    if model_ok:
        logger.info("✅ Models loaded successfully")
    else:
        logger.warning("⚠️ Models not found – run backend/services/train.py first")

    logger.info("🌐 API server ready")
    yield

    # Shutdown
    await disconnect_db()
    logger.info("👋 Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Breast Cancer AI SaaS API",
    description="Attention U-Net segmentation + CNN classification for breast ultrasound images",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for saved images
# Serve both uploads and outputs from /images/
app.mount("/images", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
# Trick: also serve outputs
from starlette.staticfiles import StaticFiles as SFiles
app.mount("/outputs", SFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Routes
app.include_router(predict_router, prefix="/api", tags=["Prediction"])
app.include_router(report_router,  prefix="/api", tags=["Report"])


@app.get("/health", tags=["System"])
async def health():
    """API health check."""
    from backend.services.inference import _seg_model, _cls_model
    
    db = get_db()
    device = get_device()

    return JSONResponse(content={
        "status": "ok",
        "model_loaded": _seg_model is not None,
        "cls_model_loaded": _cls_model is not None,
        "mongodb_connected": db is not None,
        "device": str(device),
        "message": "Breast Cancer AI API is running",
    })


@app.get("/", tags=["System"])
async def root():
    return {"message": "Breast Cancer AI SaaS API", "docs": "/docs"}
