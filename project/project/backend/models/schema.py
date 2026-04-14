"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PredictionResponse(BaseModel):
    id: Optional[str] = None
    status: str = "success"
    label: str
    confidence: float
    has_lesion: bool
    mask_coverage: float
    timestamp: str
    filename: str
    input_image_url: str
    mask_image_url: str
    overlay_image_url: str
    explanation: str
    error: Optional[str] = None


class HistoryItem(BaseModel):
    id: str
    timestamp: str
    status: str
    label: str
    confidence: float
    has_lesion: bool
    mask_coverage: float
    filename: str
    input_image: str
    mask_image: str
    overlay_image: str
    explanation: str
    error: Optional[str] = None


class HistoryResponse(BaseModel):
    count: int
    items: list


class DeleteResponse(BaseModel):
    deleted: int
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mongodb_connected: bool
    device: str
    message: str
