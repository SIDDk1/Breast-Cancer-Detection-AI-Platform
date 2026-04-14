"""
CRUD operations for prediction history.
"""

from datetime import datetime
from typing import Optional, List
from bson import ObjectId
from .db import get_collection
from ..utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION = "predictions"


async def insert_prediction(data: dict) -> Optional[str]:
    """
    Insert a prediction record into MongoDB.
    Returns inserted document ID as string, or None on failure.
    """
    col = get_collection(COLLECTION)
    if col is None:
        logger.warning("MongoDB not available – skipping history save")
        return None

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": data.get("status", "success"),
        "label": data.get("label", ""),
        "confidence": data.get("confidence", 0.0),
        "has_lesion": data.get("has_lesion", False),
        "mask_coverage": data.get("mask_coverage", 0.0),
        "input_image": data.get("input_image", ""),
        "mask_image": data.get("mask_image", ""),
        "overlay_image": data.get("overlay_image", ""),
        "explanation": data.get("explanation", ""),
        "error": data.get("error", ""),
        "filename": data.get("filename", ""),
    }

    try:
        result = await col.insert_one(record)
        logger.info(f"History saved: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Failed to insert prediction: {e}")
        return None


async def get_all_predictions(limit: int = 50) -> List[dict]:
    """
    Retrieve all predictions sorted by newest first.
    """
    col = get_collection(COLLECTION)
    if col is None:
        return []

    try:
        cursor = col.find({}).sort("timestamp", -1).limit(limit)
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results
    except Exception as e:
        logger.error(f"Failed to fetch predictions: {e}")
        return []


async def delete_all_predictions() -> int:
    """
    Delete all prediction records.
    Returns count of deleted documents.
    """
    col = get_collection(COLLECTION)
    if col is None:
        return 0

    try:
        result = await col.delete_many({})
        deleted = result.deleted_count
        logger.info(f"Deleted {deleted} prediction records")
        return deleted
    except Exception as e:
        logger.error(f"Failed to delete predictions: {e}")
        return 0


async def get_prediction_by_id(prediction_id: str) -> Optional[dict]:
    """
    Get a single prediction by its MongoDB ObjectId string.
    """
    col = get_collection(COLLECTION)
    if col is None:
        return None

    try:
        doc = await col.find_one({"_id": ObjectId(prediction_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        logger.error(f"Failed to get prediction {prediction_id}: {e}")
        return None
