"""
MongoDB database connection management.
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from ..utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "breast_cancer_saas")

_client = None
_db = None


async def connect_db():
    global _client, _db
    try:
        _client = AsyncIOMotorClient(
            MONGO_URL,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000
        )
        # Test connection
        await _client.admin.command("ping")
        _db = _client[DB_NAME]
        logger.info(f"✅ MongoDB connected: {MONGO_URL} / {DB_NAME}")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        _client = None
        _db = None
        return False


async def disconnect_db():
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB disconnected")


def get_db():
    return _db


def get_collection(name: str):
    db = get_db()
    if db is None:
        return None
    return db[name]
