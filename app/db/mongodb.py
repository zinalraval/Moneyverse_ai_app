# app/db/mongodb.py

from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Get MongoDB connection details from settings
MONGO_URI = settings.MONGODB_URL
DB_NAME = settings.MONGODB_DB

# Commented out MongoDB client creation and connection test to disable MongoDB if not required
# try:
#     # Create MongoDB client
#     client = AsyncIOMotorClient(MONGO_URI)
#     mongodb = client[DB_NAME]
#     # Test connection
#     client.admin.command('ping')
#     logger.info("Successfully connected to MongoDB")
# except Exception as e:
#     logger.error(f"Failed to connect to MongoDB: {str(e)}")
#     raise

# Collections
# pdf_collection = mongodb.get_collection("pdfs")

# Dependency to get MongoDB client
# async def get_mongo_db():
#     """Get MongoDB database instance."""
#     try:
#         yield mongodb
#     except Exception as e:
#         logger.error(f"Error getting MongoDB connection: {str(e)}")
#         raise

# Comment out client export to disable MongoDB integration
# client = AsyncIOMotorClient(MONGO_URI)
