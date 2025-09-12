from pymongo import MongoClient
from app.database.model import User
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "piece_of_mind"
COLLECTION_NAME = "users"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db[COLLECTION_NAME]

def create_user(user_data: dict):
    user = User(**user_data)
    result = users_collection.insert_one(user.dict())
    return str(result.inserted_id)

def get_user_by_email(email: str):
    user = users_collection.find_one({"personal_info.email": email})
    return user

def update_user(email: str, update_data: dict):
    result = users_collection.update_one({"personal_info.email": email}, {"$set": update_data})
    return result.modified_count

def delete_user(email: str):
    result = users_collection.delete_one({"personal_info.email": email})
    return result.deleted_count
