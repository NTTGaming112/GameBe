import os
from pymongo import MongoClient

def get_mongo_client():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI not set in environment variables")
    return MongoClient(uri)

def get_games_collection():
    client = get_mongo_client()
    db = client["ataxx_db"]
    return db["games"]

def get_move_history_collection():
    client = get_mongo_client()
    db = client["ataxx_db"]
    return db["move_history"]