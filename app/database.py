import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def get_mongo_client():
    uri = os.getenv("MONGO_URI")

    if not uri:
        raise ValueError("MONGO_URI not set in environment variables")
    return MongoClient(uri)

def get_games_collection():
    client = get_mongo_client()
    db = client["ataxx_db"]
    return db["games"]

def get_move_history_collection():
    client = get_mongo_client()
    db = client["ataxx_db"]
    return db["move_history"]
