from pymongo import MongoClient
from dotenv import load_dotenv
import os
from functools import lru_cache

load_dotenv()

@lru_cache(maxsize=1)
def get_db():
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client["ataxx"]
    return db["games"]


def get_games_collection():
    db = get_db()

    return db["games"]
def get_move_history_collection():
    db = get_db()
    return db["move_history"]

