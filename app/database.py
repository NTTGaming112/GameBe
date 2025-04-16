# backend/app/database.py
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["ataxx"]
games_collection = db["games"]
move_history_collection = db["move_history"]

def get_games_collection():
    return games_collection

def get_move_history_collection():
    return move_history_collection