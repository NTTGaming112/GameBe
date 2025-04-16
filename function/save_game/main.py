# backend-serverless/functions/save_game/main.py
import functions_framework
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["ataxx"]
games_collection = db["games"]

app = FastAPI()

class Position(BaseModel):
    row: int
    col: int

class Move(BaseModel):
    from_pos: Position
    to_pos: Position

class GameCreate(BaseModel):
    board_states: List[List[List[Optional[str]]]]
    moves: List[Move]
    winner: str

@functions_framework.http
def save_game(request):
    if request.method != "POST":
        return {"statusCode": 405, "body": "Method Not Allowed"}

    try:
        data = request.get_json()
        game = GameCreate(**data)
        game_dict = game.dict()
        games_collection.insert_one(game_dict)
        return {"statusCode": 200, "body": "Game saved successfully"}
    except Exception as e:
        return {"statusCode": 400, "body": str(e)}