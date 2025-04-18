# backend/app/models/game.py
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

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

class Game(GameCreate):
    id: Optional[str] = None
    created_at: datetime = datetime.utcnow()