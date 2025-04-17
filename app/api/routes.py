from fastapi import APIRouter, HTTPException
from app.models.game import Game, GameCreate
from app.database import get_games_collection
from app.ai.bot_trainer import train_mcts, get_trained_move

router = APIRouter()

@router.post("/games/")
async def save_game(game: GameCreate):
    db = get_games_collection()
    game_dict = game.dict()
    db.insert_one(game_dict)
    games = list(db.find())
    train_mcts(games)
    return {"message": "Game saved and bot trained"}

@router.post("/bot-move/")
async def get_bot_move(request: dict):
    board = request.get("board")
    current_player = request.get("current_player")
    if not board or not current_player:
        raise HTTPException(status_code=400, detail="Board and current_player are required")
    move = get_trained_move(board, current_player)
    if not move:
        raise HTTPException(status_code=404, detail="No valid move found")
    return move

@router.get("/games/")
async def get_games():
    db = get_games_collection()
    games = list(db.find({}, {"_id": 0}))
    return games