from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, NotFound

from app.ai.bot_trainer import get_trained_move, train_mcts
from app.database import get_db, get_games_collection
from app.models.game import GameCreate
from pydantic import ValidationError
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
async def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/bot-move/", methods=["POST"])
async def get_bot_move():
    data = request.get_json()
    if not data:
        raise BadRequest("Missing JSON body")

    board = data.get("board")
    current_player = data.get("current_player")

    if not board or not current_player:
        raise BadRequest("Board and current_player are required")

    move = get_trained_move(board, current_player)
    if not move:
        raise NotFound("No valid move found")

    return jsonify(move)


@app.route("/save_game", methods=["POST"])
def save_game():
   games_collection = get_games_collection()
   try:
       data = request.get_json()
       game = GameCreate(**data)
       games_collection.insert_one(game.dict())
       return jsonify({"message": "Game saved successfully"}), 200
   except ValidationError as e:
       return jsonify({"error": e.errors()}), 400
   except Exception as e:
       return jsonify({"error": str(e)}), 500
