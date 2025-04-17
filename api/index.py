from flask import Flask, jsonify, request
from app.models.game import Game, GameCreate
from app.database import get_games_collection
from app.ai.bot_trainer import train_mcts, get_trained_move
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route("/")
def hello_world():
    return jsonify(message="Ataxx Backend API")

@app.route("/games/", methods=["POST"])
def save_game():
    game = request.get_json()
    if not game:
        return jsonify(message="Game data is missing"), 400
    db = get_games_collection()
    game_dict = game
    db.insert_one(game_dict)
    games = list(db.find())
    train_mcts(games)
    return jsonify(message="Game saved and bot trained"), 200

@app.route("/bot-move/", methods=["POST"])
def get_bot_move():
    request_data = request.get_json()
    board = request_data.get("board")
    current_player = request_data.get("current_player")
    if not board or not current_player:
        return jsonify(message="Board and current_player are required"), 400
    move = get_trained_move(board, current_player)
    if not move:
        return jsonify(message="No valid move found"), 404
    return jsonify(move)

@app.route("/games/", methods=["GET"])
def get_games():
    db = get_games_collection()
    games = list(db.find({}, {"_id": 0}))
    return jsonify(games)

if __name__ == "__main__":
    app.run(debug=True)
