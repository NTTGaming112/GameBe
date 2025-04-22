from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, NotFound
from typing import List, Dict, Any

from app.ai.bot_trainer import play_bot_vs_bot
from app.database import get_games_collection
from app.models.game import GameCreate
from pydantic import ValidationError
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/play_bot_vs_bot/", methods=["POST"])
def play_bot_vs_bot_api():
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Missing JSON body")

        initial_board = data.get("board")
        if not initial_board:
            raise BadRequest("Board is required")

        if not isinstance(initial_board, list) or not all(isinstance(row, list) for row in initial_board):
            raise BadRequest("Board must be a 2D list")

        yellow_config = data.get("yellow", {"algorithm": "mcts-binary", "iterations": 300})
        red_config = data.get("red", {"algorithm": "mcts-binary", "iterations": 300})

        # Kiểm tra cấu hình của Yellow và Red
        valid_algorithms = [
            "random", "minimax",
            "mcts-binary", "mcts-binary-dk", "mcts-fractional",
            "mcts-fractional-dk", "mcts-minimax"
        ]
        if yellow_config["algorithm"] not in valid_algorithms:
            raise BadRequest(f"Yellow algorithm must be one of {valid_algorithms}")
        if red_config["algorithm"] not in valid_algorithms:
            raise BadRequest(f"Red algorithm must be one of {valid_algorithms}")
        if not isinstance(yellow_config["iterations"], int) or yellow_config["iterations"] < 1:
            raise BadRequest("Yellow iterations must be a positive integer")
        if not isinstance(red_config["iterations"], int) or red_config["iterations"] < 1:
            raise BadRequest("Red iterations must be a positive integer")

        # Gọi hàm play_bot_vs_bot với logic mới
        result = play_bot_vs_bot(
            initial_board=initial_board,
            yellow_algorithm=yellow_config["algorithm"],
            yellow_iterations=yellow_config["iterations"],
            red_algorithm=red_config["algorithm"],
            red_iterations=red_config["iterations"],
        )

        if "error" in result:
            raise BadRequest(result["error"])

        return jsonify(result), 200

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/save_game", methods=["POST"])
def save_game():
    games_collection = get_games_collection()
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Missing JSON body")

        # Validate dữ liệu đầu vào bằng Pydantic model GameCreate
        game_create = GameCreate(**data)

        # Chuyển dữ liệu Pydantic thành dict để lưu vào MongoDB
        game_data = game_create.dict()

        # Lưu vào MongoDB
        games_collection.insert_one(game_data)
        return jsonify({"message": "Game saved successfully"}), 200

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500