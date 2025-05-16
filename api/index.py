from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
import time
from typing import Dict, Any, Tuple

from app.ai.ataxx_state import Ataxx
from app.ai.monte_carlo import get_monte_carlo_player

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    """Home endpoint with simple instructions."""
    return """
    <h1>Ataxx MCTS API</h1>
    """

@app.route('/get_move', methods=['POST'])
def get_move():
    """
    Get the best move for the current game state using MCTS.
    
    Request JSON format:
    {
        "board": 7x7 array representing the Ataxx board,
        "current_player": "red" or "yellow",
        "algorithm": "MC" (Basic Monte Carlo), "MCD" (Monte Carlo with Domain Knowledge), 
                    "AB+MCD" (Alpha-Beta + Monte Carlo), or "MINIMAX" (Alpha-Beta Minimax),
        "iterations": number of MCTS iterations (default: 300),
        "policy_args": {
            "switch_threshold": threshold for switching algorithms in AB+MCD (default: 31),
            "use_simulation_formula": whether to use simulation formula (default: false),
            "s1_ratio": ratio of S1 simulations to iterations (default: 1.0),
            "s2_ratio": ratio of S2 simulations to iterations (default: 1.0),
            "s3_ratio": ratio of S3 simulations to iterations (default: 0.5),
            "depth": search depth for Minimax algorithm (default: 4)
        }
    }
    
    Response JSON format:
    {
        "move": {
            "from": {"row": row, "col": col},
            "to": {"row": row, "col": col}
        },
        "execution_time": execution time in seconds,
        "current_player": current player after move
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        logger.info(f"Received request: {json.dumps(data)[:100]}...")
        
        required_fields = ["board", "current_player", "algorithm"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Chuyển đổi dữ liệu từ request sang Ataxx object
        state = Ataxx()
        # Gán lại board và current_player nếu có trong data
        if "board" in data:
            # Hỗ trợ cả kiểu string ('red', 'yellow', None) và số (1, -1, 0)
            def cell_to_num(cell):
                if cell == 'red' or cell == 1:
                    return 1
                elif cell == 'yellow' or cell == -1:
                    return -1
                else:
                    return 0
            state.board = [[cell_to_num(cell) for cell in row] for row in data["board"]]
            # Cập nhật lại self.balls cho đúng với board mới
            state.balls[1] = sum(cell == 1 for row in state.board for cell in row)
            state.balls[-1] = sum(cell == -1 for row in state.board for cell in row)

        if "current_player" in data:
            # Chuyển "red"/"yellow" sang PLAYER_ONE/PLAYER_TWO nếu cần
            if data["current_player"] == "red":
                state.turn_player = 1
            elif data["current_player"] == "yellow":
                state.turn_player = -1
            else:
                state.turn_player = data["current_player"]
        
        algorithm = data.get("algorithm", "MC")
        iterations = int(data.get("iterations", 300))
        
        # Extract specific parameters from policy_args if provided
        policy_args = data.get("policy_args", {})
        switch_threshold = policy_args.get("switch_threshold", 31)
        use_simulation_formula = policy_args.get("use_simulation_formula", False)
        s1_ratio = policy_args.get("s1_ratio", 1.0)
        s2_ratio = policy_args.get("s2_ratio", 1.0)
        s3_ratio = policy_args.get("s3_ratio", 0.5)
        depth = policy_args.get("depth", 4)
        
        start_time = time.time()
        
        # Correctly pass the state as the first argument
        mcts = get_monte_carlo_player(
            state, 
            mc_type=algorithm,
            number_simulations=iterations,
            switch_threshold=switch_threshold,
            use_simulation_formula=use_simulation_formula,
            s1_ratio=s1_ratio,
            s2_ratio=s2_ratio,
            s3_ratio=s3_ratio,
            depth=depth
        )
        
        # Chạy thuật toán và lấy nước đi tốt nhất
        if hasattr(mcts, 'get_move'):
            best_move = mcts.get_move()
        elif hasattr(mcts, 'get_play'):
            best_move = mcts.get_play()
        else:
            raise AttributeError(f"AI player object {type(mcts)} has no get_move or get_play method")
        
        execution_time = time.time() - start_time
        
        if best_move is None:
            return jsonify({
                "error": "No legal moves available",
                "execution_time": execution_time
            }), 200

        # best_move có thể là (CLONE_MOVE, (x, y)) hoặc (JUMP_MOVE, (x_dest, y_dest), (x_src, y_src))
        if best_move[0] == 'c':  # Clone move
            _, to_pos = best_move
            from_pos = None  # Clone move không có from_pos rõ ràng
        else:  # Jump move
            _, to_pos, from_pos = best_move

        # Trả về current_player dạng 'red' hoặc 'yellow'
        player_val = state.turn_player if callable(getattr(state, 'current_player', None)) else state.current_player
        player_str = 'red' if player_val == 1 else 'yellow'
        response = {
            "move": {
                "from": {"row": from_pos[0], "col": from_pos[1]} if from_pos else None,
                "to": {"row": to_pos[0], "col": to_pos[1]}
            },
            "execution_time": float(execution_time),
            "current_player": player_str,
        }
        
        logger.info(f"Responding with move: {json.dumps(response)}")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/get_legal_moves', methods=['POST'])
def get_legal_moves():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Chuyển đổi dữ liệu từ request sang Ataxx object
        state = Ataxx()
        # Gán lại board và current_player nếu có trong data
        if "board" in data:
            state.board = data["board"]
        if "current_player" in data:
            # Chuyển "red"/"yellow" sang PLAYER_ONE/PLAYER_TWO nếu cần
            if data["current_player"] == "red":
                state.turn_player = 1
            elif data["current_player"] == "yellow":
                state.turn_player = -1
            else:
                state.turn_player = data["current_player"]
        
        legal_moves = state.get_legal_moves()
        
        response = {
            "legal_moves": [
                {
                    "from_pos": list(from_pos),
                    "to_pos": list(to_pos)
                }
                for from_pos, to_pos in legal_moves
            ]
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate_state', methods=['POST'])
def evaluate_state():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Chuyển đổi dữ liệu từ request sang Ataxx object
        state = Ataxx()
        # Gán lại board và current_player nếu có trong data
        if "board" in data:
            state.board = data["board"]
        if "current_player" in data:
            # Chuyển "red"/"yellow" sang PLAYER_ONE/PLAYER_TWO nếu cần
            if data["current_player"] == "red":
                state.turn_player = 1
            elif data["current_player"] == "yellow":
                state.turn_player = -1
            else:
                state.turn_player = data["current_player"]
        
        winner = state.get_winner()
        is_terminal = state.is_terminal()
        piece_counts = state.get_pieces_count()
        
        response = {
            "winner": winner,
            "is_terminal": is_terminal,
            "piece_counts": piece_counts
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500