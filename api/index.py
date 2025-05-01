from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
import time
from typing import Dict, Any, Tuple

from app.ai.ataxx_env import AtaxxState
from app.ai.mcts_factory import create_mcts

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
    <p>Send a POST request to /get_move with a JSON payload containing the game state.</p>
    <p>Example payload:</p>
    <pre>
    {
        "board": [
            ["yellow", "empty", "empty", "empty", "empty", "empty", "red"],
            ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
            ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
            ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
            ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
            ["empty", "empty", "empty", "empty", "empty", "empty", "empty"],
            ["red", "empty", "empty", "empty", "empty", "empty", "yellow"]
        ],
        "current_player": "yellow",
        "algorithm": "mcts-fractional-dk",
        "iterations": 1000,
        "policy_type": "epolicy",
        "policy_args": {"epsilon": 0.1, "capture_weight": 1.0}
    }
    </pre>
    
    <h2>Available Algorithms:</h2>
    <ul>
        <li><strong>mcts-binary</strong>: Standard MCTS with binary (win/loss) outcomes</li>
        <li><strong>mcts-uct</strong>: MCTS with UCT and heuristic evaluation</li>
        <li><strong>mcts-fractional</strong>: MCTS with fractional scores based on piece counts</li>
        <li><strong>mcts-binary-dk</strong>: MCTS with binary outcomes and heuristic-based rollout</li>
        <li><strong>mcts-fractional-dk</strong>: MCTS with fractional scores and heuristic-based rollout</li>
        <li><strong>mcts-binary-minimax2</strong>: MCTS with binary outcomes and Minimax2 playout</li>
    </ul>
    
    <h2>Available Policies:</h2>
    <ul>
        <li><strong>random</strong>: Selects moves randomly during simulation</li>
        <li><strong>heuristic</strong>: Uses game-specific heuristics to guide move selection</li>
        <li><strong>ucb</strong>: Uses UCB formula to balance exploration and exploitation</li>
        <li><strong>epolicy</strong>: Combines heuristic and UCB policies</li>
    </ul>
    """

@app.route('/get_move', methods=['POST'])
def get_move():
    """
    Get the best move for the current game state using MCTS.
    
    Request JSON format:
    {
        "board": 7x7 array representing the Ataxx board,
        "current_player": "red" or "yellow",
        "algorithm": "mcts-binary", "mcts-uct", "mcts-fractional", "mcts-binary-dk", "mcts-fractional-dk", or "mcts-binary-minimax2",
        "iterations": number of MCTS iterations,
        "time_limit": optional time limit in seconds,
        "policy_type": "random", "heuristic", "ucb", or "epolicy",
        "policy_args": optional dictionary of policy parameters
    }
    
    Response JSON format:
    {
        "from_pos": [row, col],
        "to_pos": [row, col],
        "execution_time": execution time in seconds
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
        
        state = AtaxxState(
            board=data["board"],
            current_player=data["current_player"]
        )
        
        algorithm = data.get("algorithm", "mcts-binary")
        iterations = int(data.get("iterations", 1000))
        time_limit = data.get("time_limit")
        policy_type = data.get("policy_type", "random")
        policy_args = data.get("policy_args", {})
        
        start_time = time.time()
        
        mcts = create_mcts(algorithm, iterations, time_limit, policy_type, **policy_args)
        best_move = mcts.search(state)
        
        execution_time = time.time() - start_time
        
        if best_move is None:
            return jsonify({
                "error": "No legal moves available",
                "execution_time": execution_time
            }), 200
        
        from_pos, to_pos = best_move
        response = {
            "move": {
                "from": {
                    "row": from_pos[0],
                    "col": from_pos[1]
                },
                "to": {
                    "row": to_pos[0],
                    "col": to_pos[1]
                }
            },
            "execution_time": execution_time,
            "current_player": state.current_player,
        }
        
        logger.info(f"Responding with move: {response}")
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
        
        state = AtaxxState(
            board=data["board"],
            current_player=data["current_player"]
        )
        
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
        
        state = AtaxxState(
            board=data["board"],
            current_player=data["current_player"]
        )
        
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)