from collections import defaultdict
from typing import List, Dict, Optional
from app.database import get_games_collection, get_move_history_collection

def train_mcts(games_data: Optional[List[Dict]] = None, weight_decay: float = 0.9) -> None:
    games_collection = get_games_collection()
    move_history_collection = get_move_history_collection()
    move_history = defaultdict(lambda: defaultdict(float))

    if games_data is None:
        try:
            games_data = list(games_collection.find().limit(100))  # Giới hạn số ván để tăng tốc
        except Exception as e:
            print(f"Error retrieving games data from MongoDB: {e}")
            return

    if not games_data:
        print("No games data available for training.")
        return

    try:
        history_data = move_history_collection.find_one() or {"data": {}}
        for state_key, moves in history_data.get("data", {}).items():
            for move_key, score in moves.items():
                move_history[state_key][move_key] = score
    except Exception as e:
        print(f"Error loading move history from MongoDB: {e}")
        return

    for state_key in move_history:
        for move_key in move_history[state_key]:
            move_history[state_key][move_key] *= weight_decay

    for game in games_data:
        winner = game.get("winner")
        board_states = game.get("board_states", [])
        moves = game.get("moves", [])

        if not board_states or not moves or len(board_states) <= len(moves):
            continue

        for i, board_state in enumerate(board_states):
            if i >= len(moves):
                break
            state_key = str(tuple(tuple(row) for row in board_state))
            move = moves[i]
            move_key = f"{move['from_pos']['row']},{move['from_pos']['col']}-{move['to_pos']['row']},{move['to_pos']['col']}"
            if winner == "yellow" and i % 2 == 0:
                move_history[state_key][move_key] += 1.0
            elif winner == "red" and i % 2 == 1:
                move_history[state_key][move_key] += 1.0
            elif winner == "draw":
                move_history[state_key][move_key] += 0.5
            else:
                move_history[state_key][move_key] -= 0.2

    try:
        move_history_collection.delete_many({})
        move_history_collection.insert_one({"data": {k: dict(v) for k, v in move_history.items()}})
    except Exception as e:
        print(f"Error saving move history to MongoDB: {e}")