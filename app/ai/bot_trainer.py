from collections import defaultdict
from typing import List, Dict, Optional
from app.ai.mcts.mcts import MCTS
from app.database import get_games_collection, get_move_history_collection

def get_trained_move(board: List[List[str]], current_player: str) -> Optional[Dict]:
    move_history_collection = get_move_history_collection()
    move_history = defaultdict(lambda: defaultdict(float))

    try:
        history_data = move_history_collection.find_one() or {"data": {}}
        for state_key, moves in history_data.get("data", {}).items():
            for move_key, score in moves.items():
                move_history[state_key][move_key] = score
    except Exception as e:
        print(f"Error loading move history from MongoDB: {e}")
        return None

    mcts = MCTS(board, current_player, move_history=move_history)
    return mcts.search()