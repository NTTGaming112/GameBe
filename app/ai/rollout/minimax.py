from app.ai.ataxx_env import AtaxxEnvironment
from app.ai.variants.full_minimax import FullMinimax
from typing import Dict, Any

def minimax_rollout(env: AtaxxEnvironment, player: str, depth: int = 2, max_depth: int = 50) -> AtaxxEnvironment:
    try:
        current_env = env.clone()
        current_player = player
        rollout_depth = 0
        
        while not current_env.is_game_over() and rollout_depth < max_depth:
            if not current_env.has_valid_moves(current_player):
                current_player = "red" if current_player == "yellow" else "yellow"
                current_env.current_player = current_player
                rollout_depth += 1
                continue
                
            # Sử dụng FullMinimax để chọn nước đi
            minimax = FullMinimax(current_env, current_player, depth=depth)
            move = minimax.run()
            
            if move:
                current_env.make_move(move["from"], move["to"])
            else:
                current_player = "red" if current_player == "yellow" else "yellow"
                current_env.current_player = current_player
                rollout_depth += 1
                continue
                
            current_player = "red" if current_player == "yellow" else "yellow"
            current_env.current_player = current_player
            rollout_depth += 1
        
        return current_env
    except Exception as e:
        raise Exception(f"Error in minimax_rollout: {str(e)}")