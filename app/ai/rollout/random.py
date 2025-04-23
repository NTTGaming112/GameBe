from app.ai.ataxx_env import AtaxxEnvironment
import random

def random_rollout(env: AtaxxEnvironment, player: str, max_depth: int = 50) -> AtaxxEnvironment:
    try:
        current_env = env.clone()
        current_player = player
        depth = 0
        
        while not current_env.is_game_over() and depth < max_depth:
            if not current_env.has_valid_moves(current_player):
                current_player = "red" if current_player == "yellow" else "yellow"
                current_env.current_player = current_player
                depth += 1
                continue
                
            moves = current_env.get_valid_moves()
            if not moves:
                current_player = "red" if current_player == "yellow" else "yellow"
                current_env.current_player = current_player
                depth += 1
                continue
                
            move = random.choice(moves)
            current_env.make_move(move["from"], move["to"])
            current_player = "red" if current_player == "yellow" else "yellow"
            current_env.current_player = current_player
            depth += 1
        
        return current_env
    except Exception as e:
        raise Exception(f"Error in random_rollout: {str(e)}")