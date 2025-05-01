from app.ai.ataxx_env import AtaxxEnvironment
import random

def random_rollout(env: AtaxxEnvironment, player: str) -> AtaxxEnvironment:
    if not isinstance(env, AtaxxEnvironment):
        raise ValueError("Invalid environment: must be AtaxxEnvironment")
    if player not in ["yellow", "red"]:
        raise ValueError("Invalid player: must be 'yellow' or 'red'")
    
    board_backup = env.board.copy()
    player_backup = env.current_player
    try:
        while not env.is_game_over():
            moves = env.get_valid_moves()
            if not moves:
                break
            move = random.choice(moves)
            env.make_move(move["from"], move["to"])
        return env
    except Exception as e:
        raise RuntimeError(f"Error in random_rollout: {str(e)}")
    finally:
        env.board = board_backup
        env.current_player = player_backup