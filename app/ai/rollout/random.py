import random

def random_rollout(env, bot_player, max_depth=20):  
    depth = 0
    current_player = bot_player
    while not env.is_game_over() and depth < max_depth:
        if not env.has_valid_moves(current_player):
            current_player = "red" if current_player == "yellow" else "yellow"
            continue
        moves = env.get_valid_moves()
        if not moves:
            break
        move = random.choice(moves)
        env.make_move(move["from"], move["to"])
        current_player = "red" if current_player == "yellow" else "yellow"
        depth += 1
    return env