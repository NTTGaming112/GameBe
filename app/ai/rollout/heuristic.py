import random

def heuristic_rollout(env, bot_player, max_depth=10):
    depth = 0
    current_player = bot_player
    while not env.is_game_over() and depth < max_depth:
        if not env.has_valid_moves(current_player):
            current_player = "red" if current_player == "yellow" else "yellow"
            continue
        moves = env.get_valid_moves()
        if not moves:
            break
        # Sắp xếp nước đi theo giá trị heuristic
        move_scores = [(move, env.estimate_move_value(move, current_player)) for move in moves]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        # Chọn nước đi tốt nhất với xác suất 80%, hoặc ngẫu nhiên với xác suất 20%
        if random.random() < 0.8:
            move = move_scores[0][0]
        else:
            move = random.choice(moves)
        env.make_move(move["from"], move["to"])
        current_player = "red" if current_player == "yellow" else "yellow"
        depth += 1
    return env