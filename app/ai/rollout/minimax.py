def minimax_rollout(env, bot_player, max_depth=20, minimax_depth=2):  # Giảm minimax_depth
    depth = 0
    current_player = bot_player
    while not env.is_game_over() and depth < max_depth:
        if not env.has_valid_moves(current_player):
            current_player = "red" if current_player == "yellow" else "yellow"
            continue
        moves = env.get_valid_moves()
        if not moves:
            break
        best_score = float('-inf')
        best_move = None
        for move in moves:
            clone = env.clone()
            clone.make_move(move["from"], move["to"])
            score = minimax(clone, current_player, minimax_depth - 1, False)
            if score > best_score:
                best_score = score
                best_move = move
        if best_move:
            env.make_move(best_move["from"], best_move["to"])
        current_player = "red" if current_player == "yellow" else "yellow"
        depth += 1
    return env

def minimax(env, player, depth, maximizing):
    if depth == 0 or env.is_game_over():
        scores = env.calculate_scores()
        if player == "yellow":
            return scores["yellowScore"] - scores["redScore"]
        return scores["redScore"] - scores["yellowScore"]
    if maximizing:
        max_eval = float('-inf')
        for move in env.get_valid_moves():
            clone = env.clone()
            clone.make_move(move["from"], move["to"])
            eval = minimax(clone, player, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in env.get_valid_moves():
            clone = env.clone()
            clone.make_move(move["from"], move["to"])
            eval = minimax(clone, player, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval