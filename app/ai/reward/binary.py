def binary_reward(env, bot_player):
    if not env.is_game_over(): return 0
    scores = env.calculate_scores()
    if scores["yellowScore"] > scores["redScore"]:
        return 1 if bot_player == "yellow" else -1
    elif scores["redScore"] > scores["yellowScore"]:
        return 1 if bot_player == "red" else -1
    return 0