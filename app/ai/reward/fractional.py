def fractional_reward(env, bot_player):
    scores = env.calculate_scores()
    total = scores["yellowScore"] + scores["redScore"]
    if total == 0: return 0
    if bot_player == "yellow":
        return scores["yellowScore"] / total
    return scores["redScore"] / total