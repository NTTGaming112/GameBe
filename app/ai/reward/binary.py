from app.ai.ataxx_env import AtaxxEnvironment
from typing import Union

def binary_reward(env: AtaxxEnvironment, bot_player: str, use_fractional_if_ongoing: bool = False) -> Union[int, float]:
    """
    Trả về phần thưởng cho bot dựa trên kết quả trò chơi.
    
    - Nếu trò chơi kết thúc:
        + Trả về 1 nếu bot thắng, -1 nếu thua, 0 nếu hoà.
    - Nếu chưa kết thúc:
        + Trả về 0 (mặc định) hoặc fractional reward nếu `use_fractional_if_ongoing=True`.
    """
    if bot_player not in {"yellow", "red"}:
        raise ValueError("bot_player must be 'yellow' or 'red'")

    if not env.is_game_over():
        if use_fractional_if_ongoing:
            scores = env.calculate_scores()
            bot_score = scores[f"{bot_player}Score"]
            opp_score = scores["redScore"] if bot_player == "yellow" else scores["yellowScore"]
            score_diff = bot_score - opp_score
            return max(min(score_diff / 49.0, 1.0), -1.0)
        return 0

    scores = env.calculate_scores()
    yellow, red = scores["yellowScore"], scores["redScore"]

    if yellow > red:
        return 1 if bot_player == "yellow" else -1
    elif red > yellow:
        return 1 if bot_player == "red" else -1
    else:
        return 0
