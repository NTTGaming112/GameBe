from app.ai.ataxx_env import AtaxxEnvironment
from typing import Union

def fractional_reward(env: AtaxxEnvironment, bot_player: str, align_with_binary: bool = False) -> Union[int, float]:
    """
    Trả về phần thưởng dạng fractional hoặc binary-like dựa trên chênh lệch điểm số trong Ataxx.
    
    Args:
        env: Môi trường Ataxx hiện tại.
        bot_player: 'yellow' hoặc 'red'.
        align_with_binary: Nếu True và game over → trả về 1 / -1 / 0 như binary reward.
    
    Returns:
        float ∈ [-1, 1] nếu align_with_binary=False, hoặc int nếu align_with_binary=True và game kết thúc.
    """
    if bot_player not in {"yellow", "red"}:
        raise ValueError("bot_player must be 'yellow' or 'red'")

    scores = env.calculate_scores()
    yellow = scores.get("yellowScore", 0)
    red = scores.get("redScore", 0)

    if align_with_binary and env.is_game_over():
        if yellow > red:
            return 1 if bot_player == "yellow" else -1
        elif red > yellow:
            return 1 if bot_player == "red" else -1
        return 0

    diff = (yellow - red) if bot_player == "yellow" else (red - yellow)
    return max(min(diff / 49.0, 1.0), -1.0)
