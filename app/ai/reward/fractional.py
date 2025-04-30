from app.ai.ataxx_env import AtaxxEnvironment
from typing import Tuple

def fractional_reward(env: AtaxxEnvironment, bot_player: str) -> Tuple[float, bool]:
    """
    Trả về phần thưởng liên tục thuần túy dựa trên chênh lệch điểm số trong Ataxx.
    
    - Luôn trả về phần thưởng liên tục chuẩn hóa trong [0, 1] dựa trên hiệu số điểm.
    - Chuẩn hóa từ [-1, 1] sang [0, 1] như MctsAI.
    
    Args:
        env: Môi trường Ataxx hiện tại.
        bot_player: 'yellow' hoặc 'red'.
    
    Returns:
        Tuple (reward, is_win): reward trong [0, 1], is_win là True nếu có lợi thế.
    """
    if bot_player not in {"yellow", "red"}:
        raise ValueError("bot_player must be 'yellow' or 'red'")

    scores = env.calculate_scores()
    yellow_score = scores.get("yellowScore", 0)
    red_score = scores.get("redScore", 0)
    
    # Tính giá trị gốc trong [-1, 1]
    diff = yellow_score - red_score if bot_player == "yellow" else red_score - yellow_score
    value = max(min(diff / 49.0, 1.0), -1.0)  # Chuẩn hóa theo bàn 7x7
    
    # Chuẩn hóa sang [0, 1]
    reward = (value + 1.0) / 2.0
    
    # Xác định is_win
    is_win = value > 0.0
    
    return reward, is_win