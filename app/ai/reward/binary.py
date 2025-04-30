from app.ai.ataxx_env import AtaxxEnvironment
from typing import Tuple

def binary_reward(env: AtaxxEnvironment, bot_player: str) -> Tuple[float, bool]:
    """
    Trả về phần thưởng nhị phân tối ưu hóa cho bot, lấy cảm hứng từ MctsAI.
    
    - Nếu trò chơi kết thúc:
        + Trả về reward trong [0, 1] (1.0: thắng, 0.0: thua, 0.5: hòa).
    - Nếu chưa kết thúc:
        + Trả về 0.5 (trạng thái trung lập).
    - Chuẩn hóa từ [-1, 1] sang [0, 1] như MctsAI.
    
    Args:
        env: Môi trường Ataxx hiện tại.
        bot_player: 'yellow' hoặc 'red'.
    
    Returns:
        Tuple (reward, is_win): reward trong [0, 1], is_win là True nếu thắng.
    """
    if bot_player not in {"yellow", "red"}:
        raise ValueError("bot_player must be 'yellow' or 'red'")

    scores = env.calculate_scores()
    yellow_score = scores["yellowScore"]
    red_score = scores["redScore"]
    
    # Tính giá trị gốc trong [-1, 1]
    if not env.is_game_over():
        value = 0.0  # Trung lập khi chưa kết thúc
    else:
        if yellow_score > red_score:
            value = 1.0 if bot_player == "yellow" else -1.0
        elif red_score > yellow_score:
            value = 1.0 if bot_player == "red" else -1.0
        else:
            value = 0.0
    
    # Chuẩn hóa sang [0, 1]
    reward = (value + 1.0) / 2.0
    
    # Xác định is_win
    is_win = value > 0.0
    
    return reward, is_win