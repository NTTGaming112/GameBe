from app.ai.ataxx_env import AtaxxEnvironment
from typing import Union

def fractional_reward(env: AtaxxEnvironment, bot_player: str, align_with_binary: bool = False) -> Union[int, float]:
    """
    Tính giá trị thưởng fractional dựa trên tỷ lệ hoặc chênh lệch quân trong Ataxx.
    
    Args:
        env: Môi trường Ataxx hiện tại.
        bot_player: Người chơi bot ('yellow' hoặc 'red').
        align_with_binary: Nếu True, trả về giá trị gần với binary_reward khi trò chơi kết thúc
                          (1, -1, 0). Nếu False, trả về giá trị fractional trong [-1, 1].
    
    Returns:
        Union[int, float]: Giá trị thưởng:
            - Nếu align_with_binary=True và trò chơi kết thúc: 1 (bot thắng), -1 (bot thua), 0 (hòa).
            - Nếu align_with_binary=False hoặc trò chơi chưa kết thúc: Giá trị trong [-1, 1]
              dựa trên chênh lệch quân chuẩn hóa.
    
    Raises:
        ValueError: Nếu env hoặc bot_player không hợp lệ.
        RuntimeError: Nếu có lỗi khi tính toán điểm.
    """
    if not isinstance(env, AtaxxEnvironment):
        raise ValueError("Invalid environment: must be AtaxxEnvironment")
    if bot_player not in ["yellow", "red"]:
        raise ValueError("Invalid bot_player: must be 'yellow' or 'red'")
    
    try:
        # Lấy điểm số
        scores = env.calculate_scores()
        if not isinstance(scores, dict) or "yellowScore" not in scores or "redScore" not in scores:
            raise RuntimeError("Invalid scores format from calculate_scores")
        
        yellow_score = scores["yellowScore"]
        red_score = scores["redScore"]
        total = yellow_score + red_score
        
        # Xử lý trò chơi kết thúc
        if env.is_game_over() and align_with_binary:
            if yellow_score > red_score:
                return 1 if bot_player == "yellow" else -1
            elif red_score > yellow_score:
                return 1 if bot_player == "red" else -1
            return 0
        
        # Tính thưởng fractional dựa trên chênh lệch điểm
        if total == 0:
            return 0.0  # Trường hợp hiếm khi không có quân
        
        # Chênh lệch điểm chuẩn hóa trong [-1, 1]
        score_diff = (yellow_score - red_score) if bot_player == "yellow" else (red_score - yellow_score)
        total_squares = 49  # Bàn cờ 7x7
        return max(min(score_diff / total_squares, 1.0), -1.0)
    
    except Exception as e:
        raise RuntimeError(f"Error in fractional_reward: {str(e)}")