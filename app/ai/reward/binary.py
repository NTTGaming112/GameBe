from app.ai.ataxx_env import AtaxxEnvironment
from typing import Union

def binary_reward(env: AtaxxEnvironment, bot_player: str, use_fractional_if_ongoing: bool = False) -> Union[int, float]:
    """
    Tính giá trị thưởng binary dựa trên kết quả trò chơi Ataxx.
    
    Args:
        env: Môi trường Ataxx hiện tại.
        bot_player: Người chơi bot ('yellow' hoặc 'red').
        use_fractional_if_ongoing: Nếu True, trả về thưởng fractional dựa trên chênh lệch điểm
                                  khi trò chơi chưa kết thúc (mặc định False).
    
    Returns:
        Union[int, float]: Giá trị thưởng:
            - 1: Bot thắng.
            - -1: Bot thua.
            - 0: Hòa hoặc trò chơi chưa kết thúc (nếu use_fractional_if_ongoing=False).
            - Float: Chênh lệch điểm chuẩn hóa nếu trò chơi chưa kết thúc và use_fractional_if_ongoing=True.
    
    Raises:
        ValueError: Nếu env hoặc bot_player không hợp lệ.
        RuntimeError: Nếu có lỗi khi tính toán điểm hoặc kiểm tra trạng thái trò chơi.
    """
    if not isinstance(env, AtaxxEnvironment):
        raise ValueError("Invalid environment: must be AtaxxEnvironment")
    if bot_player not in ["yellow", "red"]:
        raise ValueError("Invalid bot_player: must be 'yellow' or 'red'")
    
    try:
        # Kiểm tra trạng thái trò chơi
        if not env.is_game_over():
            if use_fractional_if_ongoing:
                scores = env.calculate_scores()
                if not isinstance(scores, dict) or "yellowScore" not in scores or "redScore" not in scores:
                    raise RuntimeError("Invalid scores format from calculate_scores")
                # Tính chênh lệch điểm chuẩn hóa (giới hạn trong [-1, 1])
                score_diff = (scores["yellowScore"] - scores["redScore"]) if bot_player == "yellow" else \
                             (scores["redScore"] - scores["yellowScore"])
                total_squares = 49  # Bàn cờ 7x7
                return max(min(score_diff / total_squares, 1.0), -1.0)
            return 0
        
        # Trò chơi đã kết thúc, tính điểm
        scores = env.calculate_scores()
        if not isinstance(scores, dict) or "yellowScore" not in scores or "redScore" not in scores:
            raise RuntimeError("Invalid scores format from calculate_scores")
        
        yellow_score = scores["yellowScore"]
        red_score = scores["redScore"]
        
        if yellow_score > red_score:
            return 1 if bot_player == "yellow" else -1
        elif red_score > yellow_score:
            return 1 if bot_player == "red" else -1
        return 0
    
    except Exception as e:
        raise RuntimeError(f"Error in binary_reward: {str(e)}")