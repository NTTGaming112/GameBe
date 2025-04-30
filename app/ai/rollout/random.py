from app.ai.ataxx_env import AtaxxEnvironment
import random

def random_rollout(env: AtaxxEnvironment, player: str, max_depth=100) -> AtaxxEnvironment:
    """
    Thực hiện mô phỏng ngẫu nhiên từ trạng thái hiện tại đến khi trò chơi kết thúc hoặc đạt độ sâu tối đa.
    
    Args:
        env: Môi trường Ataxx hiện tại.
        player: Người chơi hiện tại ('yellow' hoặc 'red').
    
    Returns:
        AtaxxEnvironment: Trạng thái bàn cờ cuối cùng sau mô phỏng.
    
    Raises:
        ValueError: Nếu môi trường hoặc người chơi không hợp lệ.
        RuntimeError: Nếu có lỗi trong quá trình mô phỏng.
    """
    if not isinstance(env, AtaxxEnvironment):
        raise ValueError("Invalid environment: must be AtaxxEnvironment")
    if player not in ["yellow", "red"]:
        raise ValueError("Invalid player: must be 'yellow' or 'red'")
    
    try:
        current_env = env.clone()  # Sao chép môi trường
        depth = 0
        while not current_env.is_game_over() and depth < max_depth:
            moves = current_env.get_valid_moves()
            if not moves:
                # Nếu không có nước đi, kết thúc lượt
                break
            # Chọn và thực hiện nước đi ngẫu nhiên
            move = random.choice(moves)
            current_env.make_move(move["from"], move["to"])  # make_move tự động chuyển lượt
            depth += 1
        return current_env
    
    except Exception as e:
        raise RuntimeError(f"Error in random_rollout: {str(e)}")