from app.ai.ataxx_env import AtaxxEnvironment
from app.ai.variants.full_minimax import FullMinimax

def minimax_rollout(env: AtaxxEnvironment, player: str, depth: int = 2, max_depth: int = 50) -> AtaxxEnvironment:
    """
    Thực hiện mô phỏng sử dụng thuật toán Minimax từ trạng thái hiện tại đến khi trò chơi kết thúc
    hoặc đạt độ sâu tối đa.
    
    Args:
        env: Môi trường Ataxx hiện tại.
        player: Người chơi hiện tại ('yellow' hoặc 'red').
        depth: Độ sâu của Minimax (mặc định 2).
        max_depth: Số nước đi tối đa trong mô phỏng (mặc định 50).
    
    Returns:
        AtaxxEnvironment: Trạng thái bàn cờ cuối cùng sau mô phỏng.
    
    Raises:
        ValueError: Nếu môi trường, người chơi, hoặc tham số không hợp lệ.
        RuntimeError: Nếu có lỗi trong quá trình mô phỏng.
    """
    if not isinstance(env, AtaxxEnvironment):
        raise ValueError("Invalid environment: must be AtaxxEnvironment")
    if player not in ["yellow", "red"]:
        raise ValueError("Invalid player: must be 'yellow' or 'red'")
    if depth < 1:
        raise ValueError("Depth must be at least 1")
    if max_depth < 1:
        raise ValueError("Max depth must be at least 1")
    
    try:
        current_env = env.clone()  # Sao chép môi trường
        rollout_depth = 0
        
        while not current_env.is_game_over() and rollout_depth < max_depth:
            moves = current_env.get_valid_moves()
            if not moves:
                # Nếu không có nước đi, chuyển lượt
                current_env.current_player = "red" if current_env.current_player == "yellow" else "yellow"
                rollout_depth += 1
                continue
            
            # Sử dụng FullMinimax để chọn nước đi
            minimax = FullMinimax(current_env, current_env.current_player, depth=depth)
            move = minimax.run()
            
            if not move or not isinstance(move, dict) or "from" not in move or "to" not in move:
                # Nếu Minimax không trả về nước đi hợp lệ, chuyển lượt
                current_env.current_player = "red" if current_env.current_player == "yellow" else "yellow"
                rollout_depth += 1
                continue
            
            # Thực hiện nước đi
            current_env.make_move(move["from"], move["to"])  # make_move tự động chuyển lượt
            rollout_depth += 1
        
        return current_env
    
    except Exception as e:
        raise RuntimeError(f"Error in minimax_rollout: {str(e)}")