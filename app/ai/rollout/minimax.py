from app.ai.ataxx_env import AtaxxEnvironment

def minimax_rollout(env: AtaxxEnvironment, player: str, depth: int = 2) -> AtaxxEnvironment:
    """
    Thực hiện mô phỏng sử dụng thuật toán Minimax từ trạng thái hiện tại đến khi trò chơi kết thúc
    hoặc đạt độ sâu tối đa.
    
    Args:
        env: Môi trường Ataxx hiện tại.
        player: Người chơi hiện tại ('yellow' hoặc 'red').
        depth: Độ sâu của Minimax (mặc định 2).
    
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
    
    try:
        current_env = env.clone()  # Sao chép môi trường
        current_player = player  # Người chơi hiện tại
        
        
        while not current_env.is_game_over():
            # Lấy các nước đi hợp lệ
            moves = current_env.get_valid_moves()
            if not moves:
                # Nếu không có nước đi, chuyển lượt
                current_env.current_player = "red" if current_env.current_player == "yellow" else "yellow"
                continue
            
            # Tạo Minimax để đánh giá các nước đi
            best_move = None
            best_score = -float('inf')
            
            for move in moves:
                # Thực hiện mô phỏng nước đi
                next_env = current_env.clone()
                next_env.make_move(move["from"], move["to"])  # make_move tự động chuyển lượt
                
                # Đánh giá trạng thái sau nước đi
                score = minimax_evaluation(next_env, current_player, depth - 1)
                
                # Cập nhật nước đi tốt nhất
                if score > best_score:
                    best_score = score
                    best_move = move
            
            # Thực hiện nước đi tốt nhất
            if best_move:
                current_env.make_move(best_move["from"], best_move["to"])  # make_move tự động chuyển lượt
            
        return current_env
    
    except Exception as e:
        raise RuntimeError(f"Error in minimax_rollout: {str(e)}")

def minimax_evaluation(env: AtaxxEnvironment, player: str, depth: int) -> int:
    """
    Đánh giá trạng thái của bàn cờ sau một số nước đi sử dụng thuật toán Minimax.

    Args:
        env: Môi trường Ataxx hiện tại.
        player: Người chơi hiện tại ('yellow' hoặc 'red').
        depth: Độ sâu của thuật toán Minimax.

    Returns:
        int: Điểm số của trạng thái bàn cờ.
    """
    if depth == 0 or env.is_game_over():
        # Trả về điểm số hiện tại khi đạt độ sâu tối đa hoặc trò chơi kết thúc
        return evaluate_board(env, player)

    # Tiến hành phân nhánh
    valid_moves = env.get_valid_moves()
    best_score = -float('inf')
    for move in valid_moves:
        next_env = env.clone()
        next_env.make_move(move["from"], move["to"])
        score = minimax_evaluation(next_env, player, depth - 1)
        best_score = max(best_score, score)
    
    return best_score

def evaluate_board(env: AtaxxEnvironment, player: str) -> int:
    """
    Đánh giá bàn cờ hiện tại dựa trên số lượng quân cờ của người chơi.
    Trả về điểm số cho người chơi hiện tại.

    Args:
        env: Môi trường Ataxx hiện tại.
        player: Người chơi hiện tại ('yellow' hoặc 'red').

    Returns:
        int: Điểm số của người chơi.
    """
    yellow_score = sum(row.count('yellow') for row in env.board)
    red_score = sum(row.count('red') for row in env.board)

    if player == "yellow":
        return yellow_score - red_score
    else:
        return red_score - yellow_score
