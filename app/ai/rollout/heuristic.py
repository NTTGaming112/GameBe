from app.ai.ataxx_env import AtaxxEnvironment

def heuristic_rollout(env: AtaxxEnvironment, player: str, max_depth: int = 50) -> AtaxxEnvironment:
    """
    Thực hiện mô phỏng sử dụng heuristic, ưu tiên sao chép (di chuyển lân cận) hơn nhảy,
    đến khi trò chơi kết thúc hoặc đạt độ sâu tối đa.
    
    Args:
        env: Môi trường Ataxx hiện tại.
        player: Người chơi hiện tại ('yellow' hoặc 'red').
        max_depth: Số nước đi tối đa trong mô phỏng (mặc định 50).
    
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
    if max_depth < 1:
        raise ValueError("Max depth must be at least 1")
    
    try:
        current_env = env.clone()  # Sao chép môi trường
        depth = 0
        
        while not current_env.is_game_over() and depth < max_depth:
            moves = current_env.get_valid_moves()
            if not moves:
                # Nếu không có nước đi, chuyển lượt
                current_env.current_player = "red" if current_env.current_player == "yellow" else "yellow"
                depth += 1
                continue
            
            # Heuristic: Ưu tiên sao chép (lân cận) và chiếm nhiều ô
            best_move = None
            best_score = float('-inf')
            center_row, center_col = 3, 3  # Trung tâm của bàn cờ 7x7
            scores = current_env.calculate_scores()
            
            for move in moves:
                from_row, from_col = move["from"]["row"], move["from"]["col"]
                to_row, to_col = move["to"]["row"], move["to"]["col"]
                
                # Xác định loại nước đi: sao chép (lân cận) hay nhảy
                distance = abs(to_row - from_row) + abs(to_col - from_col)
                is_copy = distance <= 1  # Sao chép nếu khoảng cách Manhattan <= 1
                
                # Tính khoảng cách đến trung tâm
                distance_to_center = abs(to_row - center_row) + abs(to_col - center_col)
                
                # Ước lượng số ô chiếm được
                temp_env = current_env.clone()
                temp_env.make_move(move["from"], move["to"])
                new_scores = temp_env.calculate_scores()
                score_gain = (new_scores["yellowScore"] if current_env.current_player == "yellow" else new_scores["redScore"]) - \
                             (scores["yellowScore"] if current_env.current_player == "yellow" else scores["redScore"])
                
                # Heuristic: Ưu tiên sao chép, chiếm ô, và gần trung tâm
                heuristic_score = score_gain + (5.0 if is_copy else 0.0) - distance_to_center * 0.5
                
                if heuristic_score > best_score:
                    best_score = heuristic_score
                    best_move = move
            
            if best_move:
                current_env.make_move(best_move["from"], best_move["to"])  # make_move tự động chuyển lượt
            else:
                current_env.current_player = "red" if current_env.current_player == "yellow" else "yellow"
                depth += 1
                continue
            
            depth += 1
        
        return current_env
    
    except Exception as e:
        raise RuntimeError(f"Error in heuristic_rollout: {str(e)}")