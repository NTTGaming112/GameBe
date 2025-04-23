from app.ai.ataxx_env import AtaxxEnvironment
from typing import Dict, Any

def heuristic_rollout(env: AtaxxEnvironment, player: str, max_depth: int = 50) -> AtaxxEnvironment:
    try:
        current_env = env.clone()
        current_player = player
        depth = 0
        
        while not current_env.is_game_over() and depth < max_depth:
            if not current_env.has_valid_moves(current_player):
                current_player = "red" if current_player == "yellow" else "yellow"
                current_env.current_player = current_player
                depth += 1
                continue
                
            moves = current_env.get_valid_moves()
            if not moves:
                current_player = "red" if current_player == "yellow" else "yellow"
                current_env.current_player = current_player
                depth += 1
                continue
                
            # Heuristic: Ưu tiên nước đi gần trung tâm bàn cờ hoặc có khả năng chiếm nhiều ô
            best_move = None
            best_score = float('-inf')
            center_row, center_col = 3, 3  # Trung tâm của bàn cờ 7x7
            
            scores = current_env.calculate_scores()
            for move in moves:
                to_row, to_col = move["to"]["row"], move["to"]["col"]
                # Tính khoảng cách đến trung tâm
                distance_to_center = abs(to_row - center_row) + abs(to_col - center_col)
                # Ước lượng số ô có thể chiếm
                temp_env = current_env.clone()
                temp_env.make_move(move["from"], move["to"])
                new_scores = temp_env.calculate_scores()
                score_gain = (new_scores["yellowScore"] if current_player == "yellow" else new_scores["redScore"]) - \
                            (scores["yellowScore"] if current_player == "yellow" else scores["redScore"])
                # Kết hợp khoảng cách và số ô chiếm được
                heuristic_score = score_gain - distance_to_center * 0.5
                
                if heuristic_score > best_score:
                    best_score = heuristic_score
                    best_move = move
            
            if best_move:
                current_env.make_move(best_move["from"], best_move["to"])
            else:
                current_player = "red" if current_player == "yellow" else "yellow"
                current_env.current_player = current_player
                depth += 1
                continue
                
            current_player = "red" if current_player == "yellow" else "yellow"
            current_env.current_player = current_player
            depth += 1
        
        return current_env
    except Exception as e:
        raise Exception(f"Error in heuristic_rollout: {str(e)}")