from app.ai.ataxx_env import AtaxxEnvironment
from typing import Dict, Any, Optional, Tuple
import hashlib

class FullMinimax:
    def __init__(self, board: AtaxxEnvironment, current_player: str, depth: int = 2):
        """
        Khởi tạo thuật toán Minimax với alpha-beta pruning cho Ataxx.

        Args:
            board: Môi trường AtaxxEnvironment.
            current_player: Người chơi hiện tại ('yellow' hoặc 'red').
            depth: Độ sâu tìm kiếm (mặc định 2).

        Raises:
            ValueError: Nếu board, current_player, hoặc depth không hợp lệ.
        """
        if not isinstance(board, AtaxxEnvironment):
            raise ValueError("Board must be an AtaxxEnvironment instance")
        if current_player not in ["yellow", "red"]:
            raise ValueError("Current player must be 'yellow' or 'red'")
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        
        self.env = board
        self.player = current_player
        self.depth = depth
        self.transposition_table: Dict[str, float] = {}  # Cache trạng thái

    def run(self) -> Optional[Dict[str, Any]]:
        """
        Chạy thuật toán Minimax để chọn nước đi tốt nhất.

        Returns:
            Optional[Dict[str, Any]]: Nước đi tốt nhất dạng {"from": {"row": r, "col": c}, "to": {"row": r, "col": c}},
                                     hoặc None nếu không có nước đi.
        
        Raises:
            RuntimeError: Nếu có lỗi khi xử lý môi trường hoặc tính toán.
        """
        try:
            moves = self.env.get_valid_moves()
            if not moves:
                return None
            
            # Sắp xếp nước đi để tối ưu alpha-beta pruning
            moves = self._sort_moves(moves)
            
            best_score = float('-inf')
            best_move = None
            alpha = float('-inf')
            beta = float('inf')
            
            for move in moves:
                clone = self.env.clone()
                clone.make_move(move["from"], move["to"])
                score = self.minimax(clone, self.depth - 1, False, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            
            return best_move
        
        except Exception as e:
            raise RuntimeError(f"Error in FullMinimax.run: {str(e)}")

    def minimax(self, env: AtaxxEnvironment, depth: int, maximizing: bool, alpha: float, beta: float) -> float:
        """
        Thuật toán Minimax với alpha-beta pruning.

        Args:
            env: Môi trường Ataxx hiện tại.
            depth: Độ sâu còn lại.
            maximizing: True nếu tối đa hóa (lượt của player), False nếu tối thiểu hóa.
            alpha: Giá trị alpha cho pruning.
            beta: Giá trị beta cho pruning.

        Returns:
            float: Điểm số của trạng thái.

        Raises:
            RuntimeError: Nếu có lỗi khi xử lý môi trường hoặc tính toán.
        """
        try:
            # Tạo key cho transposition table
            board_str = ''.join(''.join(row) for row in env.board)
            state_key = hashlib.md5(f"{board_str}{maximizing}{depth}".encode()).hexdigest()
            if state_key in self.transposition_table:
                return self.transposition_table[state_key]

            # Điều kiện dừng
            if depth == 0 or env.is_game_over():
                score = self._evaluate_state(env)
                self.transposition_table[state_key] = score
                return score

            moves = env.get_valid_moves()
            if not moves:
                # Không có nước đi, chuyển lượt và đánh giá lại
                clone = env.clone()
                clone.current_player = "red" if env.current_player == "yellow" else "yellow"
                score = self.minimax(clone, depth, not maximizing, alpha, beta)
                self.transposition_table[state_key] = score
                return score

            if maximizing:
                max_eval = float('-inf')
                moves = self._sort_moves(moves)
                for move in moves:
                    clone = env.clone()
                    clone.make_move(move["from"], move["to"])
                    eval_score = self.minimax(clone, depth - 1, False, alpha, beta)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
                self.transposition_table[state_key] = max_eval
                return max_eval
            else:
                min_eval = float('inf')
                moves = self._sort_moves(moves)
                for move in moves:
                    clone = env.clone()
                    clone.make_move(move["from"], move["to"])
                    eval_score = self.minimax(clone, depth - 1, True, alpha, beta)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
                self.transposition_table[state_key] = min_eval
                return min_eval
        
        except Exception as e:
            raise RuntimeError(f"Error in FullMinimax.minimax: {str(e)}")

    def _evaluate_state(self, env: AtaxxEnvironment) -> float:
        """
        Đánh giá trạng thái bàn cờ.

        Args:
            env: Môi trường Ataxx hiện tại.

        Returns:
            float: Điểm số, ưu tiên người chơi self.player.
        """
        try:
            scores = env.calculate_scores()
            if self.player == "yellow":
                base_score = scores["yellowScore"] - scores["redScore"]
            else:
                base_score = scores["redScore"] - scores["yellowScore"]
            return base_score + self.heuristic(env)
        
        except Exception as e:
            raise RuntimeError(f"Error in FullMinimax._evaluate_state: {str(e)}")

    def heuristic(self, env: AtaxxEnvironment) -> float:
        """
        Hàm heuristic đánh giá trạng thái bàn cờ, dựa trên heuristic_rollout.
        Ưu tiên số quân chiếm được, sao chép, và vị trí gần trung tâm.

        Args:
            env: Môi trường Ataxx hiện tại.

        Returns:
            float: Giá trị heuristic.
        """
        try:
            capture_potential = 0
            center_bonus = 0
            copy_bonus = 0
            center_row, center_col = env.board_size // 2, env.board_size // 2

            # Đánh giá từng quân của người chơi
            for row in range(env.board_size):
                for col in range(env.board_size):
                    if env.board[row][col] == self.player:
                        # Thưởng vị trí gần trung tâm
                        distance_to_center = abs(row - center_row) + abs(col - center_col)
                        center_bonus += (env.board_size - distance_to_center) * 0.5

                        # Đếm quân đối phương lân cận có thể chiếm
                        for dr, dc in env.neighbor_offsets:
                            nr, nc = row + dr, col + dc
                            if (env.is_valid_position(nr, nc) and
                                    env.board[nr][nc] not in [self.player, "empty", "block"]):
                                capture_potential += 1

            # Ưu tiên sao chép: Đếm số nước đi sao chép hợp lệ
            moves = env.get_valid_moves()
            for move in moves:
                from_row, from_col = move["from"]["row"], move["from"]["col"]
                to_row, to_col = move["to"]["row"], move["to"]["col"]
                distance = abs(to_row - from_row) + abs(to_col - from_col)
                if distance <= 1:  # Sao chép
                    copy_bonus += 5.0

            return capture_potential * 0.5 + center_bonus + copy_bonus * 0.1
        
        except Exception as e:
            raise RuntimeError(f"Error in FullMinimax.heuristic: {str(e)}")

    def _sort_moves(self, moves: list) -> list:
        """
        Sắp xếp nước đi để tối ưu alpha-beta pruning, ưu tiên sao chép và gần trung tâm.

        Args:
            moves: Danh sách nước đi.

        Returns:
            list: Danh sách nước đi đã sắp xếp.
        """
        def move_priority(move):
            from_row, from_col = move["from"]["row"], move["from"]["col"]
            to_row, to_col = move["to"]["row"], move["to"]["col"]
            distance = abs(to_row - from_row) + abs(to_col - from_col)
            is_copy = distance <= 1
            center = self.env.board_size // 2
            distance_to_center = abs(to_row - center) + abs(to_col - center)
            return (1 if is_copy else 0, -distance_to_center)
        
        return sorted(moves, key=move_priority, reverse=True)