from typing import List, Dict, Any
import copy

class AtaxxEnvironment:
    def __init__(self, board: List[List[str]], current_player: str):
        self.board = board
        self.current_player = current_player
        self.board_size = 7
        self.max_move_distance = 2
        self.neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        self.last_move = None
        self.valid_moves_cache = {}  # Cache nước đi hợp lệ: {player: moves}

    def is_valid_position(self, row: int, col: int) -> bool:
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def is_valid_move(self, from_pos: Dict[str, int], to_pos: Dict[str, int]) -> bool:
        from_row, from_col = from_pos["row"], from_pos["col"]
        to_row, to_col = to_pos["row"], to_pos["col"]

        if (self.board[from_row][from_col] != self.current_player or
                self.board[to_row][to_col] != "empty"):
            return False

        row_diff = abs(from_row - to_row)
        col_diff = abs(from_col - to_col)
        if row_diff > self.max_move_distance or col_diff > self.max_move_distance:
            return False

        return self.is_valid_position(to_row, to_col)

    def get_valid_moves(self) -> List[Dict[str, Any]]:
        # Kiểm tra cache
        if self.current_player in self.valid_moves_cache:
            return self.valid_moves_cache[self.current_player]

        valid_moves = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == self.current_player:
                    from_pos = {"row": row, "col": col}
                    for dr in range(-self.max_move_distance, self.max_move_distance + 1):
                        for dc in range(-self.max_move_distance, self.max_move_distance + 1):
                            if dr == 0 and dc == 0:
                                continue
                            to_row, to_col = row + dr, col + dc
                            to_pos = {"row": to_row, "col": to_col}
                            if self.is_valid_move(from_pos, to_pos):
                                valid_moves.append({"from": from_pos, "to": to_pos})
        
        # Lưu vào cache
        self.valid_moves_cache[self.current_player] = valid_moves
        return valid_moves

    def has_valid_moves(self, player: str) -> bool:
        original_player = self.current_player
        self.current_player = player
        valid_moves = self.get_valid_moves()
        self.current_player = original_player
        return len(valid_moves) > 0

    def capture_neighbors(self, to_pos: Dict[str, int]) -> None:
        to_row, to_col = to_pos["row"], to_pos["col"]
        opponent = "red" if self.current_player == "yellow" else "yellow"
        for dr, dc in self.neighbor_offsets:
            nr, nc = to_row + dr, to_col + dc
            if (self.is_valid_position(nr, nc) and self.board[nr][nc] == opponent):
                self.board[nr][nc] = self.current_player

    def make_move(self, from_pos: Dict[str, int], to_pos: Dict[str, int]) -> None:
        from_row, from_col = from_pos["row"], from_pos["col"]
        to_row, to_col = to_pos["row"], to_pos["col"]

        row_diff = abs(from_row - to_row)
        col_diff = abs(from_col - to_col)

        if row_diff <= 1 and col_diff <= 1:
            self.board[to_row][to_col] = self.current_player
        else:
            self.board[from_row][from_col] = "empty"
            self.board[to_row][to_col] = self.current_player

        self.capture_neighbors(to_pos)
        self.last_move = {"from": from_pos, "to": to_pos}
        # Xóa cache sau khi thực hiện nước đi
        self.valid_moves_cache.clear()

    def calculate_scores(self) -> Dict[str, int]:
        yellow_score = 0
        red_score = 0
        for row in self.board:
            for cell in row:
                if cell == "yellow":
                    yellow_score += 1
                elif cell == "red":
                    red_score += 1
        return {"yellowScore": yellow_score, "redScore": red_score}

    def is_board_full(self) -> bool:
        return all(cell != "empty" for row in self.board for cell in row)

    def is_game_over(self) -> bool:
        scores = self.calculate_scores()
        return (scores["yellowScore"] == 0 or scores["redScore"] == 0 or
                self.is_board_full() or
                (not self.has_valid_moves("yellow") and not self.has_valid_moves("red")))

    def clone(self):
        cloned_board = [row[:] for row in self.board]
        cloned_env = AtaxxEnvironment(cloned_board, self.current_player)
        cloned_env.last_move = copy.deepcopy(self.last_move)
        return cloned_env

    def estimate_move_value(self, move: Dict[str, Any], player: str) -> float:
        clone = self.clone()
        original_player = clone.current_player
        clone.current_player = player
        clone.make_move(move["from"], move["to"])
        scores = clone.calculate_scores()
        
        score_diff = (scores["yellowScore"] - scores["redScore"] if player == "yellow"
                      else scores["redScore"] - scores["yellowScore"])
        
        captures = 0
        to_row, to_col = move["to"]["row"], move["to"]["col"]
        for dr, dc in self.neighbor_offsets:
            nr, nc = to_row + dr, to_col + dc
            if (self.is_valid_position(nr, nc) and
                    self.board[nr][nc] in ["yellow", "red"] and
                    self.board[nr][nc] != player):
                captures += 1
        
        center = self.board_size // 2
        distance_to_center = abs(to_row - center) + abs(to_col - center)
        center_bonus = (self.board_size - distance_to_center) * 0.05
        
        return score_diff + captures * 0.1 + center_bonus