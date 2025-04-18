from typing import List, Dict, Optional, Tuple

BOARD_SIZE = 7
MAX_MOVE_DISTANCE = 2
NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

class AtaxxEnvironment:
    def __init__(self, board: List[List[str]], current_player: str):
        self.board = board
        self.current_player = current_player
        self.size = len(board)
        self._valid_moves_cache: Optional[List[Dict[str, Dict[str, int]]]] = None
        self._game_over_cache: Optional[bool] = None

    def get_state(self) -> Tuple[Tuple[str, ...], ...]:
        return tuple(tuple(row) for row in self.board)

    def get_valid_moves(self) -> List[Dict[str, Dict[str, int]]]:
        if self._valid_moves_cache is not None:
            return self._valid_moves_cache

        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] != self.current_player:
                    continue
                # Chỉ kiểm tra các ô trong khoảng cách MAX_MOVE_DISTANCE
                min_row = max(0, row - MAX_MOVE_DISTANCE)
                max_row = min(self.size - 1, row + MAX_MOVE_DISTANCE) + 1
                min_col = max(0, col - MAX_MOVE_DISTANCE)
                max_col = min(self.size - 1, col + MAX_MOVE_DISTANCE) + 1

                for r in range(min_row, max_row):
                    for c in range(min_col, max_col):
                        to_pos = {"row": r, "col": c}
                        if self.is_valid_move({"row": row, "col": col}, to_pos):
                            moves.append({"from": {"row": row, "col": col}, "to": to_pos})

        self._valid_moves_cache = moves
        return moves

    def is_valid_move(self, from_pos: Dict[str, int], to_pos: Dict[str, int]) -> bool:
        from_row, from_col = from_pos["row"], from_pos["col"]
        to_row, to_col = to_pos["row"], to_pos["col"]

        if (self.board[from_row][from_col] != self.current_player or 
            self.board[to_row][to_col] != "empty"):
            return False

        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        return row_diff <= MAX_MOVE_DISTANCE and col_diff <= MAX_MOVE_DISTANCE

    def count_captured_pieces(self, to_pos: Dict[str, int]) -> int:
        to_row, to_col = to_pos["row"], to_pos["col"]
        captured = 0
        for dr, dc in NEIGHBOR_OFFSETS:
            r, c = to_row + dr, to_col + dc
            if (0 <= r < self.size and 0 <= c < self.size and 
                self.board[r][c] not in ["empty", "block", self.current_player]):
                captured += 1
        return captured

    def make_move(self, from_pos: Dict[str, int], to_pos: Dict[str, int]) -> None:
        from_row, from_col = from_pos["row"], from_pos["col"]
        to_row, to_col = to_pos["row"], to_pos["col"]

        new_board = [row[:] for row in self.board]
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        if row_diff <= 1 and col_diff <= 1:
            new_board[to_row][to_col] = self.current_player
        else:
            new_board[to_row][to_col] = self.current_player
            new_board[from_row][from_col] = "empty"

        for dr, dc in NEIGHBOR_OFFSETS:
            r, c = to_row + dr, to_col + dc
            if (0 <= r < self.size and 0 <= c < self.size and 
                new_board[r][c] not in ["empty", "block", self.current_player]):
                new_board[r][c] = self.current_player

        self.board = new_board
        self.current_player = "red" if self.current_player == "yellow" else "yellow"
        self._valid_moves_cache = None
        self._game_over_cache = None

    def calculate_scores(self) -> Dict[str, int]:
        yellow_score = sum(row.count("yellow") for row in self.board)
        red_score = sum(row.count("red") for row in self.board)
        return {"yellowScore": yellow_score, "redScore": red_score}

    def is_game_over(self) -> bool:
        if self._game_over_cache is not None:
            return self._game_over_cache

        moves = self.get_valid_moves()
        if not moves:
            self.current_player = "red" if self.current_player == "yellow" else "yellow"
            moves = self.get_valid_moves()
            self.current_player = "red" if self.current_player == "yellow" else "yellow"
            result = len(moves) == 0
            self._game_over_cache = result
            return result
        self._game_over_cache = False
        return False

    def get_reward(self, bot_player: str) -> float:
        if not self.is_game_over():
            return 0
        scores = self.calculate_scores()
        if scores["yellowScore"] > scores["redScore"]:
            return 1 if bot_player == "yellow" else -1
        elif scores["redScore"] > scores["yellowScore"]:
            return 1 if bot_player == "red" else -1
        return 0