from typing import List, Dict, Any
import numpy as np
from enum import IntEnum

class Piece(IntEnum):
    EMPTY = 0
    YELLOW = 1
    RED = 2
    BLOCK = 3

class AtaxxEnvironment:
    def __init__(self, board: List[List[str]], current_player: str):
        self.board_size = 7
        self.max_move_distance = 2
        self.neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        # Khởi tạo Zobrist table
        self.zobrist_table = np.random.randint(0, 2**64, (self.board_size, self.board_size, 4), dtype=np.uint64)
        self.zobrist_player = {Piece.YELLOW: np.random.randint(0, 2**64, dtype=np.uint64),
                              Piece.RED: np.random.randint(0, 2**64, dtype=np.uint64)}

        self.board = self._convert_board(board)
        self.player_map = {"yellow": Piece.YELLOW, "red": Piece.RED}
        if current_player not in self.player_map:
            raise ValueError("Invalid player: must be 'yellow' or 'red'")
        self.current_player = self.player_map[current_player]

    def _convert_board(self, board: List[List[str]]) -> np.ndarray:
        if not self._is_valid_board(board):
            raise ValueError("Invalid board")
        mapping = {"empty": Piece.EMPTY, "yellow": Piece.YELLOW, "red": Piece.RED, "block": Piece.BLOCK}
        return np.array([[mapping[cell] for cell in row] for row in board], dtype=np.int8)

    def _is_valid_board(self, board: List[List[str]]) -> bool:
        if not isinstance(board, list) or len(board) != self.board_size:
            return False
        for row in board:
            if not isinstance(row, list) or len(row) != self.board_size:
                return False
        valid_values = {"yellow", "red", "empty", "block"}
        return all(cell in valid_values for row in board for cell in row)

    def get_state_key(self) -> int:
        """Tạo định danh duy nhất bằng Zobrist hashing."""
        key = self.zobrist_player[self.current_player]
        for r in range(self.board_size):
            for c in range(self.board_size):
                piece = self.board[r, c]
                key ^= self.zobrist_table[r, c, piece]
        return key

    def clone(self) -> 'AtaxxEnvironment':
        new_env = AtaxxEnvironment.__new__(AtaxxEnvironment)
        new_env.board_size = self.board_size
        new_env.max_move_distance = self.max_move_distance
        new_env.neighbor_offsets = self.neighbor_offsets
        new_env.player_map = self.player_map
        new_env.zobrist_table = self.zobrist_table
        new_env.zobrist_player = self.zobrist_player
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        return new_env

    def is_valid_position(self, row: int, col: int) -> bool:
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def is_valid_move(self, from_pos: Dict[str, int], to_pos: Dict[str, int]) -> bool:
        try:
            from_row, from_col = from_pos["row"], from_pos["col"]
            to_row, to_col = to_pos["row"], to_pos["col"]
            if not (self.is_valid_position(from_row, from_col) and self.is_valid_position(to_row, to_col)):
                return False
            if (self.board[from_row, from_col] != self.current_player or
                self.board[to_row, to_col] != Piece.EMPTY):
                return False
            row_diff = abs(from_row - to_row)
            col_diff = abs(from_col - to_col)
            return row_diff <= self.max_move_distance and col_diff <= self.max_move_distance
        except (TypeError, KeyError):
            return False

    def get_valid_moves(self) -> List[Dict[str, Any]]:
        moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == self.current_player:
                    for dr in range(-self.max_move_distance, self.max_move_distance + 1):
                        for dc in range(-self.max_move_distance, self.max_move_distance + 1):
                            if dr == 0 and dc == 0:
                                continue
                            to_r, to_c = r + dr, c + dc
                            if 0 <= to_r < self.board_size and 0 <= to_c < self.board_size:
                                if self.board[to_r, to_c] == Piece.EMPTY:
                                    moves.append({
                                        "from": {"row": int(r), "col": int(c)},
                                        "to": {"row": int(to_r), "col": int(to_c)}
                                    })
        return moves

    def has_valid_moves(self, player: str) -> bool:
        if player not in self.player_map:
            return False
        original_player = self.current_player
        self.current_player = self.player_map[player]
        moves = self.get_valid_moves()
        self.current_player = original_player
        return len(moves) > 0

    def make_move(self, from_pos: Dict[str, int], to_pos: Dict[str, int]) -> None:
        from_row, from_col = from_pos["row"], from_pos["col"]
        to_row, to_col = to_pos["row"], to_pos["col"]
        if not self.is_valid_move(from_pos, to_pos):
            raise ValueError(f"Invalid move: from_pos={from_pos}, to_pos={to_pos}")
        row_diff = abs(from_row - to_row)
        col_diff = abs(from_col - to_col)
        is_jump = row_diff > 1 or col_diff > 1
        self.board[to_row, to_col] = self.current_player
        if is_jump:
            self.board[from_row, from_col] = Piece.EMPTY
        self.capture_neighbors(to_row, to_col)
        self.current_player = Piece.RED if self.current_player == Piece.YELLOW else Piece.YELLOW

    def capture_neighbors(self, row: int, col: int) -> None:
        opponent = Piece.RED if self.current_player == Piece.YELLOW else Piece.YELLOW
        for dr, dc in self.neighbor_offsets:
            nr, nc = row + dr, col + dc
            if self.is_valid_position(nr, nc) and self.board[nr, nc] == opponent:
                self.board[nr, nc] = self.current_player

    def calculate_scores(self) -> Dict[str, int]:
        yellow_score = np.sum(self.board == Piece.YELLOW)
        red_score = np.sum(self.board == Piece.RED)
        return {"yellowScore": int(yellow_score), "redScore": int(red_score)}

    def is_board_full(self) -> bool:
        return not np.any(self.board == Piece.EMPTY)

    def is_game_over(self) -> bool:
        scores = self.calculate_scores()
        if scores["yellowScore"] == 0 or scores["redScore"] == 0 or self.is_board_full():
            return True
        yellow_moves = self.has_valid_moves("yellow")
        red_moves = self.has_valid_moves("red")
        return not (yellow_moves or red_moves)

    def get_cell(self, row: int, col: int) -> Any:
        return self.board[row, col] if self.is_valid_position(row, col) else None