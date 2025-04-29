from typing import List, Dict, Any
import copy

class AtaxxEnvironment:
    def __init__(self, board: List[List[str]], current_player: str):
        self.board_size = 7
        self.max_move_distance = 2
        self._valid_moves_cache = None  # Cache cho get_valid_moves()
        self.neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]

        if not self._is_valid_board(board):
            raise ValueError("Invalid board: Must be a 7x7 2D list with valid values ('yellow', 'red', 'empty', 'block')")

        self.board = [row[:] for row in board]
        self.current_player = current_player

    def _is_valid_board(self, board: List[List[str]]) -> bool:
        if not isinstance(board, list) or len(board) != self.board_size:
            print(f"Board has {len(board)} rows, expected {self.board_size}")
            return False
        for row in board:
            if not isinstance(row, list) or len(row) != self.board_size:
                print(f"Row has {len(row)} columns, expected {self.board_size}")
                return False
        valid_values = {"yellow", "red", "empty", "block"}
        for row in board:
            for cell in row:
                if cell not in valid_values:
                    print(f"Invalid cell value: {cell}, expected one of {valid_values}")
                    return False
        return True

    def clone(self) -> 'AtaxxEnvironment':
        return AtaxxEnvironment(copy.copy(self.board), self.current_player)

    def is_valid_position(self, row: int, col: int) -> bool:
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def is_valid_move(self, from_pos: Dict[str, int], to_pos: Dict[str, int]) -> bool:
        try:
            from_row = from_pos.get("row")
            from_col = from_pos.get("col")
            to_row = to_pos.get("row")
            to_col = to_pos.get("col")

            if not all(isinstance(x, int) for x in [from_row, from_col, to_row, to_col]):
                return False

            if not self.is_valid_position(from_row, from_col) or not self.is_valid_position(to_row, to_col):
                return False

            if (self.board[from_row][from_col] != self.current_player or
                    self.board[to_row][to_col] != "empty"):
                return False

            row_diff = abs(from_row - to_row)
            col_diff = abs(from_col - to_col)
            if row_diff > self.max_move_distance or col_diff > self.max_move_distance:
                return False

            return True
        except (TypeError, KeyError) as e:
            print(f"Invalid move format: from_pos={from_pos}, to_pos={to_pos}, error={str(e)}")
            return False

    def get_valid_moves(self) -> List[Dict[str, Any]]:
        # Nếu cache tồn tại và hợp lệ, trả về cache
        if self._valid_moves_cache is not None:
            return self._valid_moves_cache
        
        moves = []
        # Duyệt các ô có quân của người chơi hiện tại
        for from_row in range(self.board_size):
            for from_col in range(self.board_size):
                if self.board[from_row][from_col] != self.current_player:
                    continue
                # Chỉ kiểm tra các ô đích trong phạm vi max_move_distance
                for dr in range(-self.max_move_distance, self.max_move_distance + 1):
                    for dc in range(-self.max_move_distance, self.max_move_distance + 1):
                        to_row = from_row + dr
                        to_col = from_col + dc
                        if not self.is_valid_position(to_row, to_col):
                            continue
                        move = {
                            "from": {"row": from_row, "col": from_col},
                            "to": {"row": to_row, "col": to_col}
                        }
                        if self.is_valid_move(move["from"], move["to"]):
                            moves.append(move)

        # Lưu vào cache
        self._valid_moves_cache = moves
        return moves

    def has_valid_moves(self, player: str) -> bool:
        valid_moves = self.get_valid_moves()
        return len(valid_moves) > 0

    def make_move(self, from_pos: Dict[str, int], to_pos: Dict[str, int]) -> None:
        from_row = from_pos.get("row")
        from_col = from_pos.get("col")
        to_row = to_pos.get("row")
        to_col = to_pos.get("col")

        if not all(isinstance(x, int) for x in [from_row, from_col, to_row, to_col]):
            raise ValueError(f"Invalid move positions: from_pos={from_pos}, to_pos={to_pos}")

        row_diff = abs(from_row - to_row)
        col_diff = abs(from_col - to_col)

        if row_diff <= 1 and col_diff <= 1:
            self.board[to_row][to_col] = self.current_player
        else:
            self.board[from_row][from_col] = "empty"
            self.board[to_row][to_col] = self.current_player

        # Xóa cache sau khi thực hiện nước đi
        self._valid_moves_cache = None
        self.capture_neighbors(to_row, to_col)

    def capture_neighbors(self, row: int, col: int) -> None:
        for dr, dc in self.neighbor_offsets:
            nr, nc = row + dr, col + dc
            if (self.is_valid_position(nr, nc) and
                self.board[nr][nc] not in ["empty", "block", self.current_player]):
                self.board[nr][nc] = self.current_player

    def calculate_scores(self) -> Dict[str, int]:
        yellow_score = sum(row.count("yellow") for row in self.board)
        red_score = sum(row.count("red") for row in self.board)
        return {"yellowScore": yellow_score, "redScore": red_score}

    def is_board_full(self) -> bool:
        return all(cell != "empty" for row in self.board for cell in row)

    def is_game_over(self) -> bool:
        scores = self.calculate_scores()
        return (scores["yellowScore"] == 0 or scores["redScore"] == 0 or
                self.is_board_full() or
                (not self.has_valid_moves("yellow") and not self.has_valid_moves("red")))
    
    def get_cell(self, row, col):
        if 0 <= row < 7 and 0 <= col < 7:
            return self.board[row][col]
        else:
            return None  # Nếu ra ngoài phạm vi bảng, trả về None