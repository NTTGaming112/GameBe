import numpy as np
from constants import BOARD_SIZE, EMPTY, PLAYER_1, PLAYER_2

class AtaxxState:
    def __init__(self, initial_board=None):
        if initial_board is None:
            self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
            # Default starting positions
            self.board[0][0] = self.board[6][6] = PLAYER_1
            self.board[0][6] = self.board[6][0] = PLAYER_2
        else:
            self.board = initial_board.copy()
        self.current_player = PLAYER_1

    def copy(self):
        new_state = AtaxxState()
        new_state.board = self.board.copy()
        new_state.current_player = self.current_player
        return new_state

    def get_legal_moves(self):
        moves = []
        clone_destinations = set()  # Track unique (nr, nc) for clone moves

        # Step 1: Collect clone moves (1-step) and their destinations
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == self.current_player:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and 
                                self.board[nr][nc] == EMPTY and (nr, nc) not in clone_destinations):
                                moves.append((r, c, nr, nc))
                                clone_destinations.add((nr, nc))

        # Step 2: Collect jump moves (2-step) only for destinations not reachable by clone
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == self.current_player:
                    for dr in [-2, -1, 0, 1, 2]:
                        for dc in [-2, -1, 0, 1, 2]:
                            if abs(dr) <= 1 and abs(dc) <= 1:
                                continue  # Skip clone moves and (0,0)
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and 
                                self.board[nr][nc] == EMPTY and (nr, nc) not in clone_destinations):
                                moves.append((r, c, nr, nc))
                                clone_destinations.add((nr, nc))  # Prevent duplicates

        return moves

    def make_move(self, move):
        r, c, nr, nc = move
        is_clone = abs(r - nr) <= 1 and abs(c - nc) <= 1
        self.board[nr][nc] = self.current_player
        if not is_clone:
            self.board[r][c] = EMPTY
        # Capture opponent pieces
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nnr, nnc = nr + dr, nc + dc
                if 0 <= nnr < BOARD_SIZE and 0 <= nnc < BOARD_SIZE and self.board[nnr][nnc] == -self.current_player:
                    self.board[nnr][nnc] = self.current_player
        self.current_player = -self.current_player

    def is_game_over(self):
        return len(self.get_legal_moves()) == 0 or np.sum(self.board == EMPTY) == 0

    def get_winner(self):
        if not self.is_game_over():
            return 0
        own = np.sum(self.board == PLAYER_1)
        opp = np.sum(self.board == PLAYER_2)
        if own > opp:
            return PLAYER_1
        elif opp > own:
            return PLAYER_2
        return 0
    
    def get_empty_cells(self):
        return np.sum(self.board == EMPTY)

    def display_board(self):
        # Color codes
        COLORS = {
            'X': '\033[91m',  # Red
            'O': '\033[94m',  # Blue
            '#': '\033[90m',  # Gray (for blocked cells if added later)
            '.': '\033[0m',   # Reset (empty)
            'border': '\033[93m',  # Yellow
            'coord': '\033[92m',   # Green
            'reset': '\033[0m'
        }
        
        # Print column headers
        print(f"{COLORS['border']}  {COLORS['coord']}", end="")
        for c in range(BOARD_SIZE):
            print(f" {c} ", end="")
        print(COLORS['reset'])
        
        # Print each row
        for r in range(BOARD_SIZE):
            print(f"{COLORS['border']}{COLORS['coord']}{r} {COLORS['reset']}", end="")
            for c in range(BOARD_SIZE):
                cell = self.board[r][c]
                if cell == 1:
                    print(f"{COLORS['X']} X {COLORS['reset']}", end="")
                elif cell == -1:
                    print(f"{COLORS['O']} O {COLORS['reset']}", end="")
                elif cell == -2:  # For potential blocked cells
                    print(f"{COLORS['#']} # {COLORS['reset']}", end="")
                else:
                    print(f"{COLORS['.']} . {COLORS['reset']}", end="")
            print()
        
        # Print current player
        player_color = COLORS['X'] if self.current_player == PLAYER_1 else COLORS['O']
        print(f"Current player: {player_color}{'X' if self.current_player == PLAYER_1 else 'O'}{COLORS['reset']}")