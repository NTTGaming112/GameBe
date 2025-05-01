import copy
import random
import hashlib
from typing import List, Tuple, Dict, Optional, Set

class AtaxxState:
    """
    Represents a state of the Ataxx game.
    
    Board notation:
    - "empty": Empty cell
    - "red": Red player's piece
    - "yellow": Yellow player's piece
    
    The board is a 7x7 grid.
    """
    
    BOARD_SIZE = 7
    PLAYERS = ["red", "yellow"]
    
    # Directions for adjacent and jump moves
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    def __init__(self, board=None, current_player="red"):
        """Initialize the game state with a board and current player."""
        if board is None:
            # Default initial board setup
            self.board = [["empty" for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
            self.board[0][0] = "red"
            self.board[0][6] = "yellow"
            self.board[6][0] = "yellow"
            self.board[6][6] = "red"
        else:
            self.board = board
            
        self.current_player = current_player
        self._cached_legal_moves = None
    
    def clone(self):
        """Create a deep copy of the current state."""
        new_state = AtaxxState(copy.deepcopy(self.board), self.current_player)
        return new_state
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is within the board boundaries."""
        return 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE
    
    def get_legal_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get all legal moves for the current player.
        
        Returns:
            List of tuples (from_pos, to_pos), where each position is a (row, col) tuple.
        """
        if self._cached_legal_moves is not None:
            return self._cached_legal_moves
            
        legal_moves = []
        
        # Find all pieces of the current player
        player_pieces = []
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.board[row][col] == self.current_player:
                    player_pieces.append((row, col))
        
        # For each piece, find all possible moves
        for from_row, from_col in player_pieces:
            # Adjacent moves (distance 1)
            for dr, dc in self.DIRECTIONS:
                to_row, to_col = from_row + dr, from_col + dc
                if self.is_valid_position(to_row, to_col) and self.board[to_row][to_col] == "empty":
                    legal_moves.append(((from_row, from_col), (to_row, to_col)))
            
            # Jump moves (distance 2)
            for dr in [-2, -1, 0, 1, 2]:
                for dc in [-2, -1, 0, 1, 2]:
                    if dr == 0 and dc == 0:
                        continue
                    if abs(dr) <= 1 and abs(dc) <= 1:
                        continue  # Skip adjacent moves (already covered)
                    
                    to_row, to_col = from_row + dr, from_col + dc
                    if self.is_valid_position(to_row, to_col) and self.board[to_row][to_col] == "empty":
                        legal_moves.append(((from_row, from_col), (to_row, to_col)))
        
        self._cached_legal_moves = legal_moves
        return legal_moves
    
    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> None:
        """
        Make a move on the board and update the state.
        
        Args:
            from_pos: (row, col) of the source position
            to_pos: (row, col) of the destination position
        """
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # Check if move is legal
        if (from_pos, to_pos) not in self.get_legal_moves():
            raise ValueError("Illegal move")
        
        # Calculate move distance to determine if it's a jump or adjacent move
        distance = max(abs(to_row - from_row), abs(to_col - from_col))
        
        # Place a piece at the destination
        self.board[to_row][to_col] = self.current_player
        
        # If it's a jump move, the source position remains
        # If it's an adjacent move, the source position remains and a new piece is added
        if distance > 1:  # Jump move
            self.board[from_row][from_col] = "empty"
        
        # Capture adjacent opponent pieces
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                adj_row, adj_col = to_row + dr, to_col + dc
                if self.is_valid_position(adj_row, adj_col):
                    opponent = "yellow" if self.current_player == "red" else "red"
                    if self.board[adj_row][adj_col] == opponent:
                        self.board[adj_row][adj_col] = self.current_player
        
        # Switch player
        self.current_player = "yellow" if self.current_player == "red" else "red"
        
        # Clear cached legal moves
        self._cached_legal_moves = None
    
    def get_winner(self) -> Optional[str]:
        """
        Check if the game is over and return the winner.
        
        Returns:
            "red", "yellow", "draw", or None if the game is not over.
        """
        red_count = 0
        yellow_count = 0
        empty_count = 0
        
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.board[row][col] == "red":
                    red_count += 1
                elif self.board[row][col] == "yellow":
                    yellow_count += 1
                else:
                    empty_count += 1
        
        # If one player has no pieces, the other player wins
        if red_count == 0:
            return "yellow"
        if yellow_count == 0:
            return "red"
        
        # If the board is full, the player with more pieces wins
        if empty_count == 0:
            if red_count > yellow_count:
                return "red"
            elif yellow_count > red_count:
                return "yellow"
            else:
                return "draw"
        
        # If the current player has no legal moves, the other player wins
        if not self.get_legal_moves():
            return "yellow" if self.current_player == "red" else "red"
        
        # Game is not over
        return None
    
    def is_terminal(self) -> bool:
        """Check if the game is over."""
        return self.get_winner() is not None
    
    def get_pieces_count(self) -> Dict[str, int]:
        """Count the number of pieces for each player."""
        counts = {"red": 0, "yellow": 0}
        
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.board[row][col] in counts:
                    counts[self.board[row][col]] += 1
        
        return counts
    
    def get_result(self, player: str) -> float:
        """
        Get the result of the game from the perspective of the specified player.
        
        Args:
            player: "red" or "yellow"
            
        Returns:
            1.0 if player OX, 0.0 if player loses, 0.5 if draw, None if game is not over.
        """
        winner = self.get_winner()
        
        if winner is None:
            return None
        elif winner == player:
            return 1.0
        elif winner == "draw":
            return 0.5
        else:
            return 0.0
        
    def get_result_fractional(self, player: str) -> float:
        counts = self.get_pieces_count()
        opponent = "yellow" if player == "red" else "red"
        diff = counts[player] - counts[opponent]
        max_diff = self.BOARD_SIZE * self.BOARD_SIZE
        return (diff + max_diff) / (2 * max_diff)
    
    def get_random_move(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Get a random legal move."""
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            return None
        return random.choice(legal_moves)
    
    def get_board_representation(self) -> List[List[str]]:
        """Return a copy of the current board."""
        return copy.deepcopy(self.board)
    
    def to_dict(self) -> Dict:
        """Convert the state to a dictionary for JSON serialization."""
        return {
            "board": copy.deepcopy(self.board),
            "current_player": self.current_player
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AtaxxState':
        """Create a state from a dictionary."""
        return cls(board=data["board"], current_player=data["current_player"])
    
    def hash(self) -> str:
        """Generate a unique hash for the state for transposition table."""
        board_str = ''.join(''.join(row) for row in self.board)
        state_str = f"{board_str}:{self.current_player}"
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def __str__(self) -> str:
        """String representation of the state."""
        result = f"Current player: {self.current_player}\n"
        for row in self.board:
            result += " ".join([cell[0] if cell != "empty" else "." for cell in row]) + "\n"
        return result