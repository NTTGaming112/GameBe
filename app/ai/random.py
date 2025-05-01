from typing import Tuple
from app.ai.ataxx_env import AtaxxState

class RandomAlgorithm:
    """
    A simple algorithm that selects a random legal move for the Ataxx game.
    """
    
    def __init__(self):
        """Initialize the random algorithm."""
        pass
    
    def search(self, state: AtaxxState) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Select a random legal move from the current game state.
        
        Args:
            state: The current Ataxx game state.
            
        Returns:
            A tuple (from_pos, to_pos) representing the selected move.
            Each position is a tuple (row, col).
        """
        move = state.get_random_move()
        if move is None:
            raise ValueError("No legal moves available")
        return move