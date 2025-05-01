import time
from typing import Tuple

from app.ai.ataxx_env import AtaxxState
from app.ai.base_mcts import BaseMCTS

class Minimax2(BaseMCTS):
    def __init__(self, iterations: int = 1000, time_limit: float = None, **kwargs):
        super().__init__(iterations=iterations, time_limit=time_limit, policy_type="random", **kwargs)
    
    def search(self, state: AtaxxState) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        start_time = time.time()
        moves = state.get_legal_moves()
        if not moves:
            return state.get_random_move()
        
        best_move = None
        best_value = float('-inf')
        for move in moves:
            if self.time_limit and time.time() - start_time >= self.time_limit:
                break
            new_state = state.clone()
            new_state.make_move(*move)
            value = self._minimax(new_state, depth=1, maximizing=False, original_player=state.current_player)
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move if best_move else state.get_random_move()
    
    def _minimax(self, state: AtaxxState, depth: int, maximizing: bool, original_player: str) -> float:
        if depth == 0 or state.is_terminal():
            counts = state.get_pieces_count()
            opponent = "yellow" if original_player == "red" else "red"
            total = counts[original_player] + counts[opponent]
            return counts[original_player] / total if total > 0 else 0.5
        
        moves = state.get_legal_moves()
        if not moves:
            return state.get_result(original_player) or 0.5
        
        if maximizing:
            best_value = float('-inf')
            for move in moves:
                new_state = state.clone()
                new_state.make_move(*move)
                value = self._minimax(new_state, depth - 1, False, original_player)
                best_value = max(best_value, value)
            return best_value
        else:
            best_value = float('inf')
            for move in moves:
                new_state = state.clone()
                new_state.make_move(*move)
                value = self._minimax(new_state, depth - 1, True, original_player)
                best_value = min(best_value, value)
            return best_value