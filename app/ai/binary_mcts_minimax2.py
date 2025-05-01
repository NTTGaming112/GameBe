import random
from typing import Tuple, Optional

from app.ai.ataxx_env import AtaxxState
from app.ai.binary_mcts import BinaryMCTS
from app.ai.base_mcts import MCTSNode

class BinaryMCTSMinimax2(BinaryMCTS):
    def _simulation(self, node: MCTSNode) -> float:
        state = node.state.clone()
        player = state.current_player
        depth_limit = 50
        depth = 0
        while not state.is_terminal() and depth < depth_limit:
            moves = state.get_legal_moves()
            if not moves:
                break
            move = self._minimax_move(state, depth=2, player=player)
            if move is None:
                move = random.choice(moves)
            state.make_move(*move)
            depth += 1
        result = state.get_result(player)
        if result is not None:
            return result
        counts = state.get_pieces_count()
        opponent = "yellow" if player == "red" else "red"
        total = counts[player] + counts[opponent]
        return counts[player] / total if total > 0 else 0.5
    
    def _minimax_move(self, state: AtaxxState, depth: int, player: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        def minimax(state: AtaxxState, depth: int, maximizing: bool, original_player: str) -> float:
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
                    value = minimax(new_state, depth - 1, False, original_player)
                    best_value = max(best_value, value)
                return best_value
            else:
                best_value = float('inf')
                for move in moves:
                    new_state = state.clone()
                    new_state.make_move(*move)
                    value = minimax(new_state, depth - 1, True, original_player)
                    best_value = min(best_value, value)
                return best_value
        
        moves = state.get_legal_moves()
        if not moves:
            return None
        
        best_move = None
        best_value = float('-inf')
        for move in moves:
            new_state = state.clone()
            new_state.make_move(*move)
            value = minimax(new_state, depth - 1, False, player)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move