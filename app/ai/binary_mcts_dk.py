import math
import random
from typing import Tuple

from app.ai.ataxx_env import AtaxxState
from app.ai.binary_mcts import BinaryMCTS
from app.ai.base_mcts import MCTSNode

class BinaryMCTSDK(BinaryMCTS):
    def __init__(self, temperature: float = 0.7, **kwargs):
        super().__init__(policy_type="heuristic", **kwargs)
        self.temperature = temperature
    
    def _simulation(self, node: MCTSNode) -> float:
        state = node.state.clone()
        player = state.current_player
        depth_limit = 50
        depth = 0
        while not state.is_terminal() and depth < depth_limit:
            moves = state.get_legal_moves()
            if not moves:
                break
            scores = [self._evaluate_move_heuristic(state, m[0], m[1]) for m in moves]
            if all(s == 0 for s in scores):
                move = random.choice(moves)
            else:
                max_score = max(scores)
                exp_scores = [math.exp((s - max_score) / self.temperature) for s in scores]
                total = sum(exp_scores)
                probabilities = [e / total for e in exp_scores]
                move = random.choices(moves, weights=probabilities, k=1)[0]
            state.make_move(*move)
            depth += 1
        result = state.get_result(player)
        if result is not None:
            return result
        counts = state.get_pieces_count()
        opponent = "yellow" if player == "red" else "red"
        total = counts[player] + counts[opponent]
        return counts[player] / total if total > 0 else 0.5
    
    def _evaluate_move_heuristic(self, state: AtaxxState, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        distance = max(abs(to_row - from_row), abs(to_col - from_col))
        captures = 0
        friendly_neighbors = 0
        friendly_source_neighbors = 0
        s1, s2, s3, s4 = 1.0, 0.4, 0.7, 0.4
        
        opponent = "yellow" if state.current_player == "red" else "red"
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                adj_row, adj_col = to_row + dr, to_col + dc
                if state.is_valid_position(adj_row, adj_col):
                    if state.board[adj_row][adj_col] == opponent:
                        captures += 1
                    elif state.board[adj_row][adj_col] == state.current_player:
                        friendly_neighbors += 1
        
        if distance > 1:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    adj_row, adj_col = from_row + dr, from_col + dc
                    if state.is_valid_position(adj_row, adj_col) and state.board[adj_row][adj_col] == state.current_player:
                        friendly_source_neighbors += 1
        
        score = s1 * captures + s2 * friendly_neighbors + s3 * (1 if distance <= 1 else 0) - s4 * friendly_source_neighbors
        return max(0, score)