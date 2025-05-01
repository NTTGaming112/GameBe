import math
import random
from typing import List, Tuple, Dict, Optional

from app.ai.ataxx_env import AtaxxState

class Policy:
    def select_move(self, state: AtaxxState, moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        raise NotImplementedError


class RandomPolicy(Policy):
    def select_move(self, state: AtaxxState, moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return random.choice(moves)


class HeuristicPolicy(Policy):
    def __init__(self, capture_weight: float = 1.0):
        super().__init__()
        self.capture_weight = capture_weight
    
    def evaluate_move(self, state: AtaxxState, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        distance = max(abs(to_row - from_row), abs(to_col - from_col))
        captures = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                adj_row, adj_col = to_row + dr, to_col + dc
                if state.is_valid_position(adj_row, adj_col):
                    opponent = "yellow" if state.current_player == "red" else "red"
                    if state.board[adj_row][adj_col] == opponent:
                        captures += 1
        if distance <= 1:  # Adjacent move
            base_score = 5.0
        else:  # Jump move
            base_score = 1.0
        capture_score = captures * self.capture_weight
        return base_score + capture_score
    
    def select_move(self, state: AtaxxState, moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        scores = [self.evaluate_move(state, move[0], move[1]) for move in moves]
        max_score = max(scores)
        best_moves = [move for move, score in zip(moves, scores) if score == max_score]
        return random.choice(best_moves)


class UCBPolicy(Policy):
    def __init__(self, exploration_weight: float = 1.0):
        super().__init__()
        self.exploration_weight = exploration_weight
        self.move_stats = {}  # Format: {move: {"visits": int, "wins": float}}
    
    def select_move(self, state: AtaxxState, moves: List[Tuple[Tuple[int, int], Tuple[int, int]]], total_simulations: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        ucb_scores = []
        for move in moves:
            move_key = str(move)
            stats = self.move_stats.get(move_key, {"visits": 0, "wins": 0.0})
            visits = stats["visits"]
            wins = stats["wins"]
            if visits == 0:
                ucb_score = float('inf')
            else:
                exploitation = wins / visits
                exploration = self.exploration_weight * math.sqrt(math.log(total_simulations) / visits)
                ucb_score = exploitation + exploration
            ucb_scores.append(ucb_score)
        
        max_score = max(ucb_scores)
        best_moves = [move for move, score in zip(moves, ucb_scores) if score == max_score]
        return random.choice(best_moves)
    
    def update_stats(self, move: Tuple[Tuple[int, int], Tuple[int, int]], result: float) -> None:
        move_key = str(move)
        if move_key not in self.move_stats:
            self.move_stats[move_key] = {"visits": 0, "wins": 0.0}
        self.move_stats[move_key]["visits"] += 1
        self.move_stats[move_key]["wins"] += result


class EPolicyMCTS(Policy):
    def __init__(self, epsilon: float = 0.1, capture_weight: float = 1.0, exploration_weight: float = 1.0):
        super().__init__()
        self.epsilon = epsilon
        self.heuristic_policy = HeuristicPolicy(capture_weight=capture_weight)
        self.ucb_policy = UCBPolicy(exploration_weight=exploration_weight)
    
    def select_move(self, state: AtaxxState, moves: List[Tuple[Tuple[int, int], Tuple[int, int]]], total_simulations: int = 0) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if random.random() < self.epsilon:
            return self.heuristic_policy.select_move(state, moves)
        else:
            return self.ucb_policy.select_move(state, moves, total_simulations)
    
    def update_stats(self, move: Tuple[Tuple[int, int], Tuple[int, int]], result: float) -> None:
        self.ucb_policy.update_stats(move, result)


def create_policy(policy_type: str, **policy_args) -> Policy:
    if policy_type == "random":
        return RandomPolicy()
    elif policy_type == "heuristic":
        return HeuristicPolicy(**policy_args)
    elif policy_type == "ucb":
        return UCBPolicy(**policy_args)
    elif policy_type == "epolicy":
        return EPolicyMCTS(**policy_args)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")