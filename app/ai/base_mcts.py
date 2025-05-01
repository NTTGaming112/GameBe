import math
import random
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod

from app.ai.ataxx_env import AtaxxState
from app.ai.policy import create_policy

class MCTSNode:
    def __init__(self, state: AtaxxState, parent=None, move=None, transposition_table: Dict[str, 'MCTSNode'] = None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = state.get_legal_moves()
        self.player = state.current_player
        self.transposition_table = transposition_table
    
    def select_child(self, exploration_weight: float = 1.0) -> 'MCTSNode':
        return max(self.children, key=lambda c: 
            (c.wins / c.visits if c.visits > 0 else float('inf')) + 
            exploration_weight * math.sqrt(2 * math.log(self.visits) / c.visits if c.visits > 0 else float('inf'))
        )
    
    def select_child_fractional(self, exploration_weight: float = 1.0) -> 'MCTSNode':
        # Use piece count difference for UCT score
        max_diff = self.state.BOARD_SIZE * self.state.BOARD_SIZE  # 49 for 7x7 board
        return max(self.children, key=lambda c: 
            (
                # X': Normalized piece count difference
                (
                    (c.state.get_pieces_count()[self.player] - c.state.get_pieces_count()[
                        "yellow" if self.player == "red" else "red"
                    ]) + max_diff
                ) / (2 * max_diff)
                if c.visits > 0 else float('inf')
            ) + 
            exploration_weight * math.sqrt(2 * math.log(self.visits) / c.visits if c.visits > 0 else float('inf'))
        )
    
    def expand(self) -> Optional['MCTSNode']:
        if not self.untried_moves:
            return None
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        new_state = self.state.clone()
        new_state.make_move(*move)
        key = new_state.hash()
        if key in self.transposition_table:
            child = self.transposition_table[key]
        else:
            child = MCTSNode(new_state, parent=self, move=move, transposition_table=self.transposition_table)
            self.transposition_table[key] = child
        self.children.append(child)
        return child
    
    def update(self, result: float) -> None:
        self.visits += 1
        self.wins += result
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        return self.state.is_terminal()


class BaseMCTS(ABC):
    def __init__(self, iterations: int = 1000, exploration_weight: float = 1.0, time_limit: float = None, policy_type: str = "random", **policy_args):
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.time_limit = time_limit
        self.policy = create_policy(policy_type, **policy_args)
        self.transposition_table: Dict[str, MCTSNode] = {}
    
    @abstractmethod
    def search(self, state: AtaxxState) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        pass