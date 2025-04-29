from app.ai.ataxx_env import AtaxxEnvironment
from typing import Dict, Any, Optional, List, Callable
import random

class Node:
    def __init__(self, move: Optional[Dict[str, Any]] = None, parent: Optional['Node'] = None):
        self.move = move
        self.parent = parent
        self.children: List['Node'] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.wins: int = 0

class BaseMCTS:
    def __init__(
        self,
        board,
        current_player,
        rollout_fn: Callable,
        reward_fn: Callable,
        select_fn: Callable
    ):
        self.env = AtaxxEnvironment(board, current_player)
        self.player = current_player
        self.rollout_fn = rollout_fn
        self.reward_fn = reward_fn
        self.select_fn = select_fn
        self.root = Node()

    def run(self, simulations: int, reset_root: bool = False) -> Optional[Dict[str, Any]]:
        if reset_root:
            self.root = Node()

        for _ in range(simulations):
            node = self.root
            env = self.env.clone()

            # Selection
            while node.children and not env.is_game_over():
                node = self.select_fn(node)
                env.make_move(node.move["from"], node.move["to"])

            # Expansion
            if not env.is_game_over() and not node.children:
                moves = env.get_valid_moves()
                for move in moves:
                    if not any(child.move == move for child in node.children):
                        child = Node(move=move, parent=node)
                        node.children.append(child)

                unvisited = [child for child in node.children if child.visits == 0]
                if unvisited:
                    node = random.choice(unvisited)
                    env.make_move(node.move["from"], node.move["to"])

            # Simulation
            final_env = self.rollout_fn(env, self.player)

            # Backpropagation
            reward, is_win = self.reward_fn(final_env, self.player) \
                if isinstance(self.reward_fn(final_env, self.player), tuple) \
                else (self.reward_fn(final_env, self.player), self.reward_fn(final_env, self.player) > 0.5)

            while node:
                node.visits += 1
                node.value += reward
                if is_win:
                    node.wins += 1
                node = node.parent

        best_child = max(self.root.children, key=lambda c: c.visits, default=None)
        return best_child.move if best_child else None