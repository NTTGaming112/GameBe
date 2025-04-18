from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random
import numpy as np
from app.ai.ataxx_env import AtaxxEnvironment

class MCTSNode:
    def __init__(self, state: Tuple[Tuple[str, ...], ...], current_player: str, 
                 parent: Optional["MCTSNode"] = None, move: Optional[Dict] = None):
        self.state = state
        self.current_player = current_player
        self.parent = parent
        self.move = move
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.wins = 0
        self.captured_pieces = 0

class MCTS:
    def __init__(self, board: List[List[str]], current_player: str, 
                 iterations: int = 300, move_history: Optional[defaultdict] = None):
        self.env = AtaxxEnvironment([row[:] for row in board], current_player)
        self.root = MCTSNode(self.env.get_state(), current_player)
        self.iterations = iterations
        self.exploration_constant = 0.7
        self.move_history = move_history or defaultdict(lambda: defaultdict(float))
        self.bot_player = current_player
        self.history_weight = 0.6
        self.capture_weight = 0.4
        self.state_cache: Dict[Tuple[Tuple[str, ...], ...], float] = {}  # Cache kết quả mô phỏng

    def select(self) -> "MCTSNode":
        node = self.root
        while node.children:
            node = max(node.children, key=lambda c: (
                (c.wins / (c.visits or 1)) +
                self.exploration_constant * np.sqrt(np.log(node.visits or 1) / (c.visits or 1))
            ))
        return node

    def expand(self, node: "MCTSNode") -> "MCTSNode":
        env = AtaxxEnvironment([list(row) for row in node.state], node.current_player)
        valid_moves = env.get_valid_moves()
        if not valid_moves or env.is_game_over():
            return node

        for move in valid_moves:
            env_temp = AtaxxEnvironment([list(row) for row in node.state], node.current_player)
            captured = env_temp.count_captured_pieces(move["to"])
            env_temp.make_move(move["from"], move["to"])
            child = MCTSNode(env_temp.get_state(), env_temp.current_player, parent=node, move=move)
            child.captured_pieces = captured
            node.children.append(child)
        return random.choice(node.children)

    def simulate(self, node: "MCTSNode") -> float:
        state = node.state
        if state in self.state_cache:
            return self.state_cache[state]

        env = AtaxxEnvironment([list(row) for row in state], node.current_player)
        state_key = str(state)
        depth = 0
        max_depth = 20  # Giới hạn độ sâu mô phỏng để tránh chạy quá lâu

        while not env.is_game_over() and depth < max_depth:
            moves = env.get_valid_moves()
            if not moves:
                break

            best_move = None
            best_score = -float("inf")
            for move in moves:
                move_key = f"{move['from']['row']},{move['from']['col']}-{move['to']['row']},{move['to']['col']}"
                history_score = self.move_history[state_key].get(move_key, 0)
                capture_score = env.count_captured_pieces(move["to"])
                score = (self.history_weight * history_score) + (self.capture_weight * capture_score)
                if score > best_score:
                    best_score = score
                    best_move = move

            move = best_move if best_move else random.choice(moves)
            env.make_move(move["from"], move["to"])
            state_key = str(env.get_state())
            depth += 1

        result = env.get_reward(self.bot_player)
        self.state_cache[state] = result
        return result

    def backpropagate(self, node: "MCTSNode", result: float) -> None:
        while node:
            node.visits += 1
            node.wins += result
            node = node.parent

    def search(self) -> Optional[Dict]:
        for _ in range(self.iterations):
            node = self.select()
            expanded_node = self.expand(node)
            result = self.simulate(expanded_node)
            self.backpropagate(expanded_node, result)

        if not self.root.children:
            return None

        best_child = max(self.root.children, key=lambda c: (
            (c.visits * 0.7) + (c.captured_pieces * 0.3)
        ))
        return best_child.move