from app.ai.ataxx_env import AtaxxEnvironment
from typing import Dict, Any

class Node:
    def __init__(self, move: Dict[str, Any] = None, parent=None):
        self.move = move
        self.parent = parent
        self.children: list = []
        self.visits = 0
        self.value = 0.0  # Tổng giá trị (dùng cho Backpropagation)
        self.wins = 0  # Số lần thắng (dùng cho UCT Winrate)

class BaseMCTS:
    def __init__(self, board, current_player, rollout_fn, reward_fn, select_fn):
        self.env = AtaxxEnvironment(board, current_player)
        self.player = current_player
        self.rollout_fn = rollout_fn
        self.reward_fn = reward_fn
        self.select_fn = select_fn  # Hàm selection (uct_select_winrate hoặc uct_select_fractional)
        self.root = Node()

    def run(self, simulations: int):
        for _ in range(simulations):
            node = self.root
            env = self.env.clone()

            # Selection
            while node.children:
                node = self.select_fn(node)  # Dùng hàm selection được truyền vào
                env.make_move(node.move["from"], node.move["to"])

            # Expansion
            if node.visits > 0:
                moves = env.get_valid_moves()
                for move in moves:
                    child = Node(move=move, parent=node)
                    node.children.append(child)
                if node.children:
                    node = node.children[0]
                    env.make_move(node.move["from"], node.move["to"])

            # Simulation
            final_env = self.rollout_fn(env, self.player)

            # Backpropagation
            reward = self.reward_fn(final_env, self.player)
            while node:
                node.visits += 1
                node.value += reward  # Tích lũy giá trị (fractional hoặc binary)
                if self.reward_fn.__name__ == "binary_reward" and reward > 0.5:  # Nếu dùng binary reward
                    node.wins += 1
                node = node.parent

        # Chọn nước đi tốt nhất
        best_child = max(self.root.children, key=lambda c: c.visits, default=None)
        return best_child.move if best_child else None