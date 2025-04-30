import math
import random
from typing import Dict, Any, Optional, List, Callable
from app.ai.ataxx_env import AtaxxEnvironment

def gumbal_softmax(arr: List[float], temperature: float) -> List[float]:
    """Chuyển đổi mảng số thành phân phối xác suất với nhiệt độ."""
    arr = [v / temperature for v in arr]
    mx = max(arr)
    arr = [math.exp(v - mx) for v in arr]
    s = sum(arr)
    return [v / s for v in arr]

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
        board: Any,
        current_player: Any,
        rollout_fn: Callable,
        reward_fn: Callable,
        select_fn: Callable,
        c: float = 2.0
    ):
        self.env = AtaxxEnvironment(board, current_player)
        self.player = current_player
        self.rollout_fn = rollout_fn
        self.reward_fn = reward_fn
        self.select_fn = select_fn
        self.c = c
        self.root: Optional[Node] = None

    def update_root(self, new_board: Any, new_player: Any) -> None:
        """Cập nhật nút gốc dựa trên trạng thái mới, tái sử dụng cây nếu có thể."""
        new_env = AtaxxEnvironment(new_board, new_player)
        if self.root is None:
            self.root = Node()
            self.env = new_env
            self.player = new_player
            return

        # Tìm nút con khớp với trạng thái mới
        next_root = None
        for child in self.root.children:
            temp_env = self.env.clone()
            temp_env.make_move(child.move["from"], child.move["to"])
            if temp_env.board == new_board and temp_env.current_player == new_player:
                next_root = child
                break

        if next_root is None:
            # Nếu không tìm thấy, tạo cây mới
            self.root = Node()
        else:
            # Tái sử dụng cây, ngắt liên kết với cha
            next_root.parent = None
            self.root = next_root

        self.env = new_env
        self.player = new_player

    def run(
        self,
        simulations: int,
        temperature: float = 0.0,
        reset_root: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Chạy MCTS với số lần mô phỏng, hỗ trợ tái sử dụng cây và xác suất nước đi."""
        if reset_root or self.root is None:
            self.root = Node()

        for _ in range(simulations):
            node = self.root
            env = self.env.clone()

            # Selection
            while node.children and not env.is_game_over():
                node = self.select_fn(node, self.c)
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

        if not self.root.children:
            return None

        # Chọn nước đi dựa trên temperature
        if temperature == 0:
            best_child = max(self.root.children, key=lambda c: c.visits)
            return best_child.move
        else:
            visits = [child.visits for child in self.root.children]
            probs = gumbal_softmax(visits, temperature)
            acc_probs = [probs[0]]
            for i in range(1, len(probs)):
                acc_probs.append(acc_probs[-1] + probs[i])
            
            r = random.random()
            for i, prob in enumerate(acc_probs):
                if r < prob:
                    return self.root.children[i].move
            return self.root.children[-1].move

    def get_move(self, board: Any, player: Any, simulations: int, temperature: float = 0.0) -> Optional[Dict[str, Any]]:
        """Tiện ích để lấy nước đi, tự động cập nhật cây."""
        self.update_root(board, player)
        return self.run(simulations, temperature)