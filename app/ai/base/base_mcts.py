from app.ai.ataxx_env import AtaxxEnvironment
from typing import Dict, Any, Optional

class Node:
    def __init__(self, move: Optional[Dict[str, Any]] = None, parent: Optional['Node'] = None):
        self.move = move  # Move dạng {"from": (row, col), "to": (row, col)}
        self.parent = parent
        self.children: list['Node'] = []
        self.visits: int = 0
        self.value: float = 0.0  # Tổng giá trị cho Backpropagation
        self.wins: int = 0  # Số lần thắng cho UCT Winrate

class BaseMCTS:
    def __init__(self, board, current_player, rollout_fn, reward_fn, select_fn):
        self.env = AtaxxEnvironment(board, current_player)
        self.player = current_player
        self.rollout_fn = rollout_fn  # Hàm mô phỏng (simulation)
        self.reward_fn = reward_fn  # Hàm tính thưởng (binary hoặc fractional)
        self.select_fn = select_fn  # Hàm chọn nút (UCT winrate hoặc fractional)
        self.root = Node()

    def run(self, simulations: int) -> Optional[Dict[str, Any]]:
        for _ in range(simulations):
            node = self.root
            env = self.env.clone()  # Sao chép môi trường để không ảnh hưởng trạng thái gốc

            # Selection
            while node.children and not env.is_game_over():
                node = self.select_fn(node)  # Chọn nút con theo hàm select_fn
                env.make_move(node.move["from"], node.move["to"])

            # Expansion
            if not env.is_game_over() and node.visits > 0:
                moves = env.get_valid_moves()  # Lấy tất cả nước đi hợp lệ
                for move in moves:
                    # Kiểm tra xem move đã tồn tại trong children chưa
                    if not any(child.move == move for child in node.children):
                        child = Node(move=move, parent=node)
                        node.children.append(child)
                if node.children:
                    node = node.children[-1]  # Chọn nút con mới nhất để mô phỏng
                    env.make_move(node.move["from"], node.move["to"])

            # Simulation
            final_env = self.rollout_fn(env, self.player, max_depth=50)  # Tăng max_depth để mô phỏng đầy đủ

            # Backpropagation
            reward = self.reward_fn(final_env, self.player)
            while node:
                node.visits += 1
                node.value += reward  # Cập nhật giá trị fractional/binary
                # Cập nhật wins chỉ khi dùng binary reward và thưởng > 0.5 (thắng)
                if self.reward_fn.__name__ == "binary_reward" and reward > 0.5:
                    node.wins += 1
                node = node.parent

        # Chọn nước đi tốt nhất dựa trên số lượt thăm
        best_child = max(self.root.children, key=lambda c: c.visits, default=None)
        return best_child.move if best_child else None