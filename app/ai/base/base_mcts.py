from typing import List, Dict, Any, Callable, Optional
import random
from app.ai.ataxx_env import AtaxxEnvironment

class MCTSNode:
    def __init__(self, state_key: int, valid_moves: List[Dict[str, Any]], parent: Optional['MCTSNode'] = None):
        self.state_key = state_key  # Zobrist hash
        self.valid_moves = valid_moves  # Lưu trữ danh sách nước đi
        self.parent = parent
        self.children: Dict[str, 'MCTSNode'] = {}  # key: move_key, value: node
        self.wins = 0.0
        self.visits = 0
        self.value = 0.0
        self.reward_cache: Optional[tuple[float, bool]] = None  # Cache phần thưởng

class BaseMCTS:
    def __init__(
        self,
        board: List[List[str]],
        current_player: str,
        rollout_fn: Callable[[AtaxxEnvironment, str], AtaxxEnvironment],
        reward_fn: Callable[[AtaxxEnvironment, str], tuple[float, bool]],
        select_fn: Callable[[MCTSNode], MCTSNode],
        c: float = 0.7
    ):
        self.env = AtaxxEnvironment(board, current_player)
        self.player = current_player
        self.rollout_fn = rollout_fn
        self.reward_fn = reward_fn
        self.select_fn = select_fn
        self.c = c
        self.tree: Dict[int, MCTSNode] = {}
        self.root_key = self.env.get_state_key()
        self.tree[self.root_key] = MCTSNode(self.root_key, self.env.get_valid_moves())
        self.max_nodes = 10000

    def _get_move_key(self, move: Dict[str, Any]) -> str:
        return f"{move['from']['row']},{move['from']['col']}->{move['to']['row']},{move['to']['col']}"

    def run(self, simulations: int) -> None:
        for _ in range(simulations):
            if self.root_key not in self.tree or self.tree[self.root_key] is None:
                print("Root node missing or None, reinitializing")
                self.tree[self.root_key] = MCTSNode(self.root_key, self.env.get_valid_moves())
            
            node = self.select_fn(self.tree[self.root_key])
            if node is None:
                print("select_fn returned None, skipping simulation")
                continue
                
            if not self.env.is_game_over() and node.valid_moves and node.visits > 0:
                move = random.choice(node.valid_moves)
                temp_env = self.env.clone()
                temp_env.make_move(move["from"], move["to"])
                child_key = temp_env.get_state_key()
                if child_key not in self.tree or self.tree[child_key] is None:
                    self.tree[child_key] = MCTSNode(child_key, temp_env.get_valid_moves(), node)
                if self._get_move_key(move) not in node.children:
                    node.children[self._get_move_key(move)] = self.tree[child_key]
                node = self.tree[child_key]
                rollout_env = temp_env
            else:
                rollout_env = self.env.clone()
            
            final_env = self.rollout_fn(rollout_env, self.player)
            if node.reward_cache is None:
                node.reward_cache = self.reward_fn(final_env, self.player)
            reward, is_win = node.reward_cache
            
            while node is not None:
                node.visits += 1
                node.value += reward
                if is_win:
                    node.wins += 1
                node = node.parent
            
            self._limit_tree_size()

    def _limit_tree_size(self) -> None:
        if len(self.tree) > self.max_nodes:
            print("Limiting tree size")
            sorted_nodes = sorted(
                [(k, v) for k, v in self.tree.items() if k != self.root_key],
                key=lambda x: x[1].visits
            )
            self.tree = dict(sorted_nodes[-self.max_nodes + 1:] + [(self.root_key, self.tree[self.root_key])])

    def get_move(self, board: List[List[str]], current_player: str, simulations: int, temperature: float) -> Dict[str, Any]:
        if self.root_key not in self.tree or self.tree[self.root_key] is None:
            print("Root node missing or None in get_move, reinitializing")
            self.tree[self.root_key] = MCTSNode(self.root_key, self.env.get_valid_moves())
        
        self.run(simulations)
        root = self.tree[self.root_key]
        if not root.valid_moves:
            print("No valid moves available")
            return {}
        
        if not root.children:
            print("No children nodes, selecting random valid move")
            return random.choice(root.valid_moves) if root.valid_moves else {}
        
        if temperature == 0.0:
            best_move_key = max(root.children, key=lambda k: root.children[k].visits)
        else:
            scores = {k: child.visits ** (1.0 / temperature) for k, child in root.children.items()}
            total = sum(scores.values())
            if total == 0:
                print("No visits in children, selecting random move")
                return random.choice(list(root.children.keys()))
            scores = {k: v / total for k, v in scores.items()}
            best_move_key = random.choices(list(scores.keys()), weights=list(scores.values()), k=1)[0]
        
        for move in root.valid_moves:
            if self._get_move_key(move) == best_move_key:
                return move
        print("No matching move found, returning empty move")
        return {}

    def update_root(self, move: Dict[str, Any]) -> None:
        new_env = self.env.clone()
        new_env.make_move(move["from"], move["to"])
        new_root_key = new_env.get_state_key()
        
        if new_root_key not in self.tree or self.tree[new_root_key] is None:
            print(f"Creating new node for root_key: {new_root_key}")
            self.tree[new_root_key] = MCTSNode(new_root_key, new_env.get_valid_moves())
        
        self.root_key = new_root_key
        self.env = new_env
        self.player = "yellow" if self.player == "red" else "red"