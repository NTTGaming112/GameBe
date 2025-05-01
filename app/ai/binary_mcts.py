import time
from typing import Tuple

from app.ai.ataxx_env import AtaxxState
from app.ai.base_mcts import BaseMCTS, MCTSNode

class BinaryMCTS(BaseMCTS):
    def search(self, state: AtaxxState) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        root = MCTSNode(state, transposition_table=self.transposition_table)
        start_time = time.time()
        for _ in range(self.iterations):
            if self.time_limit and time.time() - start_time >= self.time_limit:
                break
            self._run_iteration(root)
        
        if not root.children:
            return state.get_random_move()
        best_child = max(root.children, key=lambda c: c.wins / c.visits if c.visits > 0 else 0)
        return best_child.move
    
    def _run_iteration(self, root: MCTSNode) -> None:
        node = self._selection(root)
        if not node.is_terminal() and node.visits > 0:
            child = node.expand()
            if child:
                node = child
        result = self._simulation(node)
        self._backpropagation(node, result)
    
    def _selection(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.select_child(self.exploration_weight)
        return node
    
    def _simulation(self, node: MCTSNode) -> float:
        state = node.state.clone()
        player = state.current_player
        depth_limit = 50
        depth = 0
        while not state.is_terminal() and depth < depth_limit:
            moves = state.get_legal_moves()
            if not moves:
                break
            move = self.policy.select_move(state, moves)
            state.make_move(*move)
            depth += 1
        result = state.get_result(player)
        if result is not None:
            return result
        counts = state.get_pieces_count()
        opponent = "yellow" if player == "red" else "red"
        total = counts[player] + counts[opponent]
        return counts[player] / total if total > 0 else 0.5
    
    def _backpropagation(self, node: MCTSNode, result: float) -> None:
        while node:
            node.update(result)
            node = node.parent
            result = 1.0 - result