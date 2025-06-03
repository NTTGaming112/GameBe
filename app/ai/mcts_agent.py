import random
import math
from ataxx_state import AtaxxState
from heuristics import evaluate

# Constants
UCT_C = 1.41

class MCTSNode:
    def __init__(self, state, move=None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = state.get_legal_moves()

    def select_child(self):
        return max(self.children, key=lambda c: c.value / c.visits + UCT_C * math.sqrt(2 * math.log(self.visits) / c.visits) if c.visits > 0 else float('inf'))

    def expand(self):
        move = self.untried_moves.pop(0)
        new_state = self.state.copy()
        new_state.make_move(move)
        child = MCTSNode(new_state, move, self)
        self.children.append(child)
        return child

class MCTSAgent:
    def __init__(self, iterations=300):
        self.iterations = iterations

    def get_move(self, state):
        root = MCTSNode(state)
        for _ in range(self.iterations):
            node = root
            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
            # Expansion
            if node.untried_moves:
                node = node.expand()
            # Simulation
            sim_state = node.state.copy()
            while not sim_state.is_game_over():
                moves = sim_state.get_legal_moves()
                if not moves:
                    break
                sim_state.make_move(random.choice(moves))
            result = evaluate(sim_state, state.current_player)
            # Backpropagation
            while node:
                node.visits += 1
                # score là từ góc nhìn của player tại nút gốc
                # Đổi dấu nếu là đối thủ
                node.value += result if node.state.current_player == state.current_player else 1 - result
                node = node.parent

        return max(root.children, key=lambda c: c.visits).move if root.children else None