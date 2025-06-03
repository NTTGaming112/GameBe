import random
import math
from heuristics import evaluate
from constants import DEFAULT_MCTS_ITERATIONS

class MCTSNode:
    def __init__(self, state, move=None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.uct_c = 1.41
        self.untried_moves = state.get_legal_moves()

    def select_child(self):
        def ucb1_value(c):
            if c.visits == 0:
                return float('inf')
            exploitation = c.value / c.visits
            exploration = self.uct_c * math.sqrt(math.log(self.visits) / c.visits)
            return exploitation + exploration
        
        return max(self.children, key=ucb1_value)

    def expand(self):
        move = self.untried_moves.pop(0)
        new_state = self.state.copy()
        new_state.make_move(move)
        child = MCTSNode(new_state, move, self)
        self.children.append(child)
        return child

class MCTSAgent:
    def __init__(self, iterations=DEFAULT_MCTS_ITERATIONS):
        self.iterations = iterations

    def get_move(self, state):
        root = MCTSNode(state)
        root_player = state.current_player
        
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
            
            # Đánh giá kết quả theo root player
            result = evaluate(sim_state, root_player)
            
            # Backpropagation
            while node:
                node.visits += 1
                
                if node.state.current_player == root_player:
                    node.value += result
                else:
                    node.value += (1 - result)
                
                node = node.parent

        if not root.children:
            return None
            
        # Chọn move có nhiều visits nhất (robust choice)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move