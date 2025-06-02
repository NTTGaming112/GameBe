import math
from collections import deque
from threading import Lock

class MonteCarloNode:
    def __init__(self, state, parent=None, move=None, mcd_instance=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.ep = 0
        self.visits = 0
        self.mcd_instance = mcd_instance
        self.untried_moves = deque(state.get_all_possible_moves()) if not state.is_game_over() else deque()
        self.lock = Lock()  # Thread-safe lock for this node

    def uct_value(self, c=1.414):
        if self.visits == 0 or self.parent.visits == 0:
            return float('inf')

        if self.parent is None:
            return self.ep / self.visits
        
        exploitation = self.ep / self.visits
        exploration = c * math.sqrt(2 * math.log(self.parent.visits) / self.visits)

        if self.mcd_instance and hasattr(self.mcd_instance, 'get_structured_prior'):
            prior = self.mcd_instance.get_structured_prior(self)
            prior_weight = max(0.1, 1.0 - self.visits / 10.0)
            return exploitation + exploration + prior_weight * prior
        
        return exploitation + exploration

    def select_child(self):
        return max(self.children, key=lambda c: c.uct_value())

    def expand(self):
        # Multiple safety checks for untried_moves
        if not hasattr(self, 'untried_moves') or self.untried_moves is None:
            return None
            
        if not self.untried_moves or len(self.untried_moves) == 0:
            return None
        
        try:
            move = self.untried_moves.popleft()
        except IndexError:
            # This should not happen with our checks above, but just in case
            return None
            
        undo_info = self.state.apply_move_with_undo(move)
        child = MonteCarloNode(self.state, self, move)
        self.children.append(child)
        self.state.undo_move(undo_info)
        return child

    def backpropagate(self, result):
        with self.lock:
            self.visits += 1
            self.ep += result
        if self.parent:
            self.parent.backpropagate(1 - result)

    def is_root(self):
        """Check if this is the root node"""
        return self.parent is None

    def is_leaf(self):
        """Check if this is a leaf node (no children)"""
        return len(self.children) == 0

    def is_fully_expanded(self):
        """Check if all possible moves have been tried"""
        return len(self.untried_moves) == 0

    def is_terminal(self):
        """Check if this represents a terminal game state"""
        return self.state.is_game_over()

    def get_win_rate(self):
        """Get the win rate for this node"""
        return self.ep / self.visits if self.visits > 0 else 0.0

    def get_depth(self):
        """Get depth of this node from root"""
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth