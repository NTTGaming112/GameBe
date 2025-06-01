
class MonteCarloNode:
    def __init__(self, state, parent=None, move=None, mcd_instance=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.mcd_instance = mcd_instance
        self.children = []
        self.ep = 0
        self.visits = 0
        self.untried_moves = state.get_all_possible_moves() if not state.is_game_over() else []
        self._prior_score = None

    def get_structured_prior(self):
        if self._prior_score is not None:
            return self._prior_score
        if self.mcd_instance is None or self.parent is None or self.move is None:
            self._prior_score = 0.5
        else:
            phase = self.mcd_instance._detect_game_phase(self.parent.state)
            self._prior_score = self.mcd_instance._calculate_structured_move_score(self.parent.state, self.move, phase)
        return self._prior_score

    def uct_value(self, c=1.414):
        if self.visits == 0:
            return float('inf')
        import math
        exploitation = self.ep / self.visits
        exploration = c * math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        prior = self.get_structured_prior()
        prior_weight = max(0.1, 1.0 - self.visits / 10.0)
        return exploitation + exploration + prior_weight * prior

    def select_child(self):
        return max(self.children, key=lambda c: c.uct_value())

    def expand(self):
        # Multiple safety checks for untried_moves
        if not hasattr(self, 'untried_moves') or self.untried_moves is None:
            return None
            
        if not self.untried_moves or len(self.untried_moves) == 0:
            return None
        
        try:
            move = self.untried_moves.pop(0)
        except IndexError:
            # This should not happen with our checks above, but just in case
            return None
            
        undo_info = self.state.apply_move_with_undo(move)
        child = MonteCarloNode(self.state, self, move, self.mcd_instance)
        self.children.append(child)
        self.state.undo_move(undo_info)
        return child

    def update(self, result):
        self.visits += 1
        self.ep += result