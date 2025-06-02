# -*- coding: utf-8 -*-
from .board import Board, StateMinimax
from .minimax import minimax
from .monte_carlo_domain import MonteCarloDomain

class AlphaBetaMonteCarlo:
    """Hybrid: Alpha-Beta Minimax + Monte Carlo with Domain Knowledge."""
    def __init__(self, state, **kwargs):
        self.state = state
        self.board = Board()
        self.switch_threshold = kwargs.get('switch_threshold', 13)
        self.minimax_depth = kwargs.get('minimax_depth', 4)
        self.time_limit = kwargs.get('time_limit', None)
        self.mcd = MonteCarloDomain(
            state,
            basic_simulations=kwargs.get('basic_simulations', 300),
            time_limit=self.time_limit,
            component_weights={
                'heuristic': kwargs.get('s1_ratio', 1.0),
                'tactical': kwargs.get('s2_ratio', 1.0),
                'strategic': kwargs.get('s3_ratio', 0.5)
            }
        )

    def get_play(self):
        return self.get_mcts_move()

    def get_mcts_move(self, time_limit=None):
        # Use provided time_limit or fall back to instance time_limit
        effective_time_limit = time_limit if time_limit is not None else self.time_limit
        
        total_pieces = self.state.balls[1] + self.state.balls[-1]
        empty_spaces = self.state.n_fields ** 2 - total_pieces

        if empty_spaces <= self.switch_threshold:
            return self.mcd.get_mcts_move(effective_time_limit)
        else:
            state_minimax = StateMinimax(self.state.get_board_array(), self.state.current_player(), self.state.balls)
            return minimax(self.board, state_minimax, self.minimax_depth, effective_time_limit)