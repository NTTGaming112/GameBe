#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid Alpha-Beta Minimax + Monte Carlo implementation for Ataxx AI.

This module combines the strengths of Alpha-Beta pruning and Monte Carlo
simulation by using Minimax for the early game and Monte Carlo for the endgame.
"""
from copy import deepcopy
from .board import Board, StateMinimax
from .minimax import minimax
from .monte_carlo_domain import MonteCarloDomain

class AlphaBetaMonteCarlo:
    """Hybrid: Alpha-Beta Minimax + Monte Carlo with Domain Knowledge.
    
    This class implements a hybrid approach that switches between Alpha-Beta Minimax
    for the early game and Monte Carlo with Domain Knowledge for the endgame.
    The switching happens based on the number of empty cells remaining on the board.
    """
    def __init__(self, state, **kwargs):
        """Initialize a hybrid Alpha-Beta + Monte Carlo player.
        
        Args:
            state: Initial game state
            **kwargs: Configuration parameters including:
                - switch_threshold: Number of empty spaces at which to switch algorithms (default: 25)
                - minimax_depth: Search depth for Minimax algorithm (default: 4)
                - Other parameters are passed to the Monte Carlo algorithm
        """
        self.state = state
        self.board = Board()
        self.switch_threshold = kwargs.get('switch_threshold', 25)
        self.mcd = MonteCarloDomain(state, **kwargs)
        self.minimax_depth = kwargs.get('minimax_depth', 4)
        
    def get_play(self):
        """Alias for get_move().
        
        Returns:
            Move: Best move found by the algorithm
        """
        return self.get_move()
        
    def get_move(self):
        """Select the best move using the hybrid approach.
        
        Returns:
            Move: Best move found by the selected algorithm
        """
        total_pieces = self.state.balls[1] + self.state.balls[-1]
        empty_spaces = self.state.n_fields**2 - total_pieces  # 7x7 board has 49 cells
        
        # Choose algorithm based on number of empty spaces
        if empty_spaces <= self.switch_threshold:
            return self.mcd.get_move()  # Use Monte Carlo with Domain Knowledge
        else:
            # Use Alpha-Beta Minimax
            state_minimax = StateMinimax(self.state.board, self.state.current_player(), self.state.balls)
            return minimax(self.board, state_minimax, self.minimax_depth)
