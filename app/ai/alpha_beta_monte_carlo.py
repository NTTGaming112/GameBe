#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
from .board import Board, StateMinimax
from .minimax import minimax
from .monte_carlo_domain import MonteCarloDomain

class AlphaBetaMonteCarlo:
    """Hybrid: Alpha-Beta + Monte Carlo Domain Knowledge."""
    def __init__(self, state, **kwargs):
        self.state = state
        self.board = Board()
        self.switch_threshold = kwargs.get('switch_threshold', 31)
        self.mcd = MonteCarloDomain(state, **kwargs)
        self.minimax_depth = kwargs.get('minimax_depth', 4)
        
    def get_play(self):
        """Alias for get_move()"""
        return self.get_move()
        
    def get_move(self):
        """Chọn nước đi tốt nhất sử dụng Hybrid: Alpha-Beta + Monte Carlo."""
        total_pieces = self.state.balls[1] + self.state.balls[-1]
        empty_spaces = 49 - total_pieces
        
        # Chọn thuật toán dựa trên số ô trống
        if empty_spaces <= self.switch_threshold:
            return self.mcd.get_move()  # Sử dụng MCD
        else:
            # Sử dụng Alpha-Beta Minimax
            state_minimax = StateMinimax(self.state.board, self.state.current_player(), self.state.balls)
            return minimax(self.board, state_minimax, self.minimax_depth)
