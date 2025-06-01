#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimax player implementation for Ataxx AI.

This module provides a wrapper for the Minimax algorithm to be used with the same
interface as the Monte Carlo players.
"""
from app.ai.constants import CLONE_MOVE, PLAYER_ONE, PLAYER_TWO
from .minimax import minimax
from .board import Board, StateMinimax

class MinimaxPlayer:
    """
    Minimax player with Alpha-Beta pruning.
    
    This class implements the Alpha-Beta Minimax algorithm with the same interface
    as the Monte Carlo players to be compatible with the API.
    """
    def __init__(self, state, depth=4, time_limit=None):
        """
        Initialize the Minimax player.
        
        Args:
            state: Current game state
            depth: Maximum search depth (default: 4)
            time_limit: Maximum time allowed for search (seconds)
        """
        self.state = self._convert_state(state)
        self.depth = depth
        self.time_limit = time_limit
        self.board = Board()

    def get_play(self):
        """
        Get the best move for the current player.
        
        Returns:
            The best move found by the algorithm
        """
        return self.get_move(self.time_limit)
    
    def get_move(self, time_limit=None):
        """
        Get the best move using Alpha-Beta Minimax.
        
        Returns:
            The best move found by the algorithm
        """
        # Use provided time_limit if given, else default
        tl = time_limit if time_limit is not None else self.time_limit
        return minimax(self.board, self.state, self.depth, time_limit=tl)
        
    def search(self, state):
        """
        Search for the best move using Alpha-Beta Minimax.
        
        Args:
            state: Current game state
            
        Returns:
            The best move found by the algorithm
        """
        # Convert Ataxx state to StateMinimax
        minimax_state = self._convert_state(state)
        
        # Get the best move using minimax
        best_move = minimax(self.board, minimax_state, self.depth)
        
        # If no move is found, return None
        if not best_move:
            return None
            
        # The move format is already (from_pos, to_pos) where from_pos can be None for clones
        # No conversion needed since minimax now returns the correct format
        return best_move
        
    def _convert_state(self, ataxx_state):
        """
        Convert Ataxx state to StateMinimax.
        
        Args:
            ataxx_state: Ataxx state object
            
        Returns:
            StateMinimax object that can be used with the minimax algorithm
        """
        # Get 2D board array from Ataxx bitboards
        board_array = ataxx_state.get_board_array()
        
        # Get current player
        player = ataxx_state.current_player()
        
        # Get piece counts directly from ataxx_state
        balls = {
            PLAYER_ONE: ataxx_state.balls[PLAYER_ONE],
            PLAYER_TWO: ataxx_state.balls[PLAYER_TWO]
        }
        
        # Create StateMinimax with required arguments
        minimax_state = StateMinimax(board_array, player, balls)
        return minimax_state
