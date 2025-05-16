#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimax player implementation for Ataxx AI.

This module provides a wrapper for the Minimax algorithm to be used with the same
interface as the Monte Carlo players.
"""
from app.ai.constants import CLONE_MOVE
from .minimax import minimax
from .board import Board, StateMinimax

class MinimaxPlayer:
    """
    Minimax player with Alpha-Beta pruning.
    
    This class implements the Alpha-Beta Minimax algorithm with the same interface
    as the Monte Carlo players to be compatible with the API.
    """
    def __init__(self, state, depth=4):
        """
        Initialize the Minimax player.
        
        Args:
            state: Current game state
            depth: Maximum search depth (default: 4)
        """
        self.state = self._convert_state(state)
        self.depth = depth
        self.board = Board()
        
    def get_move(self):
        """
        Get the best move using Alpha-Beta Minimax.
        
        Returns:
            The best move found by the algorithm
        """
        return minimax(self.board, self.state, self.depth)
        
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
            
        # Convert the move format to match the API
        if best_move[0] == CLONE_MOVE:  # Clone move
            from_pos = best_move[2]
            to_pos = best_move[1]
        else:  # Jump move
            from_pos = best_move[2]
            to_pos = best_move[1]
            
        return (from_pos, to_pos)
        
    def _convert_state(self, ataxx_state):
        """
        Convert Ataxx state to StateMinimax.
        
        Args:
            ataxx_state: Ataxx state from the API
            
        Returns:
            StateMinimax object that can be used with the minimax algorithm
        """
        # Convert board format
        board_array = []
        for row in ataxx_state.board:
            board_row = []
            for cell in row:
                # Accept both int and str for compatibility
                if cell == 'red' or cell == 1:
                    board_row.append(1)
                elif cell == 'yellow' or cell == -1:
                    board_row.append(-1)
                else:
                    board_row.append(0)
            board_array.append(board_row)
        
        # Convert player
        player = 1 if ataxx_state.current_player == 'red' else -1
        
        # Count pieces
        red_count = sum(row.count(1) for row in board_array)
        yellow_count = sum(row.count(-1) for row in board_array)
        balls = {1: red_count, -1: yellow_count}
        
        # Create StateMinimax with required arguments
        minimax_state = StateMinimax(board_array, player, balls)
        return minimax_state
