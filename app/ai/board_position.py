#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module for managing board positions to support the three-repetition rule.
"""

class BoardPositionTracker:
    """
    Class for tracking the history of board positions to detect when
    a position appears for the third time with the same player's turn.
    """
    
    def __init__(self):
        """Initialize the position tracker.
        
        The position history is stored in a dictionary:
        - Key: (board_tuple, player)
        - Value: number of occurrences
        """
        self.position_history = {}
        
    def board_to_tuple(self, board):
        """Convert a 2D board to a 1D tuple for use as a dictionary key.
        
        Args:
            board: 2D array representing the game board
            
        Returns:
            tuple: Flattened representation of the board
        """
        flat_board = []
        for row in board:
            flat_board.extend(row)
        return tuple(flat_board)
    
    def update_position(self, board, player):
        """Update the board position history.
        
        Args:
            board: 2D array representing the game board
            player: Current player (1 or -1)
            
        Returns:
            int: Number of times this position has occurred
        """
        position_key = (self.board_to_tuple(board), player)
        
        if position_key in self.position_history:
            self.position_history[position_key] += 1
        else:
            self.position_history[position_key] = 1
            
        return self.position_history[position_key]
    
    def check_three_repetitions(self, board, player):
        """Check if the current position has appeared three times.
        
        Args:
            board: 2D array representing the game board
            player: Current player (1 or -1)
            
        Returns:
            bool: True if the position has occurred 3+ times, False otherwise
        """
        position_key = (self.board_to_tuple(board), player)
        return self.position_history.get(position_key, 0) >= 3
    
    def reset(self):
        """Reset lịch sử vị trí"""
        self.position_history = {}
