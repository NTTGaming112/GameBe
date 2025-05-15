#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Board module for Ataxx AI.

This module provides game state representation and game mechanics for AI algorithms.
It wraps the Ataxx class to provide an interface for AI algorithms to interact with the game.
"""
from copy import deepcopy
from .ataxx_state import Ataxx, PLAYER_ONE, PLAYER_TWO

class StateMinimax:
    """Game state representation for minimax algorithm.
    
    This class encapsulates the board state, current player, and piece counts
    for use with minimax and other AI algorithms.
    """
    def __init__(self, board, player, balls):
        """Initialize a state for minimax algorithm.
        
        Args:
            board: 2D board array
            player: Current player (1 or -1)
            balls: Dictionary of piece counts for each player
        """
        self.board = deepcopy(board)
        self.player = player
        self.balls = {
            PLAYER_ONE: balls[PLAYER_ONE],
            PLAYER_TWO: balls[PLAYER_TWO]
        }

    def print_state(self):
        """Print the current state to console."""
        print(f"Player: {self.player}")
        s = [[str(e) for e in row] for row in self.board]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))

    def __str__(self):
        """String representation of state for debugging."""
        return f"{self.player} - {self.board} P1:{self.balls[PLAYER_ONE]} P2:{self.balls[PLAYER_TWO]}"

    def __hash__(self):
        """Hash function for using state in dictionaries and sets."""
        return hash(str(self))

    def __eq__(self, other):
        """Equality check for two states."""
        return (self.player == other.player and 
                self.board == other.board and 
                self.balls[PLAYER_TWO] == other.balls[PLAYER_TWO] and 
                self.balls[PLAYER_ONE] == other.balls[PLAYER_ONE])


class Board:
    """Board wrapper providing an interface for AI algorithms.
    
    This class acts as a bridge between the Ataxx game mechanics and
    AI search algorithms by providing methods to generate and apply moves,
    check for game end conditions, and evaluate game states.
    """
    def __init__(self):
        """Initialize a new Board instance with an Ataxx game."""
        self.game = Ataxx()

    def current_player(self, state):
        """Get the current player.
        
        Args:
            state: Current game state
            
        Returns:
            int: Current player (1 or -1)
        """
        return state.player

    def next_state(self, state, play):
        """Apply a move to the current state and return the new state.
        
        Args:
            state: Current game state
            play: Move to apply
            
        Returns:
            StateMinimax: New game state after applying the move
        """
        # Create a new state object from the current state
        new_state = StateMinimax(state.board, state.player, state.balls)

        # Update the game with the current state
        self.game.update_board(new_state)

        # Apply the move and switch players
        self.game.move_with_position(play)
        self.game.turn_player = -self.game.turn_player
        
        # Update the new state with the game changes
        new_state.player = self.game.turn_player
        new_state.balls[PLAYER_TWO] = self.game.balls[PLAYER_TWO]
        new_state.balls[PLAYER_ONE] = self.game.balls[PLAYER_ONE]
        
        return new_state

    def legal_plays(self, state):
        """Get all legal moves for the current player.
        
        Args:
            state: Current game state
            
        Returns:
            list: All valid moves for the current player
        """
        self.game.update_board(state)
        return self.game.get_all_possible_moves()

    def is_gameover(self, state):
        """Check if the game has ended.
        
        Args:
            state: Current game state
            
        Returns:
            bool: True if the game is over, False otherwise
        """
        self.game.update_board(state)
        return self.game.is_game_over()

    def winner(self, state):
        """Get the winner of the game.
        
        Args:
            state: Current game state
            
        Returns:
            int: PLAYER_ONE (1) if white wins, PLAYER_TWO (-1) if black wins,
                 0 if the game is ongoing, different value for a draw
        """
        self.game.update_board(state)
        if not self.game.is_game_over():
            return 0

        return self.game.get_winner()

    def get_score(self, state, player):
        """Get the piece ratio score for a player.
        
        Args:
            state: Current game state
            player: Player to get score for (PLAYER_ONE or PLAYER_TWO)
            
        Returns:
            float: Ratio of player's pieces to total pieces (0 to 1)
        """
        self.game.update_board(state)
        return self.game.get_score(player)

    def get_score_pieces(self, state, player):
        """Get the board coverage ratio for a player.
        
        Args:
            state: Current game state
            player: Player to get score for (PLAYER_ONE or PLAYER_TWO)
            
        Returns:
            float: Ratio of player's pieces to total board cells (0 to 1)
        """
        self.game.update_board(state)
        return self.game.get_score_pieces(player)