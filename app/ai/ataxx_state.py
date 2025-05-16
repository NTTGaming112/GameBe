#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ataxx game implementation for AI players.

This module provides the Ataxx class which represents the game state
and implements the game rules and mechanics.
"""
import random
import numpy as np
from copy import deepcopy
from collections import deque as dl

from app.ai.constants import (BOARD_SIZE, PLAYER_ONE, PLAYER_TWO, EMPTY_CELL,
                        CLONE_MOVE, JUMP_MOVE, ADJACENT_POSITIONS, JUMP_POSITIONS, REPEAT_THRESHOLD)

class Ataxx:
    """
    Class representing the Ataxx game (also known as Infection or Enhanced Reversi).
    
    Rules:
    - Players can Clone or Jump their pieces
    - Clone: creates a new piece in an adjacent cell, original piece remains
    - Jump: moves an existing piece to a cell 2 steps away, original disappears
    - After each move, opponent pieces adjacent to the new piece are captured
    - A player with no possible moves loses their turn
    - Game ends when board is full or a player has no pieces left
    - The player with more pieces wins
    
    Enhancement:
    - Game also ends if a board position appears for the 3rd time with the same player
    """
    def __init__(self):
        """Initialize a new Ataxx game with standard setup."""
        # Initialize piece counts and move counts
        self.balls = {PLAYER_ONE: 0, PLAYER_TWO: 0}
        self.moves = {PLAYER_ONE: 0, PLAYER_TWO: 0}
        
        # Create empty board
        self.n_fields = BOARD_SIZE
        self.board = [[EMPTY_CELL for x in range(self.n_fields)] for y in range(self.n_fields)]
        
        # Start with player 1
        self.turn_player = PLAYER_ONE

        # Place initial pieces (standard setup: corners)
        self.add_ball(0, 0)  # Top left: Player 1
        self.add_ball(self.n_fields-1, self.n_fields-1)  # Bottom right: Player 1
        
        self.turn_player = PLAYER_TWO
        self.add_ball(0, self.n_fields-1)  # Top right: Player 2
        self.add_ball(self.n_fields-1, 0)  # Bottom left: Player 2
        
        # Reset back to Player 1 for game start
        self.turn_player = PLAYER_ONE

        # Game state tracking
        self.stop_game = False
        
        # Position history for detecting three-time repetition
        # Key: (flattened board tuple, current player)
        # Value: number of times this position has occurred
        self.position_history = {}

    def get_all_possible_moves(self):
        """Get all legal moves for the current player.
        
        Returns:
            list: List of possible moves in the format:
                - (CLONE_MOVE, (x, y)) for Clone moves
                - (JUMP_MOVE, (x_dest, y_dest), (x_src, y_src)) for Jump moves
        """
        possible_moves = []
        
        # Check each cell on the board
        for x in range(self.n_fields):
            for y in range(self.n_fields):
                # Skip cells that don't contain current player's piece
                if self.board[x][y] != self.turn_player:
                    continue
                    
                source_pos = (x, y)
                
                # Add all valid Clone moves
                clone_targets = self.get_empty_pos(source_pos, ADJACENT_POSITIONS)
                possible_moves.extend([(CLONE_MOVE, target) for target in clone_targets])
                
                # Add all valid Jump moves
                jump_targets = self.get_empty_pos(source_pos, JUMP_POSITIONS)
                possible_moves.extend([(JUMP_MOVE, target, source_pos) for target in jump_targets])

        return possible_moves

    def update_board(self, state):
        """Update this board with state from an external state object.
        
        Args:
            state: External state object with board, player and piece counts
        """
        self.board = state.board
        # Support both 'player' and 'turn_player' attributes for compatibility
        self.turn_player = getattr(state, "player", getattr(state, "turn_player", None))
        self.balls[PLAYER_ONE] = state.balls[PLAYER_ONE]
        self.balls[PLAYER_TWO] = state.balls[PLAYER_TWO]

    def toggle_player(self):
        """Switch to the other player's turn."""
        self.turn_player = -self.turn_player

    def current_player(self):
        """Get the current player.
        
        Returns:
            int: PLAYER_ONE (1) or PLAYER_TWO (-1)
        """
        return self.turn_player

    def move_with_position(self, position):
        """Execute a move at the specified position.
        
        Args:
            position: Move specification:
                - (CLONE_MOVE, (x, y)) for Clone move
                - (JUMP_MOVE, (x_dest, y_dest), (x_src, y_src)) for Jump move
        """
        if position[0] == CLONE_MOVE:
            self.copy_stone_position(position[1])
        else:
            self.jump_stone_position(position[1], position[2])
            
        # Update position history after each move
        self.update_position_history()

    def copy_stone_position(self, target_pos):
        """Execute a Clone move to the target position.
        
        Args:
            target_pos: (x, y) tuple of the target position
        """
        x, y = target_pos
        self.add_ball(x, y)
        self.take_stones(x, y)
        self.increase_move()

    def jump_stone_position(self, target_pos, source_pos):
        """Execute a Jump move from source to target position.
        
        Args:
            target_pos: (x, y) tuple of the target position
            source_pos: (x, y) tuple of the source position
        """
        x, y = target_pos
        self.add_ball(x, y)
        self.remove_ball(source_pos[0], source_pos[1])
        self.take_stones(x, y)
        self.increase_move()

    def move(self):
        """Make a random move for the current player.
        
        Returns:
            bool: True if a move was made, False if no moves were available
        """
        moves = self.get_all_possible_moves()
        if not moves:
            return False
            
        idx = np.random.randint(0, len(moves))
        self.move_with_position(moves[idx])
        return True

    def increase_move(self):
        """Increment the move counter for the current player."""
        self.moves[self.turn_player] += 1

    def get_amount_moves(self):
        """Get the total number of moves made in the game.
        
        Returns:
            int: Total number of moves by both players
        """
        return self.moves[PLAYER_ONE] + self.moves[PLAYER_TWO]

    def get_empty_pos(self, base_pos, directions):
        """Get list of empty positions relative to a base position.
        
        Args:
            base_pos: (x, y) tuple of the base position
            directions: List of relative (dx, dy) offsets to check
            
        Returns:
            list: List of valid empty positions
        """
        x_base, y_base = base_pos
        return [(x_base + dx, y_base + dy) for dx, dy in directions 
                if 0 <= x_base + dx < self.n_fields 
                and 0 <= y_base + dy < self.n_fields 
                and self.is_empty(x_base + dx, y_base + dy)]

    def get_full_pos(self, base_pos, directions):
        """Get list of occupied positions relative to a base position.
        
        Args:
            base_pos: (x, y) tuple of the base position
            directions: List of relative (dx, dy) offsets to check
            
        Returns:
            list: List of valid occupied positions
        """
        x_base, y_base = base_pos
        return [(x_base + dx, y_base + dy) for dx, dy in directions 
                if 0 <= x_base + dx < self.n_fields 
                and 0 <= y_base + dy < self.n_fields 
                and not self.is_empty(x_base + dx, y_base + dy)]

    def get_copy_position(self, base_pos):
        """Get a random valid clone position from a base position.
        
        Args:
            base_pos: (x, y) tuple of the base position
            
        Returns:
            tuple: (x, y) of the target position, or base_pos if no valid targets
        """
        valid_targets = self.get_empty_pos(base_pos, ADJACENT_POSITIONS)

        if not valid_targets:
            return base_pos

        return valid_targets[np.random.randint(0, len(valid_targets))]

    def get_jump_position(self, base_pos):
        """Get a random valid jump position from a base position.
        
        Args:
            base_pos: (x, y) tuple of the base position
            
        Returns:
            tuple: (x, y) of the target position, or base_pos if no valid targets
        """
        valid_targets = self.get_empty_pos(base_pos, JUMP_POSITIONS)

        if not valid_targets:
            return base_pos

        return valid_targets[np.random.randint(0, len(valid_targets))]

    def choose_ball(self):
        """Randomly select one of the current player's pieces.
        
        Returns:
            list: [x, y] coordinates of the selected piece
        """
        if self.balls[self.turn_player] <= 0:
            return None
            
        idx = np.random.randint(1, self.balls[self.turn_player] + 1)
        counter = 1
        
        for x in range(self.n_fields):
            for y in range(self.n_fields):
                if self.board[x][y] == self.turn_player:
                    if counter == idx:
                        return [x, y]
                    counter += 1
        
        return None  # Should never reach here if counts are correct

    def copy_stone(self):
        """Execute a random Clone move for current player."""
        base_pos = self.choose_ball()
        if not base_pos:
            return
            
        x, y = base_pos
        target_x, target_y = self.get_copy_position(base_pos)
        
        # Only proceed if we found a valid target different from source
        if [target_x, target_y] != base_pos:
            self.add_ball(target_x, target_y)
            self.take_stones(target_x, target_y)
            self.increase_move()

    def jump_stone(self):
        """Execute a random Jump move for current player."""
        base_pos = self.choose_ball()
        if not base_pos:
            return
            
        target_x, target_y = self.get_jump_position(base_pos)
        
        # Only proceed if we found a valid target different from source
        if [target_x, target_y] != base_pos:
            self.add_ball(target_x, target_y)
            self.remove_ball(base_pos[0], base_pos[1])
            self.take_stones(target_x, target_y)
            self.increase_move()

    def take_stones(self, x, y):
        """Capture opponent's pieces adjacent to (x, y).
        
        Args:
            x: X-coordinate of the newly placed piece
            y: Y-coordinate of the newly placed piece
        """
        occupied_neighbors = self.get_full_pos([x, y], ADJACENT_POSITIONS)
        
        for nx, ny in occupied_neighbors:
            # If the piece belongs to the opponent, capture it
            if self.board[nx][ny] == -self.turn_player:
                self.change_ball_player(nx, ny)

    def change_ball_player(self, x, y):
        """Convert a piece at (x, y) to the current player's color.
        
        Args:
            x: X-coordinate of the piece to convert
            y: Y-coordinate of the piece to convert
        """
        # Flip the piece's owner
        self.board[x][y] = -self.board[x][y]
        
        # Update piece counts
        self.balls[self.turn_player] += 1
        self.balls[-self.turn_player] -= 1
        
        # Sanity check
        assert self.balls[PLAYER_ONE] >= 0
        assert self.balls[PLAYER_TWO] >= 0

    def is_empty(self, x, y):
        """Check if a cell is empty.
        
        Args:
            x: X-coordinate of the cell
            y: Y-coordinate of the cell
            
        Returns:
            bool: True if the cell is empty, False otherwise
        """
        return self.board[x][y] == EMPTY_CELL

    def add_ball(self, x, y):
        """Add a piece of the current player at (x, y).
        
        Args:
            x: X-coordinate where to add
            y: Y-coordinate where to add
        """
        assert self.board[x][y] == EMPTY_CELL
        self.board[x][y] = self.turn_player
        self.balls[self.turn_player] += 1

    def remove_ball(self, x, y):
        """Remove a piece at (x, y).
        
        Args:
            x: X-coordinate where to remove
            y: Y-coordinate where to remove
        """
        assert self.board[x][y] != EMPTY_CELL
        self.board[x][y] = EMPTY_CELL
        self.balls[self.turn_player] -= 1

    def full_squares(self):
        """Check if the board is full.
        
        Returns:
            bool: True if no empty cells remain, False otherwise
        """
        total_pieces = self.balls[PLAYER_ONE] + self.balls[PLAYER_TWO]
        return total_pieces >= self.n_fields * self.n_fields

    def is_game_over(self):
        """Check if the game has ended.
        
        Returns:
            bool: True if the game is over, False otherwise
        """
        # Manually stopped game
        if self.stop_game:
            return True
            
        # One player has no pieces left
        if self.balls[PLAYER_ONE] == 0 or self.balls[PLAYER_TWO] == 0:
            return True
            
        # Board is full
        if self.full_squares():
            return True
            
        # Position repeated three times
        if self.position_repeated_three_times():
            return True
            
        return False

    def print_winner(self):
        """Print the winner of the game to the console."""
        if self.balls[PLAYER_ONE] > self.balls[PLAYER_TWO]:
            print("Winner: White (Player 1)")
        elif self.balls[PLAYER_TWO] > self.balls[PLAYER_ONE]:
            print("Winner: Black (Player 2)")
        else:
            print("Draw")

    def get_winner(self):
        """Get the winner of the game.
        
        Returns:
            int: PLAYER_ONE (1) if white wins, PLAYER_TWO (-1) if black wins,
                 100 for a draw
        """
        if self.balls[PLAYER_TWO] > self.balls[PLAYER_ONE]:
            return PLAYER_TWO
        elif self.balls[PLAYER_ONE] > self.balls[PLAYER_TWO]:
            return PLAYER_ONE
        else:
            return 100  # Draw

    def get_winner_without_gameover(self):
        """Get the current leader without checking if game is over.
        
        Returns:
            int: PLAYER_ONE (1) if white leads, PLAYER_TWO (-1) if black leads,
                 0 for a tie
        """
        if self.balls[PLAYER_TWO] > self.balls[PLAYER_ONE]:
            return PLAYER_TWO
        elif self.balls[PLAYER_ONE] > self.balls[PLAYER_TWO]:
            return PLAYER_ONE
        else:
            return 0  # Tie

    def get_score(self, player):
        """Get the score ratio for a player.
        
        Args:
            player: Player to get score for (PLAYER_ONE or PLAYER_TWO)
            
        Returns:
            float: Ratio of player's pieces to total pieces (0 to 1)
        """
        assert self.balls[PLAYER_ONE] >= 0
        assert self.balls[PLAYER_TWO] >= 0
        total_pieces = self.balls[PLAYER_ONE] + self.balls[PLAYER_TWO]
        
        if total_pieces == 0:
            return 0.0
            
        return float(self.balls[player]) / total_pieces

    def get_score_pieces(self, player):
        """Get the board coverage ratio for a player.
        
        Args:
            player: Player to get score for (PLAYER_ONE or PLAYER_TWO)
            
        Returns:
            float: Ratio of player's pieces to total board cells (0 to 1)
        """
        # Ensure we don't divide by zero
        total_cells = self.n_fields ** 2
        if total_cells == 0:
            return 0.0
            
        return float(self.balls[player]) / total_cells
        
    def board_to_tuple(self):
        """Convert 2D board to a 1D tuple for dictionary storage.
        
        Returns:
            tuple: Flattened representation of the board
        """
        flat_board = []
        for row in self.board:
            flat_board.extend(row)
        return tuple(flat_board)
        
    def update_position_history(self):
        """Update the history of board positions."""
        position_key = (self.board_to_tuple(), self.turn_player)
        
        if position_key in self.position_history:
            self.position_history[position_key] += 1
        else:
            self.position_history[position_key] = 1
            
    def position_repeated_three_times(self):
        """Check if current position has occurred three times.
        
        Returns:
            bool: True if position repeated 3+ times, False otherwise
        """
        position_key = (self.board_to_tuple(), self.turn_player)
        return self.position_history.get(position_key, 0) >= REPEAT_THRESHOLD

    def print_board(self):
        """Print the current board state to console."""
        print(f"Player: {self.turn_player}")
        s = [[str(e) for e in row] for row in self.board]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
        print(f"White pieces: {self.balls[PLAYER_ONE]} | Black pieces: {self.balls[PLAYER_TWO]}")

    def show_board(self):
        """Get string representation of the current board state.
        
        Returns:
            str: Formatted string representation of the board
        """
        msg = f"Player: {self.turn_player}\n"
        s = [[str(e) for e in row] for row in self.board]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        msg += '\n'.join(table)
        msg += f"\nWhite pieces: {self.balls[PLAYER_ONE]}"
        msg += f" | Black pieces: {self.balls[PLAYER_TWO]}"
        return msg

    def play(self):
        """Play a full game automatically using random moves.
        
        The game continues until a terminal state is reached.
        """
        no_moves_count = 0  # Count consecutive turns with no legal moves
        
        while not self.is_game_over():
            # Try to make a move
            if not self.move():
                # No legal moves, increment counter
                no_moves_count += 1
                if no_moves_count >= 2:
                    # Both players have no moves, end the game
                    self.stop_game = True
                    break
            else:
                # Move made successfully, reset counter
                no_moves_count = 0
                
            # Switch to the other player
            self.toggle_player()
