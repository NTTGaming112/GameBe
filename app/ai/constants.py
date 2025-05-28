#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Game constants for Ataxx AI.

This module defines the constants used throughout the Ataxx AI implementation.
"""

# Game constants
BOARD_SIZE = 7  # Standard 7x7 board
PLAYER_ONE = 1  # First player (white/red)
PLAYER_TWO = -1  # Second player (black/yellow)
EMPTY_CELL = 0  # Empty cell

# Move types
CLONE_MOVE = 'c'  # Clone move type
JUMP_MOVE = 'j'  # Jump move type

# Board evaluation constants
WIN_BONUS_FULL_BOARD = 50  # Bonus for winning on a full board
WIN_BONUS_EARLY = 500  # Bonus for winning before board is full

# Position constants
ADJACENT_POSITIONS = [(-1,1),(0,1),(1,1),(-1,0),(1,0),(-1,-1),(0,-1),(1,-1)]
JUMP_POSITIONS = [(-2, 2), (-2, 1), (-2, 0), (-2, -1), (-2, -2), 
             (-1, 2), (-1, -2), (0, 2), (0, -2), (1, 2), 
             (1, -2), (2, 2), (2, 1), (2, 0), (2, -1), (2, -2)]
BOARD_TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE  # Total number of cells (7x7 = 49)

# Minimax move weights (used in minimax.py)
MOVE_WEIGHTS = {
    "capture": 1.2,  # s1: Weight for captured opponent pieces
    "target_surroundings": 0.6,  # s2: Weight for own pieces around target (increased)
    "clone_bonus": 1.5,  # s3: Bonus for clone moves (significantly increased)
    "jump_penalty": 0.4   # s4: Penalty for leaving own pieces around jump source (increased)
}

REPEAT_THRESHOLD = 3  # A position appearing 3 times ends the game

# Monte Carlo algorithm types
MC_TYPE_BASIC = "MC"           # Basic Monte Carlo Tree Search
MC_TYPE_DOMAIN = "MCD"         # Monte Carlo with Domain Knowledge
MC_TYPE_ALPHA_BETA = "AB+MCD"  # Hybrid: Alpha-Beta + Monte Carlo Domain
MC_TYPE_MINIMAX = "MINIMAX"    # Alpha-Beta Minimax

