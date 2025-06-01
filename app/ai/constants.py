#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Game constants for Ataxx AI.

This module defines the constants used throughout the Ataxx AI implementation.
"""

# Game constants
BOARD_SIZE = 7
BOARD_TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE
PLAYER_ONE = 1
PLAYER_TWO = -1
EMPTY_CELL = 0
REPEAT_THRESHOLD = 3

# Move types
CLONE_MOVE = 'c'  # Clone move type
JUMP_MOVE = 'j'  # Jump move type

# Board evaluation constants
WIN_BONUS_FULL_BOARD = 50  # Bonus for winning on a full board
WIN_BONUS_EARLY = 500  # Bonus for winning before board is full

# Position constants
ADJACENT_POSITIONS = [
    (-1, 1), (0, 1), (1, 1),
    (-1, 0),         (1, 0),
    (-1, -1), (0, -1), (1, -1)
]

JUMP_POSITIONS = [
    (-2, 2), (-2, 1), (-2, 0), (-2, -1), (-2, -2),
    (-1, 2),                     (-1, -2),
    (0, 2),                       (0, -2),
    (1, 2),                       (1, -2),
    (2, 2), (2, 1), (2, 0), (2, -1), (2, -2)
]

# Move weights for evaluation
MOVE_WEIGHTS = {
    "capture": 1.2,  # s1: Weight for captured opponent pieces
    "target_surroundings": 0.5,  # s2: Weight for own pieces around target (increased)
    "clone_bonus": 0.8,  # s3: Bonus for clone moves (significantly increased)
    "jump_penalty": 0.3   # s4: Penalty for leaving own pieces around jump source (increased)
}

REPEAT_THRESHOLD = 3  # A position appearing 3 times ends the game

# Monte Carlo algorithm types
MC_TYPE_BASIC = "MC"           # Basic Monte Carlo Tree Search
MC_TYPE_DOMAIN = "MCD"         # Monte Carlo with Domain Knowledge
MC_TYPE_ALPHA_BETA = "AB+MCD"  # Hybrid: Alpha-Beta + Monte Carlo Domain
MC_TYPE_MINIMAX = "MINIMAX"    # Alpha-Beta Minimax

# Phase-based dynamic weights for heuristic formula S(m,s,p) = s1×C + s2×A + s3×B - s4×P
PHASE_WEIGHTS = {
    'opening': {'s1': 0.8, 's2': 1.2, 's3': 0.6, 's4': 1.0},
    'midgame': {'s1': 1.0, 's2': 1.0, 's3': 1.0, 's4': 1.0},
    'endgame': {'s1': 1.2, 's2': 0.8, 's3': 1.4, 's4': 1.2}
}

# Temperature schedule for softmax move selection
TEMPERATURE_SCHEDULE = {
    'opening': 2.0,   # High exploration in opening
    'midgame': 1.0,   # Balanced exploration/exploitation  
    'endgame': 0.1    # Low exploration in endgame (more deterministic)
}

# Component weights for comprehensive move evaluation
COMPONENT_WEIGHTS = {
    'heuristic': 1.0,   # Weight for S(m,s,p) heuristic formula
    'tactical': 1.0,    # Weight for tactical evaluation (safety, corners, captures)
    'strategic': 1.0    # Weight for strategic evaluation (tempo, mobility, connectivity)
}

