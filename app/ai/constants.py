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



# =================== MONTE CARLO DOMAIN CONSTANTS ===================

# Tournament System Configuration
TOURNAMENT_CONFIG = {
    'K1': 8,  # Top moves advance from Round 1
    'K2': 4,  # Top moves advance from Round 2
    'ROUND1_SIM_RATIO': 1.0,  # Round 1: 100% of base_simulations
    'ROUND2_SIM_RATIO': 1.5,  # Round 2: 150% of base_simulations
    'ROUND3_SIM_RATIO': 2.0,  # Round 3: 200% of base_simulations
    'PARALLEL_THRESHOLD': 6,
    'MAX_WORKERS': 4,
    'SIGMOID_FACTOR': 4.0
}

# Temperature schedule for softmax move selection
TEMPERATURE_SCHEDULE = {
    'early': 2.0,   # High exploration in early
    'mid': 1.0,   # Balanced exploration/exploitation  
    'late': 0.1    # Low exploration in late (more deterministic)
}

# Component weights for comprehensive move evaluation
COMPONENT_WEIGHTS = {
    'heuristic': 1.0,   # Weight for S(m,s,p) heuristic formula
    'tactical': 1.0,    # Weight for tactical evaluation (safety, corners, captures)
    'strategic': 1.0    # Weight for strategic evaluation (tempo, mobility, connectivity)
}

# Phase-adaptive Heuristic Formula Coefficients
PHASE_ADAPTIVE_HEURISTIC_COEFFS = {
    'early': {
        's1': 1.0,  # Lower capture weight in early game (focus on expansion)
        's2': 1.2,  # Higher attack weight for territory control
        's3': 2.0,  # High clone bonus for expansion
        's4': 1.0   # Lower jump penalty for mobility
    },
    'mid': {
        's1': 1.2,  # Balanced capture weight
        's2': 0.8,  # Moderate attack weight
        's3': 1.5,  # Moderate clone bonus
        's4': 2.0   # Standard jump penalty
    },
    'late': {
        's1': 1.5,  # Higher capture weight in endgame
        's2': 0.6,  # Lower attack weight (focus on captures)
        's3': 1.0,  # Lower clone bonus (efficiency over expansion)
        's4': 2.5   # Higher jump penalty (avoid risky moves)
    }
}

# Phase-adaptive Clone Bonus and Jump Penalty
PHASE_BONUS_PENALTY = {
    'early': {
        'clone_bonus': 1.2,    # High bonus for expansion in early game
        'jump_penalty': 0.2    # Low penalty for mobility in early game
    },
    'mid': {
        'clone_bonus': 0.8,    # Moderate bonus in mid game
        'jump_penalty': 0.4    # Moderate penalty in mid game
    },
    'late': {
        'clone_bonus': 0.5,    # Lower bonus in late game (focus on efficiency)
        'jump_penalty': 0.6    # Higher penalty in late game (avoid risky jumps)
    }
}

# Phase-adaptive Component Weights (Tournament System)
MCD_PHASE_WEIGHTS = {
    'early': {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.2, 'delta': 0.1},
    'mid': {'alpha': 0.3, 'beta': 0.3, 'gamma': 0.3, 'delta': 0.1}, 
    'late': {'alpha': 0.2, 'beta': 0.2, 'gamma': 0.2, 'delta': 0.4}
}