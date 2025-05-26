#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search base implementation for Ataxx AI.

This module provides the core Monte Carlo Tree Search algorithm with random rollouts,
which serves as the foundation for more advanced Monte Carlo variants.
"""
import random
import math
import time
from copy import deepcopy

from app.ai.constants import WIN_BONUS_EARLY, WIN_BONUS_FULL_BOARD

class MonteCarloNode:
    """Base node class for Monte Carlo Tree Search.
    
    This class represents a node in the Monte Carlo search tree, containing
    the game state, statistics, and links to parent and children nodes.
    """
    def __init__(self, state, parent=None, move=None):
        """Initialize a Monte Carlo search tree node.
        
        Args:
            state: Game state at this node
            parent: Parent node (None for root)
            move: Move that led to this state
        """
        self.state = deepcopy(state)
        self.parent = parent
        self.move = move  # Move that led to this state
        self.children = []
        self.ep = 0
        self.visits = 0
        self.untried_moves = state.get_all_possible_moves() if not state.is_game_over() else []
        
    def uct_value(self, c=1.414):
        """Calculate UCT (Upper Confidence Bound applied to Trees) value.
        
        Args:
            c: Exploration parameter (default: sqrt(2))
            
        Returns:
            float: UCT value of this node
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.ep / self.visits
        exploration = c * math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_child(self):
        """Select child with highest UCT value.
        
        Returns:
            MonteCarloNode: Child node with highest UCT value
        """
        return max(self.children, key=lambda c: c.uct_value())
    
    def expand(self):
        """Expand the search tree by creating a new child node.
        
        Returns:
            MonteCarloNode: Newly created child node, or None if no untried moves
        """
        if not self.untried_moves:
            return None
        
        move = self.untried_moves.pop(0)
        next_state = deepcopy(self.state)
        next_state.move_with_position(move)
        next_state.toggle_player()
        
        child = MonteCarloNode(next_state, self, move)
        self.children.append(child)
        return child
    
    def update(self, result):
        """Update node statistics after a simulation.
        
        Args:
            result: Simulation result (typically 0 or 1)
        """
        self.visits += 1
        self.ep += result

class MonteCarloBase:
    """Basic Monte Carlo Tree Search with random rollouts.
    
    This class implements the basic Monte Carlo Tree Search algorithm
    using pure random rollouts without domain-specific knowledge.
    """
    def __init__(self, state, **kwargs):
        """Initialize Basic Monte Carlo player.
        
        Args:
            state: Initial game state
            **kwargs: Configuration parameters including:
                - basic_simulations: Base number of simulations (default: 300)
                - exploration: Exploration parameter (default: 1.414)
                - max_time: Maximum search time in seconds (default: 1.0)
                - tournament_sizes: Tournament simulation sizes (default: [600, 600, 300])
                - use_simulation_formula: Whether to use simulation scaling (default: True)
                - time_limit: Time limit for move calculation (default: 50 seconds)
        """
        self.root_state = state
        self.basic_simulations = kwargs.get('basic_simulations', 300)
        self.exploration = kwargs.get('exploration', 1.414)
        self.max_time = kwargs.get('max_time', 1.0)
        self.tournament_sizes = kwargs.get('tournament_sizes', [600, 600, 300])
        self.use_simulation_formula = kwargs.get('use_simulation_formula', True)
        self.time_limit = kwargs.get('time_limit', 50)
        
    def calculate_simulations(self, state):
        """Calculate number of simulations based on board state.
        
        Uses formula Stotal = Sbasic * (1 + 0.1 * nfilled) which increases
        rollouts as the board fills up.
        
        With Sbasic = {300, 600, 1200} depending on configuration.
        
        Examples:
        - With Sbasic=300, empty board: 300 simulations
        - With Sbasic=300, half-filled board: 300 * (1 + 0.1*24.5) ≈ 345 simulations
        - With Sbasic=300, full board: 300 * (1 + 0.1*49) = 447 simulations
        
        Args:
            state: Current game state
            
        Returns:
            int: Number of simulations to run
        """
        # nfilled is the total number of pieces on the board
        total_pieces = state.balls[1] + state.balls[-1]
        
        # If using formula, apply Stotal = Sbasic * (1 + 0.1 * nfilled)
        if self.use_simulation_formula:
            return int(self.basic_simulations * (1 + 0.1 * total_pieces))
        # Otherwise, return Sbasic
        else:
            return self.basic_simulations
        
    def calculate_tournament_simulations(self, state):
        """Calculate simulation counts for each tournament round based on board state.
        
        When use_simulation_formula=True:
            Applies formula Stotal = Sbasic * (1 + 0.1 * nfilled) to each tournament round.
        When use_simulation_formula=False:
            Uses fixed values from tournament_sizes.
            
        Default configuration for MCD: (S1, S2, S3) = (600, 600, 300)
        
        Args:
            state: Current game state
            
        Returns:
            list: List [S1, S2, S3] of simulation counts for each tournament round
        """
        # Get base values for each tournament round
        # Default: (S1, S2, S3) = (600, 600, 300)
        S1_base, S2_base, S3_base = self.tournament_sizes
        
        # If using formula, apply Stotal = Sbasic * (1 + 0.1 * nfilled) to each round
        if self.use_simulation_formula:
            # Calculate total pieces on board (nfilled)
            total_pieces = state.balls[1] + state.balls[-1]
            
            S1 = int(S1_base * (1 + 0.1 * total_pieces))
            S2 = int(S2_base * (1 + 0.1 * total_pieces))
            S3 = int(S3_base * (1 + 0.1 * total_pieces))
        # Otherwise, use fixed values from tournament_sizes
        else:
            S1, S2, S3 = S1_base, S2_base, S3_base
        
        # Make sure no simulation count is less than 1
        S1 = max(1, S1)
        S2 = max(1, S2)
        S3 = max(1, S3)
        
        return [S1, S2, S3]
    
    def get_play(self):
        """Alias for get_move().
        
        Returns:
            Move: Best move found by the algorithm
        """
        return self.get_move()
        
    def get_move(self, time_limit=None):
        """Find the best move using Monte Carlo Tree Search with time limit support."""
        if time_limit is None:
            time_limit = self.time_limit
        start_time = time.time()
        root = MonteCarloNode(self.root_state)
        simulations = self.calculate_simulations(self.root_state)
        simulation_count = 0
        
        # Handle case where time_limit is None - run all simulations
        if time_limit is None:
            time_condition = True  # Always True, only check simulation count
        else:
            time_condition = (time.time() - start_time) < time_limit
            
        while simulation_count < simulations and time_condition:
            # 1. Selection
            node = root
            state = deepcopy(self.root_state)
            
            # Traverse down the tree until reaching a node with untried moves or a leaf
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
                state.move_with_position(node.move)
                state.toggle_player()
            
            # 2. Expansion
            if node.untried_moves != []:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                
                next_state = deepcopy(state)
                next_state.move_with_position(move)
                next_state.toggle_player()
                
                child = MonteCarloNode(next_state, parent=node, move=move)
                node.children.append(child)
                node = child
            
            # 3. Simulation and 4. Backpropagation
            result = self._simulate(state)
            while node:
                node.update(result)
                node = node.parent
                result = 1 - result  # Flip result for each level
                
            simulation_count += 1
            
            # Update time condition if time_limit is not None
            if time_limit is not None:
                time_condition = (time.time() - start_time) < time_limit
        
        # Choose best move based on visit count
        if not root.children:
            return None
            
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
        
    def _evaluate_final_position(self, state, player):
        """Evaluate final position using a unified evaluation function.
        
        Evaluation function: E(p) = Nown - Nopp
        - Win with full board: E(p) + 50
        - Win before full board: E(p) + 500
        - Loss with full board: E(p) - 50
        - Loss before full board: E(p) - 500
        
        Args:
            state: Game state to evaluate
            player: Player perspective to evaluate from
            
        Returns:
            float: Evaluation score converted to [0, 1] range for Monte Carlo
        """
        opponent = -player
        
        # Count pieces for each side
        num_own = state.balls[player]
        num_opp = state.balls[opponent]
        
        # Basic evaluation
        score = num_own - num_opp
        
        # If game is over, add bonus/penalty
        if state.is_game_over():
            if num_own > num_opp:  # Win
                # Check if board is full or not
                total_pieces = num_own + num_opp
                empty_spaces = state.n_fields**2 - total_pieces  # 7x7 board has 49 cells
                if empty_spaces == 0:  # Full board
                    score += WIN_BONUS_FULL_BOARD
                else:  # Win before full board
                    score += WIN_BONUS_EARLY
            elif num_own < num_opp:  # Loss
                total_pieces = num_own + num_opp
                empty_spaces = state.n_fields**2 - total_pieces
                if empty_spaces == 0:  # Full board
                    score -= WIN_BONUS_FULL_BOARD
                else:  # Loss before full board
                    score -= WIN_BONUS_EARLY

        # Normalize score to [0, 1] range     
        return (score + 549) / 1098
        
    def _simulate(self, state):
        """Simulate a random game from the current state until game end.
        
        Args:
            state: Starting game state
            
        Returns:
            float: Evaluation score in [0, 1] range
        """
        state = deepcopy(state)
        player = state.current_player()
        
        while not state.is_game_over():
            moves = state.get_all_possible_moves()
            if not moves:
                break
            
            # Choose a random move
            move = random.choice(moves)
            state.move_with_position(move)
            state.toggle_player()
        
        return self._evaluate_final_position(state, player)