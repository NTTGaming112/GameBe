#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Domain Knowledge-enhanced Monte Carlo for Ataxx AI.

This module extends the basic Monte Carlo approach by incorporating domain knowledge
via tournament layering and probability-based move selection during rollouts.
"""
import random
from copy import deepcopy

from app.ai.constants import CLONE_MOVE, MOVE_WEIGHTS
from .monte_carlo_base import MonteCarloBase

class MonteCarloDomain(MonteCarloBase):
    """Monte Carlo with Domain Knowledge.
    
    This class implements an enhanced Monte Carlo algorithm that uses domain knowledge
    to improve move selection in both the tree search and simulation phases.
    """
    def __init__(self, state, **kwargs):
        """Initialize Monte Carlo with Domain Knowledge player.
        
        Args:
            state: Initial game state
            **kwargs: Configuration parameters including:
                - tournament_rounds: Number of tournament rounds (default: 3)
                - tournament_sizes: List of simulation counts [S1, S2, S3] for each round
                  Default is [600, 600, 300]
        """
        # Tournament parameters
        self.tournament_rounds = kwargs.get('tournament_rounds', 3)
        # Default (S1, S2, S3) = (600, 600, 300)
        self.tournament_sizes = kwargs.get('tournament_sizes', [600, 600, 300])
        super().__init__(state, **kwargs)
    
    def get_play(self):
        """Alias for get_move().
        
        Returns:
            Move: Best move found by the algorithm
        """
        return self.get_move()
    
    def get_move(self, time_limit=50):
        """Use Tournament Layering to select the best move, with optional time limit (seconds).
        
        Tournament Layering is a technique where moves are filtered through
        multiple rounds of evaluation, with each round using more simulations
        for a smaller set of candidate moves.
        
        Args:
            time_limit (float, optional): Maximum time in seconds to spend on move selection.
        
        Returns:
            Move: Best move found, or None if no legal moves
        """
        import time as _time
        moves = self.root_state.get_all_possible_moves()
        if not moves or len(moves) == 1:
            return moves[0] if moves else None
            
        # Tournament Layering configuration
        k1 = min(5, len(moves))       # Number of moves to keep after round 1
        k2 = min(3, k1)               # Number of moves to keep after round 2
        
        # Calculate simulation counts for each tournament round
        # S1, S2, S3 depend on use_simulation_formula
        tournament_sizes = self.calculate_tournament_simulations(self.root_state)
        S1, S2, S3 = tournament_sizes  # Simulations per round
        
        start_time = _time.time()
        # Round 1: Run S1 rollouts for each move
        print(f"Tournament round 1: Evaluating {len(moves)} moves with {S1} simulations each")
        move_scores = []
        for move in moves:
            if time_limit and _time.time() - start_time > time_limit:
                
                break
            next_state = deepcopy(self.root_state)
            next_state.move_with_position(move)
            next_state.toggle_player()
            score = self._evaluate_move(next_state, S1)
            move_scores.append((move, score))
        
        # Sort and select top k1 moves
        move_scores.sort(key=lambda x: x[1], reverse=True)
        candidates = move_scores[:k1]
        print(f"Round 1 results: Selected top {len(candidates)} moves")
        
        # Round 2: Run S2 additional rollouts for top k1 moves
        if len(candidates) > 1:
            print(f"Tournament round 2: Evaluating {len(candidates)} moves with {S2} simulations each")
            new_scores = []
            
            for move, prev_score in candidates:
                if time_limit and _time.time() - start_time > time_limit:
                    break
                next_state = deepcopy(self.root_state)
                next_state.move_with_position(move)
                next_state.toggle_player()
                additional_score = self._evaluate_move(next_state, S2)
                combined_score = (prev_score * S1 + additional_score * S2) / (S1 + S2)
                new_scores.append((move, combined_score))
            
            new_scores.sort(key=lambda x: x[1], reverse=True)
            candidates = new_scores[:k2]
            print(f"Round 2 results: Selected top {len(candidates)} moves")
        
        # Round 3: Run S3 additional rollouts for top k2 moves
        if len(candidates) > 1:
            print(f"Tournament round 3: Evaluating {len(candidates)} moves with {S3} simulations each")
            final_scores = []
            
            for move, prev_score in candidates:
                if time_limit and _time.time() - start_time > time_limit:
                    break
                next_state = deepcopy(self.root_state)
                next_state.move_with_position(move)
                next_state.toggle_player()
                additional_score = self._evaluate_move(next_state, S3)
                combined_score = (prev_score * (S1 + S2) + additional_score * S3) / (S1 + S2 + S3)
                final_scores.append((move, combined_score))
            
            final_scores.sort(key=lambda x: x[1], reverse=True)
            candidates = final_scores
        
        # Return the best move
        if not candidates:
            print("No candidates found (possibly due to time limit). Returning best found so far or None.")
            if move_scores:
                return move_scores[0][0]
            else:
                return None
        best_move = candidates[0][0]
        print(f"Final result: Selected move with score {candidates[0][1]:.4f}")
        return best_move
        
    def _evaluate_move(self, state, simulations):
        """Evaluate a move using Monte Carlo with domain knowledge.
        
        Args:
            state: State after applying the move
            simulations: Number of simulations to run
            
        Returns:
            float: Win ratio (0 to 1)
        """
        # If no simulations requested, return a default value of 0.5
        if simulations <= 0:
            return 0.5
            
        wins = 0
        for _ in range(simulations):
            result = self._simulate_with_domain_knowledge(state)
            wins += result
        return wins / simulations
        
    def _simulate_with_domain_knowledge(self, state):
        """Simulate with domain knowledge and probability distribution.
        
        Args:
            state: Starting game state for simulation
            
        Returns:
            float: Evaluation score in [0, 1] range
        """
        state = deepcopy(state)
        player = state.current_player()
        
        while not state.is_game_over():
            moves = state.get_all_possible_moves()
            if not moves:
                break
            
            # Apply domain knowledge and probability distribution to select move
            move = self._select_move_with_probability_distribution(state, moves)
            state.move_with_position(move)
            state.toggle_player()
        
        # Evaluate result with unified evaluation function
        return self._evaluate_final_position(state, player)
        
    def _select_move_with_probability_distribution(self, state, moves):
        """Select a move based on probability distribution P(m_i) = S_i^2 / ∑_j=1^M S_j^2.
        
        Args:
            state: Current game state
            moves: List of legal moves
            
        Returns:
            Move: Selected move based on probability distribution
        """
        if not moves:
            return None
        
        # Calculate score and square for each move
        scored_moves = [(move, self._score_move(state, move)) for move in moves]
        scores_squared = [score**2 for _, score in scored_moves]
        sum_scores_squared = sum(scores_squared)
        
        # Handle case where all scores are 0
        if sum_scores_squared == 0:
            return random.choice(moves)
        
        # Select a move randomly based on probability distribution
        r = random.random() * sum_scores_squared
        current_sum = 0
        for i, score_squared in enumerate(scores_squared):
            current_sum += score_squared
            if r <= current_sum:
                return scored_moves[i][0]
        
        return scored_moves[-1][0]
    
    def _score_move(self, state, move):
        """Score a move based on heuristics.
        
        Formula: Si = s1·(opponent pieces captured) + s2·(own pieces around destination) 
               + s3·1{Clone} − s4·(own pieces around source if Jump)
        
        Args:
            state: Current game state
            move: Move to evaluate
            
        Returns:
            float: Heuristic score for the move
        """
        player = state.current_player()
        s1, s2, s3, s4 = MOVE_WEIGHTS.values()  # Heuristic weights
        
        # Create new state and calculate captures
        next_state = deepcopy(state)
        next_state.move_with_position(move)
        captures = state.balls[-player] - next_state.balls[-player]
        
        # Determine move type and positions
        if move[0] == CLONE_MOVE:  # Clone
            dest_x, dest_y = move[1]
            is_clone = 1
            adjacent_friendly_pieces_source = 0
        else:  # Jump
            dest_x, dest_y = move[1]
            source_x, source_y = move[2]
            is_clone = 0
            
            # Count friendly pieces around source
            adjacent_friendly_pieces_source = sum(1 for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                                              if (dx != 0 or dy != 0) and 0 <= source_x + dx < 7 
                                              and 0 <= source_y + dy < 7 
                                              and state.board[source_x + dx][source_y + dy] == player)
        
        # Count friendly pieces around destination (before move)
        adjacent_friendly_pieces_dest = sum(1 for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                                        if (dx != 0 or dy != 0) and 0 <= dest_x + dx < 7 
                                        and 0 <= dest_y + dy < 7 
                                        and state.board[dest_x + dx][dest_y + dy] == player)
        
        # Calculate score according to formula and ensure non-negative
        score = (s1 * captures + s2 * adjacent_friendly_pieces_dest + 
                s3 * is_clone - s4 * adjacent_friendly_pieces_source)
        
        return max(0, score)