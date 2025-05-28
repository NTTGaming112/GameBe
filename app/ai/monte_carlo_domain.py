#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized Domain Knowledge-enhanced Monte Carlo Tree Search for Ataxx AI.

This module implements an optimized version of MCTS with domain knowledge that
significantly outperforms pure MCTS through intelligent move selection,
adaptive parameters, and enhanced evaluation functions.
"""
import random
import math
import time
from copy import deepcopy
from collections import defaultdict

from app.ai.constants import CLONE_MOVE
from .monte_carlo_base import MonteCarloBase, MonteCarloNode

class MCTSDomainNode(MonteCarloNode):
    """Optimized MCTS node with advanced domain knowledge and caching."""
    
    def __init__(self, state, parent=None, move=None, mcd_instance=None):
        super().__init__(state, parent, move)
        self.mcd_instance = mcd_instance
        
        # Initialize additional attributes needed for MCTS
        self.wins = 0  # Track wins for this node
        
        # Advanced caching for performance
        self._heuristic_cache = {}
        self._children_values_cache = {}
        self._position_evaluation_cache = None
        
        # Domain knowledge enhancements
        self._move_urgency = None
        self._tactical_value = None
        self._strategic_importance = None
    
    def is_terminal(self):
        """Check if this node represents a terminal game state."""
        return self.state.is_game_over()
    
    def is_fully_expanded(self):
        """Check if all possible moves have been tried."""
        if not hasattr(self, 'untried_moves') or self.untried_moves is None:
            self.untried_moves = self.state.get_all_possible_moves()
        return len(self.untried_moves) == 0
    
    def uct_value(self, exploration_weight=1.414):
        """Calculate UCT value for this node."""
        if self.visits == 0:
            return float('inf')
        
        # Use ep (expected value) from base class instead of wins
        exploitation = self.ep / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
        
    def select_child_with_enhanced_uct(self, exploration_weight=1.414):
        """Enhanced UCT selection combining multiple domain factors."""
        if not self.children:
            return None
            
        best_child = None
        best_value = float('-inf')
        
        for child in self.children:
            # Standard UCT value
            uct_value = child.uct_value(exploration_weight)
            
            # Domain knowledge enhancement
            domain_bonus = self._calculate_domain_bonus(child)
            
            # Progressive bias: stronger early, weaker as visits increase
            bias_strength = max(0.1, 1.0 / (1.0 + child.visits * 0.1))
            enhanced_value = uct_value + bias_strength * domain_bonus
            
            if enhanced_value > best_value:
                best_value = enhanced_value
                best_child = child
                
        return best_child
    
    def _calculate_domain_bonus(self, child):
        """Calculate comprehensive domain knowledge bonus."""
        if not child.move or not self.mcd_instance:
            return 0.0
            
        cache_key = str(child.move)
        if cache_key in self._children_values_cache:
            return self._children_values_cache[cache_key]
        
        # Multi-factor domain evaluation
        heuristic_score = self.mcd_instance._score_move(self.state, child.move)
        tactical_value = self._evaluate_tactical_value(child.move)
        urgency_factor = self._evaluate_move_urgency(child.move)
        
        # Normalize and combine factors
        normalized_heuristic = min(1.0, heuristic_score / 3.0)
        domain_bonus = (0.6 * normalized_heuristic + 
                       0.3 * tactical_value + 
                       0.1 * urgency_factor)
        
        # Cache the result
        self._children_values_cache[cache_key] = domain_bonus
        return domain_bonus
    
    def _evaluate_tactical_value(self, move):
        """Evaluate immediate tactical value of a move."""
        if not self.mcd_instance:
            return 0.0
            
        # Count immediate captures
        captures = self.mcd_instance._count_potential_captures(
            self.state, move[1][0], move[1][1]
        )
        
        # Normalize captures (typically 0-8 max)
        return min(1.0, captures / 4.0)
    
    def _evaluate_move_urgency(self, move):
        """Evaluate how urgent/critical a move is."""
        # Clone moves are generally less urgent than jumps
        if move[0] == CLONE_MOVE:
            return 0.3
        else:
            return 0.7


class MonteCarloDomain(MonteCarloBase):
    """Optimized Monte Carlo Tree Search with Domain Knowledge.
    
    Key optimizations over pure MCTS:
    1. Intelligent expansion using domain knowledge
    2. Enhanced simulation with probability-based move selection
    3. Adaptive exploration parameters
    4. Multi-level evaluation caching
    5. Progressive bias in node selection
    6. Tournament-style move evaluation
    """
    
    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        
        # Core heuristic parameters (optimized values)
        self.s1 = kwargs.get('s1', 1.2)    # Increased capture importance
        self.s2 = kwargs.get('s2', 0.5)    # Clustering value
        self.s3 = kwargs.get('s3', 0.8)    # Clone bonus
        self.s4 = kwargs.get('s4', 0.3)    # Jump penalty (reduced)
        
        # MCTS optimization parameters
        self.use_tree_search = kwargs.get('use_tree_search', True)
        self.max_tree_depth = kwargs.get('max_tree_depth', 8)
        self.adaptive_exploration = kwargs.get('adaptive_exploration', True)
        self.progressive_bias = kwargs.get('progressive_bias', True)
        
        # Performance optimization
        self.use_move_ordering = kwargs.get('use_move_ordering', True)
        self.use_early_termination = kwargs.get('use_early_termination', True)
        self.simulation_depth_limit = kwargs.get('simulation_depth_limit', 15)
        
        # NEW: Simulation-based parameters
        self.base_simulations = kwargs.get('basic_simulations', 600)  # User input simulations
        self.max_simulations = kwargs.get('max_simulations', 50000)  # Safety limit
        self.min_simulations_per_move = kwargs.get('min_simulations_per_move', 50)
        
        # Tournament distribution ratios (customizable)
        self.round1_ratio = kwargs.get('round1_ratio', 0.4)    # 40% of total sims
        self.round2_ratio = kwargs.get('round2_ratio', 0.35)   # 35% of total sims  
        self.round3_ratio = kwargs.get('round3_ratio', 0.25)   # 25% of total sims
        
        # Ensure ratios sum to 1.0
        total_ratio = self.round1_ratio + self.round2_ratio + self.round3_ratio
        if abs(total_ratio - 1.0) > 0.001:  # Allow small floating point differences
            self.round1_ratio /= total_ratio
            self.round2_ratio /= total_ratio
            self.round3_ratio /= total_ratio
        
        # Caching for performance
        self._evaluation_cache = {}
        self._move_score_cache = {}
        self._position_hash_cache = {}
        
        # Adaptive parameters
        self._game_phase = None
        self._exploration_decay = 1.0
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'early_terminations': 0,
            'deep_searches': 0
        }
        
        print(f"OptimizedMCTSDomain initialized with enhanced parameters")
        print(f"Domain weights: s1={self.s1}, s2={self.s2}, s3={self.s3}, s4={self.s4}")
    
    def get_move(self, time_limit=None):
        """Get best move using optimized tournament with time management.
        
        Args:
            time_limit: Time limit in seconds. If None, use unlimited time with full simulations.
            simulations: Number of simulations to use. If None, use default base_simulations.
        """
        moves = self.root_state.get_all_possible_moves()
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        
        # Set current simulations to base_simulations (capped by max_simulations)
        self.current_simulations = min(self.base_simulations, self.max_simulations)
            
        print(f"🎯 Monte Carlo Domain: {len(moves)} moves, {self.current_simulations} simulations")
        
        start_time = time.time()
        
        # If no time limit, use unlimited simulation mode
        if time_limit is None:
            return self._unlimited_simulation_search()
        
        # Adaptive strategy based on number of moves and time limit
        if len(moves) <= 8 and time_limit >= 5.0:
            # Use full tree search for small move sets
            return self._optimized_tree_search(time_limit)
        else:
            # Use hybrid tournament for large move sets
            return self._hybrid_tournament_search(time_limit)
    
    def _optimized_tree_search(self, time_limit):
        """Optimized tree search with domain knowledge integration."""
        start_time = time.time()
        moves = self.root_state.get_all_possible_moves()
        
        # Handle unlimited time case
        if time_limit is None:
            print("🔥 Unlimited Optimized Tree Search")
            return self._unlimited_optimized_tree_search(moves)
        
        # Pre-filter and order moves using domain knowledge
        if self.use_move_ordering:
            moves = self._order_moves_by_potential(moves)
        
        # Focus on top moves
        focus_moves = moves[:min(6, len(moves))]
        
        # Distribute simulations among focus moves
        simulations_per_move = max(
            self.min_simulations_per_move,
            self.current_simulations // len(focus_moves)
        )
        
        print(f"🚀 Tree Search: {len(focus_moves)} focus moves, {simulations_per_move} sims/move")
        
        move_evaluations = {}
        for move in focus_moves:
            score = self._deep_tree_evaluation(move, simulations_per_move, time_limit)
            move_evaluations[move] = score
            
            # Early termination if time is running out
            if time.time() - start_time > time_limit * 0.9:
                break
        
        # Select best move
        if move_evaluations:
            best_move = max(move_evaluations.items(), key=lambda x: x[1])[0]
            best_score = move_evaluations[best_move]
            print(f"✅ Selected move with score {best_score:.4f}")
            return best_move
        
        return moves[0]
    
    def _deep_tree_evaluation(self, move, iterations, time_limit):
        """Deep tree evaluation for a specific move."""
        # Apply the move
        next_state = deepcopy(self.root_state)
        next_state.move_with_position(move)
        next_state.toggle_player()
        
        # Create enhanced tree node
        root = MCTSDomainNode(next_state, mcd_instance=self)
        original_player = self.root_state.current_player()
        
        # Adaptive exploration based on game phase
        exploration_param = self._get_adaptive_exploration()
        
        for i in range(iterations):
            # MCTS phases with domain knowledge
            leaf = self._enhanced_selection(root, exploration_param)
            
            if not leaf.is_terminal() and leaf.visits > 0:
                leaf = self._intelligent_expansion(leaf)
            
            # Enhanced simulation
            result = self._enhanced_simulation(leaf, original_player)
            
            # Backpropagation
            self._backpropagate_with_decay(leaf, result)
        
        # Return evaluation
        if root.visits == 0:
            return 0.5
        return root.ep / root.visits
    
    def _enhanced_selection(self, node, exploration_param):
        """Enhanced selection phase with domain knowledge."""
        while not node.is_terminal() and node.children:
            if not node.is_fully_expanded():
                return node
            node = node.select_child_with_enhanced_uct(exploration_param)
        return node
    
    def _intelligent_expansion(self, node):
        """Intelligent expansion using domain knowledge."""
        if not node.untried_moves:
            node.untried_moves = node.state.get_all_possible_moves()
        
        if not node.untried_moves:
            return node
        
        # Select best move for expansion using domain knowledge
        if len(node.untried_moves) > 1:
            best_move = self._select_best_expansion_move(node.state, node.untried_moves)
        else:
            best_move = node.untried_moves[0]
        
        # Create child node
        next_state = deepcopy(node.state)
        next_state.move_with_position(best_move)
        next_state.toggle_player()
        
        child = MCTSDomainNode(next_state, parent=node, move=best_move, mcd_instance=self)
        node.children.append(child)
        node.untried_moves.remove(best_move)
        
        return child
    
    def _enhanced_simulation(self, node, original_player):
        """Enhanced simulation with domain knowledge."""
        if node.is_terminal():
            return self._evaluate_final_position(node.state, original_player)
        
        # Multi-component evaluation
        heuristic_eval = self._fast_position_evaluation(node.state, original_player)
        
        # Short tactical simulation
        tactical_eval = self._tactical_simulation(node.state, original_player)
        
        # Combine evaluations
        combined_score = 0.7 * heuristic_eval + 0.3 * tactical_eval
        
        return max(0.0, min(1.0, combined_score))
    
    def _fast_position_evaluation(self, state, original_player):
        """Fast position evaluation using multiple factors."""
        # Check cache first
        state_hash = hash(str(state.board))
        if state_hash in self._evaluation_cache:
            self.stats['cache_hits'] += 1
            return self._evaluation_cache[state_hash]
        
        self.stats['cache_misses'] += 1
        
        # Multi-factor evaluation
        material_score = self._evaluate_material_balance(state, original_player)
        mobility_score = self._evaluate_mobility(state, original_player)
        position_score = self._evaluate_position_quality(state, original_player)
        threat_score = self._evaluate_threat_balance(state, original_player)
        
        # Game phase adaptive weights
        phase = self._detect_game_phase(state)
        if phase == 'opening':
            weights = [0.3, 0.3, 0.3, 0.1]
        elif phase == 'midgame':
            weights = [0.4, 0.2, 0.2, 0.2]
        else:  # endgame
            weights = [0.6, 0.1, 0.2, 0.1]
        
        combined_score = (weights[0] * material_score +
                         weights[1] * mobility_score +
                         weights[2] * position_score +
                         weights[3] * threat_score)
        
        # Cache result
        self._evaluation_cache[state_hash] = combined_score
        
        return combined_score
    
    def _tactical_simulation(self, state, original_player, max_depth=3):
        """Short tactical simulation with domain knowledge."""
        sim_state = deepcopy(state)
        depth = 0
        
        while not sim_state.is_game_over() and depth < max_depth:
            moves = sim_state.get_all_possible_moves()
            if not moves:
                break
            
            # Use probability-based move selection
            move = self._select_move_with_enhanced_probability(sim_state, moves)
            sim_state.move_with_position(move)
            sim_state.toggle_player()
            depth += 1
        
        return self._evaluate_final_position(sim_state, original_player)
    
    def _select_move_with_enhanced_probability(self, state, moves):
        """Enhanced probability-based move selection."""
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        
        # Limit moves for performance in simulations
        if len(moves) > 8:
            moves = self._get_top_moves(state, moves, 8)
        
        # Calculate enhanced scores
        enhanced_scores = []
        for move in moves:
            base_score = self._score_move(state, move)
            tactical_bonus = self._get_tactical_bonus(state, move)
            enhanced_score = base_score + 0.3 * tactical_bonus
            enhanced_scores.append(max(0.1, enhanced_score))  # Minimum score
        
        # Probability distribution with temperature
        temperature = 0.8
        probabilities = [(score ** (1/temperature)) for score in enhanced_scores]
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        
        # Select move
        r = random.random()
        cumulative = 0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return moves[i]
        
        return moves[-1]
    
    def _get_tactical_bonus(self, state, move):
        """Calculate tactical bonus for immediate threats/opportunities."""
        bonus = 0.0
        dest_x, dest_y = move[1]
        
        # Immediate capture bonus
        captures = self._count_potential_captures(state, dest_x, dest_y)
        bonus += captures * 0.5
        
        # Center control bonus
        center_distance = abs(dest_x - 3) + abs(dest_y - 3)
        bonus += (4 - center_distance) * 0.1
        
        # Threat creation bonus
        threats = self._count_threats_created(state, move)
        bonus += threats * 0.2
        
        return bonus
    
    def _count_threats_created(self, state, move):
        """Count new threats created by this move."""
        dest_x, dest_y = move[1]
        player = state.current_player()
        enemy = -player
        
        threats = 0
        # Check all positions this move could threaten
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if abs(dx) <= 1 and abs(dy) <= 1:
                    continue  # Skip clone range
                if abs(dx) > 2 or abs(dy) > 2:
                    continue  # Too far
                
                threat_x, threat_y = dest_x + dx, dest_y + dy
                if (0 <= threat_x < 7 and 0 <= threat_y < 7 and 
                    state.board[threat_x][threat_y] == enemy):
                    threats += 1
        
        return threats
    
    def _get_top_moves(self, state, moves, count):
        """Get top N moves based on quick evaluation."""
        if len(moves) <= count:
            return moves
        
        scored_moves = [(move, self._quick_move_score(state, move)) for move in moves]
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        return [move for move, _ in scored_moves[:count]]
    
    def _quick_move_score(self, state, move):
        """Quick move scoring for filtering."""
        cache_key = str(move) + str(hash(str(state.board)))
        if cache_key in self._move_score_cache:
            return self._move_score_cache[cache_key]
        
        # Fast scoring
        dest_x, dest_y = move[1]
        player = state.current_player()
        
        score = 0.0
        
        # Quick capture count
        captures = self._count_potential_captures(state, dest_x, dest_y)
        score += captures * 1.5
        
        # Clone bonus
        if move[0] == CLONE_MOVE:
            score += 0.8
        
        # Center preference
        center_distance = abs(dest_x - 3) + abs(dest_y - 3)
        score += (4 - center_distance) * 0.1
        
        self._move_score_cache[cache_key] = score
        return score
    
    def _get_adaptive_exploration(self):
        """Get adaptive exploration parameter based on game state."""
        if not self.adaptive_exploration:
            return 1.414
        
        # Detect game phase
        total_pieces = sum(self.root_state.balls.values())
        
        if total_pieces <= 10:  # Opening
            return 1.6  # More exploration
        elif total_pieces >= 35:  # Endgame
            return 1.0  # Less exploration
        else:  # Midgame
            return 1.414  # Standard
    
    def _order_moves_by_potential(self, moves):
        """Order moves by their potential using domain knowledge."""
        scored_moves = []
        for move in moves:
            score = self._score_move(self.root_state, move)
            tactical_value = self._get_tactical_bonus(self.root_state, move)
            total_score = score + tactical_value
            scored_moves.append((move, total_score))
        
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored_moves]
    
    def _select_best_expansion_move(self, state, untried_moves):
        """Select best move for expansion using domain knowledge."""
        if len(untried_moves) == 1:
            return untried_moves[0]
        
        best_move = None
        best_score = -1
        
        for move in untried_moves:
            score = self._score_move(state, move)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move if best_move else untried_moves[0]
    
    def _backpropagate_with_decay(self, node, result):
        """Backpropagation with value decay for recent experience."""
        current_result = result
        
        while node is not None:
            # Use the base class update method
            node.update(current_result)
            
            # Flip result for alternating players
            current_result = 1.0 - current_result
            node = node.parent
    
    def _hybrid_tournament_search(self, time_limit):
        """Hybrid tournament search for large move sets or limited time."""
        moves = self.root_state.get_all_possible_moves()
        
        # Quick filtering to reduce search space
        if len(moves) > 12:
            moves = self._get_top_moves(self.root_state, moves, 12)
        
        # Handle unlimited time case
        if time_limit is None:
            print("🚀 Unlimited tournament mode")
            return self._unlimited_enhanced_tournament(moves)
        
        # Adaptive tournament rounds based on available time and simulations
        total_sims = self.current_simulations
        
        if time_limit >= 8.0 and total_sims >= 1000:
            return self._three_round_tournament_sims(moves, total_sims)
        elif time_limit >= 4.0 and total_sims >= 500:
            return self._two_round_tournament_sims(moves, total_sims)
        else:
            return self._single_round_evaluation_sims(moves, total_sims)
    
    def _three_round_tournament(self, moves, time_limit):
        """Traditional three-round tournament."""
        start_time = time.time()
        time_per_round = time_limit / 3.0
        
        print(f"🏆 Three-Round Tournament: {len(moves)} moves")
        
        # Round 1: Evaluate all moves
        round1_scores = {}
        simulations_per_move = max(100, int(800 / len(moves)))
        
        for move in moves:
            if time.time() - start_time > time_per_round:
                break
            score = self._evaluate_move_quick(move, simulations_per_move)
            round1_scores[move] = score
        
        # Select top 5 for round 2
        if len(round1_scores) < 5:
            top5 = list(round1_scores.keys())
        else:
            sorted_moves = sorted(round1_scores.items(), key=lambda x: x[1], reverse=True)
            top5 = [move for move, _ in sorted_moves[:5]]
        
        # Round 2: Deeper evaluation of top 5
        round2_scores = {}
        simulations_per_move = 300
        
        round2_start = time.time()
        for move in top5:
            if time.time() - start_time > 2 * time_per_round:
                break
            score = self._evaluate_move_deep(move, simulations_per_move)
            round2_scores[move] = score
        
        # Select top 3 for round 3
        if len(round2_scores) < 3:
            top3 = list(round2_scores.keys())
        else:
            sorted_moves = sorted(round2_scores.items(), key=lambda x: x[1], reverse=True)
            top3 = [move for move, _ in sorted_moves[:3]]
        
        # Round 3: Final evaluation of top 3
        final_scores = {}
        simulations_per_move = 500
        
        for move in top3:
            if time.time() - start_time > time_limit * 0.95:
                break
            score = self._evaluate_move_deep(move, simulations_per_move)
            final_scores[move] = score
        
        # Select best move
        if final_scores:
            best_move = max(final_scores.items(), key=lambda x: x[1])[0]
            print(f"✅ Tournament winner: {best_move} with score {final_scores[best_move]:.4f}")
            return best_move
        elif round2_scores:
            return max(round2_scores.items(), key=lambda x: x[1])[0]
        elif round1_scores:
            return max(round1_scores.items(), key=lambda x: x[1])[0]
        else:
            return moves[0]
    
    def _two_round_tournament(self, moves, time_limit):
        """Two-round tournament for medium time limits."""
        start_time = time.time()
        
        # Round 1: Quick evaluation
        round1_scores = {}
        simulations_per_move = max(150, int(1000 / len(moves)))
        
        for move in moves:
            if time.time() - start_time > time_limit * 0.6:
                break
            score = self._evaluate_move_quick(move, simulations_per_move)
            round1_scores[move] = score
        
        # Select top 5 for round 2
        if len(round1_scores) <= 5:
            finalists = list(round1_scores.keys())
        else:
            sorted_moves = sorted(round1_scores.items(), key=lambda x: x[1], reverse=True)
            finalists = [move for move, _ in sorted_moves[:5]]
        
        # Round 2: Final evaluation
        final_scores = {}
        simulations_per_move = 400
        
        for move in finalists:
            if time.time() - start_time > time_limit * 0.95:
                break
            score = self._evaluate_move_deep(move, simulations_per_move)
            final_scores[move] = score
        
        if final_scores:
            return max(final_scores.items(), key=lambda x: x[1])[0]
        elif round1_scores:
            return max(round1_scores.items(), key=lambda x: x[1])[0]
        else:
            return moves[0]
    
    def _single_round_evaluation(self, moves, time_limit):
        """Single round evaluation for limited time."""
        move_scores = {}
        simulations_per_move = max(200, int(1500 / len(moves)))
        start_time = time.time()
        
        for move in moves:
            if time.time() - start_time > time_limit * 0.9:
                break
            score = self._evaluate_move_quick(move, simulations_per_move)
            move_scores[move] = score
        
        if move_scores:
            return max(move_scores.items(), key=lambda x: x[1])[0]
        else:
            return moves[0]
    
    def _three_round_tournament_sims(self, moves, total_simulations):
        """Three-round tournament with simulation-based distribution."""
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
            
        print(f"🏆 Three-Round Tournament: {len(moves)} moves, {total_simulations} total sims")
        
        # Calculate simulations for each round
        round1_sims = max(1, int(total_simulations * self.round1_ratio))
        round2_sims = max(1, int(total_simulations * self.round2_ratio))
        round3_sims = max(1, int(total_simulations * self.round3_ratio))
        
        # Round 1: Evaluate all moves
        round1_sims_per_move = max(
            self.min_simulations_per_move,
            round1_sims // len(moves)
        )
        
        print(f"  Round 1: {len(moves)} moves × {round1_sims_per_move} sims = {len(moves) * round1_sims_per_move}")
        
        round1_scores = {}
        for move in moves:
            score = self._evaluate_move_with_simulations(move, round1_sims_per_move)
            round1_scores[move] = score
        
        # Select top 5 for round 2
        if len(round1_scores) <= 5:
            top5 = list(round1_scores.keys())
        else:
            sorted_moves = sorted(round1_scores.items(), key=lambda x: x[1], reverse=True)
            top5 = [move for move, _ in sorted_moves[:5]]
        
        # Round 2: Deeper evaluation of top 5
        round2_sims_per_move = max(
            self.min_simulations_per_move,
            round2_sims // len(top5)
        )
        
        print(f"  Round 2: {len(top5)} moves × {round2_sims_per_move} sims = {len(top5) * round2_sims_per_move}")
        
        round2_scores = {}
        for move in top5:
            score = self._evaluate_move_with_simulations(move, round2_sims_per_move)
            round2_scores[move] = score
        
        # Select top 3 for round 3
        if len(round2_scores) <= 3:
            top3 = list(round2_scores.keys())
        else:
            sorted_moves = sorted(round2_scores.items(), key=lambda x: x[1], reverse=True)
            top3 = [move for move, _ in sorted_moves[:3]]
        
        # Round 3: Final evaluation of top 3
        round3_sims_per_move = max(
            self.min_simulations_per_move,
            round3_sims // len(top3)
        )
        
        print(f"  Round 3: {len(top3)} moves × {round3_sims_per_move} sims = {len(top3) * round3_sims_per_move}")
        
        final_scores = {}
        for move in top3:
            score = self._evaluate_move_with_simulations(move, round3_sims_per_move)
            final_scores[move] = score
        
        # Select best move
        if final_scores:
            best_move = max(final_scores.items(), key=lambda x: x[1])[0]
            best_score = final_scores[best_move]
            print(f"✅ Tournament winner: {best_move} with score {best_score:.4f}")
            
            # Print simulation distribution summary
            total_used = (len(moves) * round1_sims_per_move + 
                         len(top5) * round2_sims_per_move + 
                         len(top3) * round3_sims_per_move)
            print(f"📊 Simulations used: {total_used}/{total_simulations} ({total_used/total_simulations:.1%})")
            
            return best_move
        
        return moves[0] if moves else None
    
    def _two_round_tournament_sims(self, moves, total_simulations):
        """Two-round tournament with simulation distribution."""
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
            
        print(f"🥈 Two-Round Tournament: {len(moves)} moves, {total_simulations} total sims")
        
        # Split simulations: 60% round 1, 40% round 2
        round1_sims = max(1, int(total_simulations * 0.6))
        round2_sims = max(1, int(total_simulations * 0.4))
        
        # Round 1: Evaluate all moves
        round1_sims_per_move = max(
            self.min_simulations_per_move,
            round1_sims // len(moves)
        )
        
        round1_scores = {}
        for move in moves:
            score = self._evaluate_move_with_simulations(move, round1_sims_per_move)
            round1_scores[move] = score
        
        # Select top 5 for round 2
        if len(round1_scores) <= 5:
            finalists = list(round1_scores.keys())
        else:
            sorted_moves = sorted(round1_scores.items(), key=lambda x: x[1], reverse=True)
            finalists = [move for move, _ in sorted_moves[:5]]
        
        # Round 2: Final evaluation
        round2_sims_per_move = max(
            self.min_simulations_per_move,
            round2_sims // len(finalists)
        )
        
        final_scores = {}
        for move in finalists:
            score = self._evaluate_move_with_simulations(move, round2_sims_per_move)
            final_scores[move] = score
        
        if final_scores:
            best_move = max(final_scores.items(), key=lambda x: x[1])[0]
            print(f"✅ Two-round winner: {best_move}")
            return best_move
        
        return moves[0]
    
    def _single_round_evaluation_sims(self, moves, total_simulations):
        """Single round evaluation with simulation distribution."""
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
            
        print(f"🥉 Single Round: {len(moves)} moves, {total_simulations} total sims")
        
        simulations_per_move = max(
            self.min_simulations_per_move,
            total_simulations // len(moves)
        )
        
        move_scores = {}
        for move in moves:
            score = self._evaluate_move_with_simulations(move, simulations_per_move)
            move_scores[move] = score
        
        if move_scores:
            best_move = max(move_scores.items(), key=lambda x: x[1])[0]
            print(f"✅ Single round winner: {best_move}")
            return best_move
        
        return moves[0]
    
    def _evaluate_move_with_simulations(self, move, simulations):
        """Evaluate a single move with specified number of simulations."""
        next_state = deepcopy(self.root_state)
        next_state.move_with_position(move)
        next_state.toggle_player()
        
        # Use MCTS tree search for evaluation
        root = MCTSDomainNode(next_state, mcd_instance=self)
        original_player = self.root_state.current_player()
        exploration_param = self._get_adaptive_exploration()
        
        for i in range(simulations):
            # Standard MCTS phases
            leaf = self._enhanced_selection(root, exploration_param)
            
            if not leaf.is_terminal() and leaf.visits > 0:
                leaf = self._intelligent_expansion(leaf)
            
            result = self._enhanced_simulation(leaf, original_player)
            self._backpropagate_with_decay(leaf, result)
        
        # Return average score
        if root.visits == 0:
            return 0.5
        return root.ep / root.visits
    
    def set_simulation_parameters(self, simulations=None, round1_ratio=None, 
                                 round2_ratio=None, round3_ratio=None):
        """Set custom simulation parameters."""
        if simulations is not None:
            self.base_simulations = min(simulations, self.max_simulations)
            
        if round1_ratio is not None:
            self.round1_ratio = round1_ratio
        if round2_ratio is not None:
            self.round2_ratio = round2_ratio
        if round3_ratio is not None:
            self.round3_ratio = round3_ratio
            
        # Normalize ratios to sum to 1.0
        total_ratio = self.round1_ratio + self.round2_ratio + self.round3_ratio
        if abs(total_ratio - 1.0) > 0.001:  # Allow small floating point differences
            self.round1_ratio /= total_ratio
            self.round2_ratio /= total_ratio
            self.round3_ratio /= total_ratio
            
        print(f"📊 Updated simulation parameters:")
        print(f"  Base simulations: {self.base_simulations}")
        print(f"  Round ratios: {self.round1_ratio:.2f}, {self.round2_ratio:.2f}, {self.round3_ratio:.2f}")
    
    # Keep all core methods from original implementation
    def _score_move(self, state, move):
        """Core heuristic scoring function."""
        player = state.current_player()
        
        # Create a copy to simulate the move and count captures
        next_state = deepcopy(state)
        next_state.move_with_position(move)
        
        # s1 term: enemy stones captured
        enemy_captures = state.balls[-player] - next_state.balls[-player]
        
        # Get move positions
        dest_x, dest_y = move[1]
        
        # s2 term: own stones around target
        own_around_target = self._count_adjacent_pieces(state, dest_x, dest_y, player)
        
        # s3 and s4 terms
        if move[0] == CLONE_MOVE:
            move_type_bonus = self.s3
            source_penalty = 0
        else:
            move_type_bonus = 0
            source_x, source_y = move[2]
            own_around_source = self._count_adjacent_pieces(state, source_x, source_y, player)
            source_penalty = self.s4 * own_around_source
        
        # Calculate final score
        score = (self.s1 * enemy_captures + 
                self.s2 * own_around_target + 
                move_type_bonus - 
                source_penalty)
        
        return max(0.0, score)
    
    def _count_adjacent_pieces(self, state, x, y, player):
        """Count adjacent pieces of a player."""
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < 7 and 0 <= ny < 7 and state.board[nx][ny] == player:
                    count += 1
        return count
    
    def _count_potential_captures(self, state, x, y):
        """Count potential enemy captures at position."""
        player = state.current_player()
        enemy = -player
        count = 0
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < 7 and 0 <= ny < 7 and state.board[nx][ny] == enemy:
                    count += 1
        return count
    
    def _evaluate_material_balance(self, state, player):
        """Evaluate material balance."""
        my_pieces = state.balls[player]
        opp_pieces = state.balls[-player]
        total = my_pieces + opp_pieces
        return my_pieces / total if total > 0 else 0.5
    
    def _evaluate_mobility(self, state, player):
        """Evaluate mobility."""
        current_player = state.current_player()
        my_moves = len(state.get_all_possible_moves())
        
        temp_state = deepcopy(state)
        temp_state.toggle_player()
        opp_moves = len(temp_state.get_all_possible_moves())
        
        total = my_moves + opp_moves
        if total == 0:
            return 0.5
        
        if current_player == player:
            return my_moves / total
        else:
            return opp_moves / total
    
    def _evaluate_position_quality(self, state, player):
        """Evaluate position quality."""
        score = 0.0
        count = 0
        
        for x in range(7):
            for y in range(7):
                if state.board[x][y] == player:
                    count += 1
                    center_distance = abs(x - 3) + abs(y - 3)
                    score += (7 - center_distance) / 7.0
                    
                    neighbors = self._count_adjacent_pieces(state, x, y, player)
                    score += neighbors * 0.1
        
        return score / max(1, count)
    
    def _evaluate_threat_balance(self, state, player):
        """Evaluate threat balance."""
        my_threats = 0
        opp_threats = 0
        
        for x in range(7):
            for y in range(7):
                if state.board[x][y] == player:
                    my_threats += self._count_potential_captures(state, x, y)
                elif state.board[x][y] == -player:
                    temp_state = deepcopy(state)
                    temp_state.toggle_player()
                    opp_threats += self._count_potential_captures(temp_state, x, y)
        
        total = my_threats + opp_threats
        return my_threats / total if total > 0 else 0.5
    
    def _detect_game_phase(self, state):
        """Detect current game phase."""
        total_pieces = sum(state.balls.values())
        if total_pieces <= 12:
            return 'opening'
        elif total_pieces >= 35:
            return 'endgame'
        else:
            return 'midgame'
    
    def print_stats(self):
        """Print performance statistics."""
        print(f"📊 Performance Stats:")
        print(f"  Cache hits: {self.stats['cache_hits']}")
        print(f"  Cache misses: {self.stats['cache_misses']}")
        hit_rate = self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
        print(f"  Hit rate: {hit_rate:.2%}")
    
    def _unlimited_simulation_search(self):
        """Unlimited simulation search using configurable simulation distribution."""
        moves = self.root_state.get_all_possible_moves()
        
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        
        # Use maximum simulations for unlimited mode
        total_simulations = self.current_simulations
        print(f"🔥 Unlimited Simulation Mode: {len(moves)} moves, {total_simulations} total sims")
        
        # Choose strategy based on number of moves
        if len(moves) <= 5:
            # Direct deep evaluation for few moves
            return self._unlimited_direct_evaluation(moves, total_simulations)
        elif len(moves) <= 12:
            # Enhanced three-round tournament
            return self._unlimited_three_round_tournament(moves, total_simulations)
        else:
            # Filter to top moves first, then tournament
            top_moves = self._get_top_moves(self.root_state, moves, 15)
            return self._unlimited_three_round_tournament(top_moves, total_simulations)
    
    def _unlimited_direct_evaluation(self, moves, total_simulations):
        """Direct evaluation with unlimited simulations for few moves."""
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
            
        move_scores = {}
        simulations_per_move = total_simulations // len(moves)
        
        print(f"🎯 Direct Unlimited Evaluation: {len(moves)} moves × {simulations_per_move} sims each")
        
        for i, move in enumerate(moves):
            print(f"  Evaluating move {i+1}/{len(moves)}: {move}")
            score = self._evaluate_move_with_simulations(move, simulations_per_move)
            move_scores[move] = score
            print(f"    Score: {score:.6f}")
        
        best_move = max(move_scores.items(), key=lambda x: x[1])[0]
        best_score = move_scores[best_move]
        print(f"✅ Best unlimited move: {best_move} with score {best_score:.6f}")
        return best_move
    
    def _unlimited_three_round_tournament(self, moves, total_simulations):
        """Enhanced three-round tournament using configurable simulation ratios."""
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
            
        print(f"🏆 Unlimited Three-Round Tournament: {len(moves)} moves, {total_simulations} total sims")
        
        # Use the same ratio distribution as regular tournament
        round1_sims = max(1, int(total_simulations * self.round1_ratio))
        round2_sims = max(1, int(total_simulations * self.round2_ratio))
        round3_sims = max(1, int(total_simulations * self.round3_ratio))
        
        print(f"  Distribution: R1={round1_sims} ({self.round1_ratio:.1%}), R2={round2_sims} ({self.round2_ratio:.1%}), R3={round3_sims} ({self.round3_ratio:.1%})")
        
        # Round 1: Evaluate all moves
        round1_sims_per_move = max(
            self.min_simulations_per_move,
            round1_sims
        )
        
        print(f"  Round 1: {len(moves)} moves × {round1_sims_per_move} sims each")
        round1_scores = {}
        for i, move in enumerate(moves):
            print(f"    Evaluating {i+1}/{len(moves)}: {move}")
            score = self._evaluate_move_with_simulations(move, round1_sims_per_move)
            round1_scores[move] = score
            print(f"      Score: {score:.6f}")
        
        # Select top candidates for round 2 (more generous selection)
        num_round2 = min(8, max(5, len(moves) // 2))
        if len(round1_scores) <= num_round2:
            round2_moves = list(round1_scores.keys())
        else:
            sorted_moves = sorted(round1_scores.items(), key=lambda x: x[1], reverse=True)
            round2_moves = [move for move, _ in sorted_moves[:num_round2]]
        
        # Round 2: Deeper evaluation
        round2_sims_per_move = max(
            self.min_simulations_per_move,
            round2_sims
        )
        
        print(f"  Round 2: {len(round2_moves)} moves × {round2_sims_per_move} sims each")
        round2_scores = {}
        for i, move in enumerate(round2_moves):
            print(f"    Deep evaluating {i+1}/{len(round2_moves)}: {move}")
            score = self._evaluate_move_with_simulations(move, round2_sims_per_move)
            round2_scores[move] = score
            print(f"      Score: {score:.6f}")
        
        # Select top 3 for final round
        num_round3 = min(3, len(round2_moves))
        if len(round2_scores) <= num_round3:
            final_moves = list(round2_scores.keys())
        else:
            sorted_moves = sorted(round2_scores.items(), key=lambda x: x[1], reverse=True)
            final_moves = [move for move, _ in sorted_moves[:num_round3]]
        
        # Round 3: Maximum depth evaluation
        round3_sims_per_move = max(
            self.min_simulations_per_move,
            round3_sims
        )
        
        print(f"  Round 3: {len(final_moves)} moves × {round3_sims_per_move} sims each")
        final_scores = {}
        for i, move in enumerate(final_moves):
            print(f"    Final evaluating {i+1}/{len(final_moves)}: {move}")
            score = self._evaluate_move_with_simulations(move, round3_sims_per_move)
            final_scores[move] = score
            print(f"      Final score: {score:.6f}")
        
        # Select best move
        if final_scores:
            best_move = max(final_scores.items(), key=lambda x: x[1])[0]
            best_score = final_scores[best_move]
            
            # Print summary
            total_used = (len(moves) * round1_sims_per_move + 
                         len(round2_moves) * round2_sims_per_move + 
                         len(final_moves) * round3_sims_per_move)
            print(f"📊 Unlimited tournament used: {total_used}/{total_simulations} ({total_used/total_simulations:.1%})")
            print(f"🎉 Unlimited tournament winner: {best_move} with score {best_score:.6f}")
            
            return best_move
        
        return moves[0]
    
    def _evaluate_move_deep(self, move, simulations):
        """Deep evaluation of a move with specified simulations using enhanced MCTS."""
        next_state = deepcopy(self.root_state)
        next_state.move_with_position(move)
        next_state.toggle_player()
        
        # Use enhanced MCTS for deep evaluation
        root = MCTSDomainNode(next_state, mcd_instance=self)
        original_player = self.root_state.current_player()
        
        # Enhanced parameters for deep evaluation
        exploration_param = self._get_adaptive_exploration()
        
        for i in range(simulations):
            # Enhanced MCTS phases with domain knowledge
            leaf = self._enhanced_selection(root, exploration_param)
            
            if not leaf.is_terminal() and leaf.visits > 0:
                leaf = self._intelligent_expansion(leaf)
            
            # Use enhanced simulation with domain knowledge
            result = self._enhanced_simulation(leaf, original_player)
            
            # Backpropagate with decay
            self._backpropagate_with_decay(leaf, result)
            
            # Update exploration parameter dynamically
            if i % 100 == 0:
                exploration_param = self._get_adaptive_exploration()
        
        # Calculate final score with confidence weighting
        if root.visits == 0:
            return 0.5
            
        # Base score from MCTS
        base_score = root.ep / root.visits
        
        # Add confidence factor based on number of visits
        confidence = min(1.0, root.visits / (simulations * 0.8))
        
        # Combine with static evaluation for robustness
        static_score = self._score_move(self.root_state, move)
        normalized_static = static_score / (1.0 + static_score)  # Normalize to [0,1]
        
        # Weight combination: more MCTS weight with higher confidence
        final_score = (confidence * base_score + 
                      (1 - confidence) * 0.3 * normalized_static + 
                      confidence * 0.7 * base_score)
        
        return final_score
