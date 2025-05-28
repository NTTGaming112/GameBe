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
        self.min_simulations_per_move = kwargs.get('min_simulations_per_move', 100)
        
        # Tournament distribution ratios (customizable)
        self.round1_ratio = kwargs.get('round1_ratio', 1)    # 40% of total sims
        self.round2_ratio = kwargs.get('round2_ratio', 1)   # 35% of total sims  
        self.round3_ratio = kwargs.get('round3_ratio', 0.5)   # 25% of total sims
        
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
        """Get best move using unified strategy selection system.
        
        Args:
            time_limit: Time limit in seconds. If None, use unlimited mode.
        """
        moves = self.root_state.get_all_possible_moves()
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        
        # Set current simulations to base_simulations (capped by max_simulations)
        self.current_simulations = min(self.base_simulations, self.max_simulations)
            
        print(f"🎯 Monte Carlo Domain: {len(moves)} moves, {self.current_simulations} simulations")
        
        # Use unified strategy selection for all time_limit cases
        return self._unified_search_strategy(moves, time_limit)
    
    def _optimized_tree_search(self, time_limit):
        """Optimized tree search - now delegates to unified system for consistency."""
        moves = self.root_state.get_all_possible_moves()
        
        # For small move sets, we can still use tree search approach
        # but through the unified tournament system for consistency
        return self._unified_search_strategy(moves, time_limit)
    
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
        """Legacy method - now delegates to unified search strategy."""
        moves = self.root_state.get_all_possible_moves()
        
        # Quick filtering to reduce search space for very large move sets
        if len(moves) > 15:
            moves = self._get_top_moves(self.root_state, moves, 15)
        
        # Delegate to unified strategy
        return self._unified_search_strategy(moves, time_limit)
    
    # Core unified tournament system - this is the only tournament method needed
    def _unified_search_strategy(self, moves, time_limit):
        """Unified strategy using single 3-round tournament for all cases.
        
        Always uses 3-round tournament with configuration (len(moves), 5, 3):
        - Round 1: Evaluate all moves 
        - Round 2: Top 5 moves (or fewer if less moves available)
        - Round 3: Top 3 moves (or fewer if less moves available)
        """
        # Get search configuration (always 3-round tournament)
        config = self._get_search_configuration(moves, time_limit)
        
        print(f"🔧 Strategy: {config['strategy']} | Budget: {config['simulation_budget']} sims | Rounds: {config['rounds']}")
        
        # Execute unified tournament with the determined configuration
        return self._execute_unified_tournament(moves, config)
    
    def _get_search_configuration(self, moves, time_limit):
        """Determine search configuration - always use 3-round tournament."""
        num_moves = len(moves)
        total_simulations = self.current_simulations
        
        # Always use 3-round tournament with (len(moves), 5, 3) configuration
        config = {
            'strategy': 'three_round_unified',
            'simulation_budget': total_simulations,
            'rounds': 3,
            'round_ratios': [self.round1_ratio, self.round2_ratio, self.round3_ratio],
            'candidates_per_round': [num_moves, min(5, num_moves), min(3, num_moves)],
            'min_sims_per_move': self.min_simulations_per_move
        }
        
        return config
    
    def _execute_unified_tournament(self, moves, config):
        """Execute unified tournament with given configuration."""
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
            
        strategy = config['strategy']
        rounds = config['rounds']
        round_ratios = config['round_ratios']
        candidates_per_round = config['candidates_per_round']
        total_simulations = config['simulation_budget']
        
        print(f"🏆 {strategy.replace('_', ' ').title()} Tournament: {len(moves)} moves, {total_simulations} total sims")
        
        current_moves = moves[:]
        round_scores = {}
        
        for round_num in range(rounds):
            # Each round gets the full budget (not multiplied)
            round_budget = total_simulations
            target_candidates = candidates_per_round[round_num] 
            
            # Each move in this round gets the full budget
            sims_per_move = max(config['min_sims_per_move'], round_budget)
            actual_round_sims = sims_per_move * len(current_moves)
            
            print(f"  Round {round_num + 1}: {len(current_moves)} moves × {sims_per_move} sims = {actual_round_sims} (budget: {round_budget})")
            
            # Evaluate all moves in current round
            round_scores = {}
            for i, move in enumerate(current_moves):
                score = self._evaluate_move_with_simulations(move, sims_per_move)
                round_scores[move] = score
            
            # Select candidates for next round (if not final round)
            if round_num < rounds - 1:
                # Always select exactly target_candidates for next round
                sorted_moves = sorted(round_scores.items(), key=lambda x: x[1], reverse=True)
                next_round_candidates = candidates_per_round[round_num + 1]
                current_moves = [move for move, _ in sorted_moves[:next_round_candidates]]
                    
                print(f"Advanced to next round: {len(current_moves)} moves")
        
        # Select final best move
        if round_scores:
            best_move = max(round_scores.items(), key=lambda x: x[1])[0]
            best_score = round_scores[best_move]
            
            # Calculate simulation utilization 
            total_used = 0
            for i in range(rounds):
                round_budget = total_simulations
                round_moves = candidates_per_round[i]
                sims_per_move = max(config['min_sims_per_move'], round_budget)
                total_used += sims_per_move * round_moves
            
            print(f"📊 Total simulations used: {total_used} across {rounds} rounds (each round gets full budget)")
            print(f"✅ {strategy.replace('_', ' ').title()} winner: {best_move} with score {best_score:.6f}")
            
            return best_move
        
        return moves[0] if moves else None
    
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
            self.current_simulations = self.base_simulations
            
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
