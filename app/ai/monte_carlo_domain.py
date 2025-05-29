#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IMPROVED Optimized Domain Knowledge-enhanced Monte Carlo Tree Search for Ataxx AI.

Major improvements over original:
1. Efficient resource allocation
2. Dynamic weight adaptation
3. Smart caching with memory management
4. Enhanced domain knowledge integration
5. Optimized tournament structure
"""
import random
import math
import time
from copy import deepcopy
from collections import defaultdict, deque
import gc

from app.ai.constants import CLONE_MOVE
from .monte_carlo_base import MonteCarloBase, MonteCarloNode

class MCTSDomainNode(MonteCarloNode):
    """Enhanced MCTS node with improved domain knowledge and efficient caching."""
    
    def __init__(self, state, parent=None, move=None, mcd_instance=None):
        super().__init__(state, parent, move)
        self.mcd_instance = mcd_instance
        
        # Optimized caching with LRU-style management
        self._domain_cache = {}
        self._max_cache_size = 100
        self._cache_access_order = deque()
        
        # Enhanced domain knowledge with confidence scores
        self._move_features = None
        self._position_features = None
        self._tactical_score = None
        self._strategic_score = None
        self._confidence = 0.0
        
        # Performance tracking
        self._last_update_time = time.time()
        self._evaluation_count = 0
    
    def is_terminal(self):
        """Check if this node represents a terminal game state."""
        return self.state.is_game_over()
    
    def is_fully_expanded(self):
        """Check if all possible moves have been tried."""
        if not hasattr(self, 'untried_moves') or self.untried_moves is None:
            self.untried_moves = self.state.get_all_possible_moves()
        return len(self.untried_moves) == 0
    
    def uct_value(self, exploration_weight=1.414):
        """Calculate UCT value with enhanced confidence factor."""
        if self.visits == 0:
            return float('inf')
        
        # Standard UCT components
        exploitation = self.ep / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        # Add confidence-based bonus
        confidence_bonus = self._confidence * 0.1
        
        return exploitation + exploration + confidence_bonus
    
    def select_child_with_enhanced_uct(self, exploration_weight=1.414, game_phase='midgame'):
        """Enhanced UCT selection with game phase awareness."""
        if not self.children:
            return None
        
        best_child = None
        best_value = float('-inf')
        
        # Adjust exploration based on game phase
        phase_multiplier = {
            'opening': 1.4,    # More exploration
            'midgame': 1.0,    # Balanced
            'endgame': 0.8     # More exploitation
        }
        adjusted_exploration = exploration_weight * phase_multiplier.get(game_phase, 1.0)
        
        for child in self.children:
            # Enhanced UCT value
            uct_value = child.uct_value(adjusted_exploration)
            
            # Domain knowledge enhancement with caching
            domain_bonus = self._get_cached_domain_bonus(child, game_phase)
            
            # Progressive bias with visit-based decay
            bias_strength = max(0.05, 0.5 / (1.0 + child.visits * 0.05))
            enhanced_value = uct_value + bias_strength * domain_bonus
            
            if enhanced_value > best_value:
                best_value = enhanced_value
                best_child = child
        
        return best_child
    
    def _get_cached_domain_bonus(self, child, game_phase):
        """Get domain bonus with intelligent caching."""
        if not child.move or not self.mcd_instance:
            return 0.0
        
        cache_key = f"{child.move}_{game_phase}"
        
        # Check cache first
        if cache_key in self._domain_cache:
            self._update_cache_access(cache_key)
            return self._domain_cache[cache_key]
        
        # Calculate domain bonus
        domain_bonus = self._calculate_enhanced_domain_bonus(child, game_phase)
        
        # Cache with LRU-style management
        self._cache_with_management(cache_key, domain_bonus)
        
        return domain_bonus
    
    def _calculate_enhanced_domain_bonus(self, child, game_phase):
        """Calculate comprehensive domain knowledge bonus with game phase adaptation."""
        # Multi-factor evaluation
        heuristic_score = self.mcd_instance._score_move(self.state, child.move)
        tactical_value = self._evaluate_enhanced_tactical_value(child.move)
        strategic_value = self._evaluate_strategic_value(child.move, game_phase)
        positional_value = self._evaluate_positional_value(child.move)
        
        # Dynamic weights based on game phase
        weights = self._get_phase_weights(game_phase)
        
        # Normalize components
        norm_heuristic = min(1.0, heuristic_score / 4.0)
        norm_tactical = min(1.0, tactical_value)
        norm_strategic = min(1.0, strategic_value)
        norm_positional = min(1.0, positional_value)
        
        # Weighted combination
        domain_bonus = (weights['heuristic'] * norm_heuristic +
                       weights['tactical'] * norm_tactical +
                       weights['strategic'] * norm_strategic +
                       weights['positional'] * norm_positional)
        
        # Update confidence based on consistency
        child._confidence = self._calculate_confidence(
            [norm_heuristic, norm_tactical, norm_strategic, norm_positional]
        )
        
        return domain_bonus
    
    def _get_phase_weights(self, game_phase):
        """Get dynamic weights based on game phase."""
        weights = {
            'opening': {
                'heuristic': 0.4, 'tactical': 0.2, 'strategic': 0.3, 'positional': 0.1
            },
            'midgame': {
                'heuristic': 0.5, 'tactical': 0.3, 'strategic': 0.1, 'positional': 0.1
            },
            'endgame': {
                'heuristic': 0.6, 'tactical': 0.3, 'strategic': 0.05, 'positional': 0.05
            }
        }
        return weights.get(game_phase, weights['midgame'])
    
    def _evaluate_enhanced_tactical_value(self, move):
        """Enhanced tactical evaluation."""
        if not self.mcd_instance:
            return 0.0
        
        dest_x, dest_y = move[1]
        
        # Immediate captures
        captures = self.mcd_instance._count_potential_captures(self.state, dest_x, dest_y)
        
        # Threat creation and defense
        threats_created = self.mcd_instance._count_threats_created(self.state, move)
        threats_blocked = self._count_threats_blocked(move)
        
        # Chain potential (multiple captures in sequence)
        chain_potential = self._evaluate_chain_potential(move)
        
        tactical_score = (captures * 0.4 + 
                         threats_created * 0.3 + 
                         threats_blocked * 0.2 + 
                         chain_potential * 0.1)
        
        return min(1.0, tactical_score / 3.0)
    
    def _evaluate_strategic_value(self, move, game_phase):
        """Evaluate strategic value based on game phase."""
        if game_phase == 'opening':
            return self._evaluate_opening_strategy(move)
        elif game_phase == 'midgame':
            return self._evaluate_midgame_strategy(move)
        else:  # endgame
            return self._evaluate_endgame_strategy(move)
    
    def _evaluate_opening_strategy(self, move):
        """Opening strategy: expansion and territory control."""
        dest_x, dest_y = move[1]
        
        # Center control
        center_distance = abs(dest_x - 3) + abs(dest_y - 3)
        center_value = (4 - center_distance) / 4.0
        
        # Expansion value
        expansion_value = 0.8 if move[0] == CLONE_MOVE else 0.3
        
        # Territory potential
        territory_value = self._evaluate_territory_potential(dest_x, dest_y)
        
        return (center_value * 0.4 + expansion_value * 0.3 + territory_value * 0.3)
    
    def _evaluate_midgame_strategy(self, move):
        """Midgame strategy: tactical advantage and positioning."""
        dest_x, dest_y = move[1]
        
        # Control of key positions
        key_position_value = self._evaluate_key_positions(dest_x, dest_y)
        
        # Mobility preservation
        mobility_value = self._evaluate_mobility_impact(move)
        
        # Enemy restriction
        restriction_value = self._evaluate_enemy_restriction(move)
        
        return (key_position_value * 0.4 + 
                mobility_value * 0.3 + 
                restriction_value * 0.3)
    
    def _evaluate_endgame_strategy(self, move):
        """Endgame strategy: maximizing captures and securing win."""
        # Direct material gain
        material_gain = self._evaluate_immediate_material(move)
        
        # Game ending potential
        ending_potential = self._evaluate_game_ending_potential(move)
        
        # Safety (avoiding counterattacks)
        safety_value = self._evaluate_move_safety(move)
        
        return (material_gain * 0.5 + 
                ending_potential * 0.3 + 
                safety_value * 0.2)
    
    def _evaluate_positional_value(self, move):
        """Evaluate positional advantages."""
        dest_x, dest_y = move[1]
        
        # Edge control
        edge_value = self._evaluate_edge_control(dest_x, dest_y)
        
        # Cluster formation
        cluster_value = self._evaluate_cluster_formation(dest_x, dest_y)
        
        # Future potential
        future_potential = self._evaluate_future_potential(dest_x, dest_y)
        
        return (edge_value * 0.3 + cluster_value * 0.4 + future_potential * 0.3)
    
    def _calculate_confidence(self, scores):
        """Calculate confidence based on score consistency."""
        if not scores:
            return 0.0
        
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        
        # Lower variance = higher confidence
        confidence = max(0.0, 1.0 - variance)
        return confidence
    
    def _cache_with_management(self, key, value):
        """Cache with LRU-style management."""
        if len(self._domain_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = self._cache_access_order.popleft()
            del self._domain_cache[oldest_key]
        
        self._domain_cache[key] = value
        self._cache_access_order.append(key)
    
    def _update_cache_access(self, key):
        """Update cache access order for LRU."""
        if key in self._cache_access_order:
            self._cache_access_order.remove(key)
            self._cache_access_order.append(key)
    
    # Helper methods for strategic evaluation (simplified implementations)
    def _count_threats_blocked(self, move):
        """Count enemy threats blocked by this move."""
        # Simplified implementation
        return 0.0
    
    def _evaluate_chain_potential(self, move):
        """Evaluate potential for chain captures."""
        # Simplified implementation
        return 0.0
    
    def _evaluate_territory_potential(self, x, y):
        """Evaluate territory control potential."""
        # Simplified implementation
        return 0.5
    
    def _evaluate_key_positions(self, x, y):
        """Evaluate control of key board positions."""
        # Simplified implementation
        return 0.5
    
    def _evaluate_mobility_impact(self, move):
        """Evaluate impact on future mobility."""
        # Simplified implementation
        return 0.5
    
    def _evaluate_enemy_restriction(self, move):
        """Evaluate how much this move restricts enemy options."""
        # Simplified implementation
        return 0.5
    
    def _evaluate_immediate_material(self, move):
        """Evaluate immediate material gain."""
        # Simplified implementation
        return 0.5
    
    def _evaluate_game_ending_potential(self, move):
        """Evaluate potential to end game favorably."""
        # Simplified implementation
        return 0.5
    
    def _evaluate_move_safety(self, move):
        """Evaluate safety of the move."""
        # Simplified implementation
        return 0.5
    
    def _evaluate_edge_control(self, x, y):
        """Evaluate edge control value."""
        # Simplified implementation
        return 0.5
    
    def _evaluate_cluster_formation(self, x, y):
        """Evaluate cluster formation potential."""
        # Simplified implementation
        return 0.5
    
    def _evaluate_future_potential(self, x, y):
        """Evaluate future move potential from this position."""
        # Simplified implementation
        return 0.5


class MonteCarloDomain(MonteCarloBase):
    """IMPROVED Monte Carlo Tree Search with Enhanced Domain Knowledge.
    
    Key improvements:
    1. Efficient resource allocation across tournament rounds
    2. Dynamic weight adaptation based on game state
    3. Smart memory management with cache cleanup
    4. Enhanced domain knowledge integration
    5. Optimized tournament structure with proper budgeting
    """
    
    def __init__(self, state, **kwargs):
        super().__init__(state, **kwargs)
        
        # Enhanced heuristic parameters with dynamic adaptation
        self.base_weights = {
            's1': kwargs.get('s1', 1.2),  # Capture importance
            's2': kwargs.get('s2', 0.5),  # Clustering value
            's3': kwargs.get('s3', 0.8),  # Clone bonus
            's4': kwargs.get('s4', 0.3)   # Jump penalty
        }
        self.current_weights = self.base_weights.copy()
        
        # MCTS optimization parameters
        self.use_tree_search = kwargs.get('use_tree_search', True)
        self.max_tree_depth = kwargs.get('max_tree_depth', 10)
        self.adaptive_exploration = kwargs.get('adaptive_exploration', True)
        self.progressive_bias = kwargs.get('progressive_bias', True)
        
        # Performance optimization
        self.use_move_ordering = kwargs.get('use_move_ordering', True)
        self.use_early_termination = kwargs.get('use_early_termination', True)
        self.simulation_depth_limit = kwargs.get('simulation_depth_limit', 12)
        
        # Simulation parameters with PROPER budgeting
        self.base_simulations = kwargs.get('basic_simulations', 600)
        self.max_simulations = kwargs.get('max_simulations', 50000)
        self.min_simulations_per_move = kwargs.get('min_simulations_per_move', 50)
        
        # FIXED: Proper tournament ratios that sum to 1.0
        self.round1_ratio = kwargs.get('round1_ratio', 0.5)   # 50% for round 1
        self.round2_ratio = kwargs.get('round2_ratio', 0.3)   # 30% for round 2
        self.round3_ratio = kwargs.get('round3_ratio', 0.2)   # 20% for round 3
        
        # Normalize ratios
        self._normalize_tournament_ratios()
        
        # Enhanced caching with memory management
        self._evaluation_cache = {}
        self._move_score_cache = {}
        self._position_hash_cache = {}
        self._cache_cleanup_frequency = 1000
        self._cache_access_count = 0
        
        # Adaptive parameters with learning
        self._game_phase_history = deque(maxlen=10)
        self._move_quality_history = deque(maxlen=20)
        self._exploration_decay = 1.0
        
        # Enhanced statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_cleanups': 0,
            'early_terminations': 0,
            'deep_searches': 0,
            'weight_adaptations': 0,
            'total_simulations_used': 0,
            'average_simulation_efficiency': 0.0
        }
        
        print(f"🚀 ImprovedMCTSDomain initialized with enhanced optimizations")
        print(f"Domain weights: {self.current_weights}")
        print(f"Tournament ratios: R1={self.round1_ratio:.2f}, R2={self.round2_ratio:.2f}, R3={self.round3_ratio:.2f}")
    
    def _normalize_tournament_ratios(self):
        """Ensure tournament ratios sum to 1.0."""
        total = self.round1_ratio + self.round2_ratio + self.round3_ratio
        if abs(total - 1.0) > 0.001:
            self.round1_ratio /= total
            self.round2_ratio /= total
            self.round3_ratio /= total
    
    def get_move(self, time_limit=None):
        """Get best move using improved strategy selection system."""
        moves = self.root_state.get_all_possible_moves()
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        
        # Detect and adapt to game phase
        game_phase = self._detect_enhanced_game_phase()
        self._adapt_weights_to_phase(game_phase)
        
        # Set simulation budget
        self.current_simulations = min(self.base_simulations, self.max_simulations)
        
        print(f"🎯 Improved Monte Carlo Domain: {len(moves)} moves, {self.current_simulations} sims, Phase: {game_phase}")
        
        # Use improved tournament system
        best_move = self._improved_tournament_search(moves, time_limit, game_phase)
        
        # Update learning history
        self._update_learning_history(game_phase, best_move)
        
        # Periodic cache cleanup
        self._periodic_cache_cleanup()
        
        return best_move
    
    def _improved_tournament_search(self, moves, time_limit, game_phase):
        """Improved tournament system with escalating simulation budget per round."""
        if len(moves) == 1:
            return moves[0]
        
        # Pre-filter moves for efficiency
        if len(moves) > 12:
            moves = self._intelligent_move_filtering(moves, 12)
        
        # NEW: Escalating simulation budget per move per round
        base_budget = self.current_simulations
        round_multipliers = {
            1: 1.0,    # Round 1: each move gets base_budget simulations
            2: 1.5,    # Round 2: each move gets 1.5 × base_budget simulations
            3: 2.0     # Round 3: each move gets 2.0 × base_budget simulations
        }
        
        print(f"🏆 Escalating Tournament: Base budget={base_budget}")
        print(f"   Round multipliers: R1={round_multipliers[1]}x, R2={round_multipliers[2]}x, R3={round_multipliers[3]}x")
        
        # Execute tournament rounds
        current_moves = moves[:]
        round_scores = {}
        total_simulations_used = 0
        
        for round_num in range(1, 4):
            candidates = self._get_round_candidates(current_moves, round_num)
            
            if not candidates:
                break
            
            # Each move gets: base_budget × round_multiplier simulations
            sims_per_move = int(base_budget * round_multipliers[round_num])
            sims_per_move = max(self.min_simulations_per_move, sims_per_move)
            
            actual_round_sims = sims_per_move * len(candidates)
            total_simulations_used += actual_round_sims
            
            print(f"  Round {round_num}: {len(candidates)} moves × {sims_per_move} sims = {actual_round_sims:,}")
            
            # Evaluate moves with escalating simulation budget
            round_scores = {}
            for move in candidates:
                score = self._evaluate_move_with_enhanced_simulations(
                    move, sims_per_move, game_phase
                )
                round_scores[move] = score
            
            # Select candidates for next round
            if round_num < 3:
                sorted_moves = sorted(round_scores.items(), key=lambda x: x[1], reverse=True)
                next_round_size = self._get_next_round_size(round_num, len(candidates))
                current_moves = [move for move, _ in sorted_moves[:next_round_size]]
                print(f"    Advanced to Round {round_num + 1}: {len(current_moves)} moves")
        
        # Select final winner
        if round_scores:
            best_move = max(round_scores.items(), key=lambda x: x[1])[0]
            best_score = round_scores[best_move]
            
            # Update statistics
            self.stats['total_simulations_used'] += total_simulations_used
            efficiency = best_score * (base_budget / max(1, total_simulations_used / len(moves)))
            self.stats['average_simulation_efficiency'] = (
                (self.stats['average_simulation_efficiency'] + efficiency) / 2
            )
            
            print(f"✅ Tournament winner: {best_move} with score {best_score:.6f}")
            print(f"📊 Total simulations used: {total_simulations_used:,} (base: {base_budget})")
            print(f"📊 Escalation factor: {total_simulations_used / base_budget:.1f}x")
            
            return best_move
        
        return moves[0]
    
    def _get_round_candidates(self, moves, round_num):
        """Get candidate moves for each tournament round."""
        if round_num == 1:
            return moves  # All moves in round 1
        elif round_num == 2:
            return moves[:min(5, len(moves))]  # Top 5 for round 2
        else:  # round_num == 3
            return moves[:min(3, len(moves))]  # Top 3 for round 3
    
    def _get_next_round_size(self, current_round, current_size):
        """Determine size of next round."""
        if current_round == 1:
            return min(5, current_size)
        elif current_round == 2:
            return min(3, current_size)
        return current_size
    
    def _intelligent_move_filtering(self, moves, target_count):
        """Intelligently filter moves to reduce search space."""
        if len(moves) <= target_count:
            return moves
        
        # Quick scoring for filtering
        scored_moves = []
        for move in moves:
            quick_score = (
                self._quick_move_score(self.root_state, move) * 0.7 +
                self._get_tactical_bonus(self.root_state, move) * 0.3
            )
            scored_moves.append((move, quick_score))
        
        # Sort and select top moves
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Include some randomness to avoid deterministic filtering
        top_moves = [move for move, _ in scored_moves[:target_count]]
        
        # Add one random move from remaining to maintain diversity
        if len(scored_moves) > target_count:
            remaining = [move for move, _ in scored_moves[target_count:]]
            if remaining:
                random_move = random.choice(remaining)
                top_moves[-1] = random_move  # Replace last move with random
        
        return top_moves
    
    def _evaluate_move_with_enhanced_simulations(self, move, simulations, game_phase):
        """Enhanced move evaluation with game phase awareness."""
        next_state = deepcopy(self.root_state)
        next_state.move_with_position(move)
        next_state.toggle_player()
        
        # Create enhanced node
        root = MCTSDomainNode(next_state, mcd_instance=self)
        original_player = self.root_state.current_player()
        
        # Phase-adaptive exploration
        exploration_param = self._get_phase_adaptive_exploration(game_phase)
        
        # Enhanced MCTS simulation
        for i in range(simulations):
            leaf = self._enhanced_selection_with_phase(root, exploration_param, game_phase)
            
            if not leaf.is_terminal() and leaf.visits > 0:
                leaf = self._intelligent_expansion_with_phase(leaf, game_phase)
            
            result = self._enhanced_simulation_with_phase(leaf, original_player, game_phase)
            self._backpropagate_with_learning(leaf, result)
        
        # Calculate confidence-weighted score
        if root.visits == 0:
            return 0.5
        
        base_score = root.ep / root.visits
        confidence_factor = min(1.0, root.visits / (simulations * 0.8))
        
        # Combine with static evaluation
        static_score = self._score_move_with_phase(self.root_state, move, game_phase)
        normalized_static = static_score / (1.0 + static_score)
        
        final_score = (confidence_factor * base_score + 
                      (1 - confidence_factor) * 0.3 * normalized_static)
        
        return final_score
    
    def _detect_enhanced_game_phase(self):
        """Enhanced game phase detection with context."""
        total_pieces = sum(self.root_state.balls.values())
        empty_spaces = 49 - total_pieces  # 7x7 board minus pieces
        
        # Multi-factor phase detection
        if total_pieces <= 8:
            phase = 'opening'
        elif total_pieces >= 40 or empty_spaces <= 8:
            phase = 'endgame'
        else:            
            phase = 'midgame'
        
        return phase
    
    def _adapt_weights_to_phase(self, game_phase):
        """Dynamically adapt heuristic weights based on game phase."""
        adaptations = {
            'opening': {'s1': 0.8, 's2': 0.7, 's3': 1.2, 's4': 0.2},
            'midgame': {'s1': 1.2, 's2': 0.5, 's3': 0.8, 's4': 0.3},
            'endgame': {'s1': 1.6, 's2': 0.3, 's3': 0.4, 's4': 0.5}
        }
        
        if game_phase in adaptations:
            old_weights = self.current_weights.copy()
            self.current_weights.update(adaptations[game_phase])
            
            # Check if weights actually changed
            if old_weights != self.current_weights:
                self.stats['weight_adaptations'] += 1
                print(f"🔧 Adapted weights for {game_phase}: {self.current_weights}")
    
    def _get_phase_adaptive_exploration(self, game_phase):
        """Get exploration parameter adapted to game phase."""
        base_exploration = 1.414
        
        phase_multipliers = {
            'opening': 1.6,
            'midgame': 1.0,
            'endgame': 0.6
        }
        
        return base_exploration * phase_multipliers.get(game_phase, 1.0)
    
    def _enhanced_selection_with_phase(self, node, exploration_param, game_phase):
        """Enhanced selection with game phase awareness."""
        while not node.is_terminal() and node.children:
            if not node.is_fully_expanded():
                return node
            node = node.select_child_with_enhanced_uct(exploration_param, game_phase)
        return node
    
    def _intelligent_expansion_with_phase(self, node, game_phase):
        """Intelligent expansion with phase-specific move ordering."""
        if not node.untried_moves:
            node.untried_moves = node.state.get_all_possible_moves()
        
        if not node.untried_moves:
            return node
        
        # Phase-specific move selection
        best_move = self._select_best_expansion_move_with_phase(
            node.state, node.untried_moves, game_phase
        )
        
        # Create child node
        next_state = deepcopy(node.state)
        next_state.move_with_position(best_move)
        next_state.toggle_player()
        
        child = MCTSDomainNode(
            next_state, parent=node, move=best_move, mcd_instance=self
        )
        node.children.append(child)
        node.untried_moves.remove(best_move)
        
        return child
    
    def _enhanced_simulation_with_phase(self, node, original_player, game_phase):
        """Enhanced simulation with phase-specific strategy."""
        if node.is_terminal():
            return self._evaluate_final_position(node.state, original_player)
        
        # Phase-specific simulation depth
        max_depth = {'opening': 4, 'midgame': 3, 'endgame': 2}.get(game_phase, 3)
        
        # Multi-component evaluation
        heuristic_eval = self._fast_position_evaluation_with_phase(
            node.state, original_player, game_phase
        )
        tactical_eval = self._tactical_simulation_with_phase(
            node.state, original_player, game_phase, max_depth
        )
        
        # Phase-specific weight combination
        if game_phase in ['opening']:
            combined_score = 0.6 * heuristic_eval + 0.4 * tactical_eval
        elif game_phase == 'endgame':
            combined_score = 0.8 * tactical_eval + 0.2 * heuristic_eval
        else:
            combined_score = 0.7 * heuristic_eval + 0.3 * tactical_eval
        
        return max(0.0, min(1.0, combined_score))
    
    def _backpropagate_with_learning(self, node, result):
        """Backpropagation with learning-based updates."""
        current_result = result
        depth = 0
        
        while node is not None:
            # Standard update
            node.update(current_result)
            
            # Learning-based confidence update
            if hasattr(node, '_confidence'):
                visits_factor = min(1.0, node.visits / 50.0)
                node._confidence = (node._confidence * 0.9 + visits_factor * 0.1)
            
            # Apply result decay for alternating players
            current_result = 1.0 - current_result
            node = node.parent
            depth += 1
    
    def _periodic_cache_cleanup(self):
        """Periodic cache cleanup to prevent memory bloat."""
        self._cache_access_count += 1
        
        if self._cache_access_count % self._cache_cleanup_frequency == 0:
            # Clean up evaluation cache
            if len(self._evaluation_cache) > 5000:
                # Keep only recent 60% of entries
                items = list(self._evaluation_cache.items())
                keep_count = int(len(items) * 0.6)
                self._evaluation_cache = dict(items[-keep_count:])
                
            # Clean up move score cache
            if len(self._move_score_cache) > 3000:
                items = list(self._move_score_cache.items())
                keep_count = int(len(items) * 0.6)
                self._move_score_cache = dict(items[-keep_count:])
            
            self.stats['cache_cleanups'] += 1
            
            # Force garbage collection
            gc.collect()
    
    def _update_learning_history(self, game_phase, selected_move):
        """Update learning history for future improvements."""
        self._game_phase_history.append(game_phase)
        
        # Evaluate quality of selected move
        move_quality = self._score_move_with_phase(self.root_state, selected_move, game_phase)
        self._move_quality_history.append(move_quality)
    
    # Enhanced core methods with phase awareness
    def _score_move_with_phase(self, state, move, game_phase):
        """Score move with phase-specific weights."""
        player = state.current_player()
        
        # Create a copy to simulate the move
        next_state = deepcopy(state)
        next_state.move_with_position(move)
        
        # s1 term: enemy captures
        enemy_captures = state.balls[-player] - next_state.balls[-player]
        
        # Get positions
        dest_x, dest_y = move[1]
        
        # s2 term: clustering
        own_around_target = self._count_adjacent_pieces(state, dest_x, dest_y, player)
        
        # s3 and s4 terms
        if move[0] == CLONE_MOVE:
            move_type_bonus = self.current_weights['s3']
            source_penalty = 0
        else:
            move_type_bonus = 0
            source_x, source_y = move[2]
            own_around_source = self._count_adjacent_pieces(state, source_x, source_y, player)
            source_penalty = self.current_weights['s4'] * own_around_source
        
        # Calculate score with current (phase-adapted) weights
        score = (self.current_weights['s1'] * enemy_captures + 
                self.current_weights['s2'] * own_around_target + 
                move_type_bonus - 
                source_penalty)
        
        return max(0.0, score)
    
    def _fast_position_evaluation_with_phase(self, state, original_player, game_phase):
        """Fast position evaluation with phase-specific factors."""
        # Check cache with phase key
        state_hash = hash(str(state.board) + game_phase)
        if state_hash in self._evaluation_cache:
            self.stats['cache_hits'] += 1
            return self._evaluation_cache[state_hash]
        
        self.stats['cache_misses'] += 1
        
        # Multi-factor evaluation
        material_score = self._evaluate_material_balance(state, original_player)
        mobility_score = self._evaluate_mobility(state, original_player)
        position_score = self._evaluate_position_quality_with_phase(state, original_player, game_phase)
        threat_score = self._evaluate_threat_balance(state, original_player)
        
        # Phase-specific weights
        weights = self._get_evaluation_weights_for_phase(game_phase)
        
        combined_score = (weights[0] * material_score +
                         weights[1] * mobility_score +
                         weights[2] * position_score +
                         weights[3] * threat_score)
        
        # Cache result
        self._evaluation_cache[state_hash] = combined_score
        
        return combined_score
    
    def _get_evaluation_weights_for_phase(self, game_phase):
        """Get evaluation weights specific to game phase."""
        phase_weights = {
            'opening': [0.2, 0.3, 0.4, 0.1],
            'midgame': [0.4, 0.2, 0.2, 0.2],
            'endgame': [0.6, 0.1, 0.2, 0.1]
        }
        return phase_weights.get(game_phase, [0.4, 0.2, 0.2, 0.2])
    
    def print_enhanced_stats(self):
        """Print comprehensive performance statistics."""
        print(f"\n📊 Enhanced Performance Statistics:")
        print(f"  Cache Performance:")
        print(f"    Hits: {self.stats['cache_hits']}")
        print(f"    Misses: {self.stats['cache_misses']}")
        hit_rate = self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
        print(f"    Hit rate: {hit_rate:.2%}")
        print(f"    Cleanups: {self.stats['cache_cleanups']}")
        
        print(f"  Search Performance:")
        print(f"    Total simulations used: {self.stats['total_simulations_used']}")
        print(f"    Average efficiency: {self.stats['average_simulation_efficiency']:.4f}")
        print(f"    Weight adaptations: {self.stats['weight_adaptations']}")
        
        print(f"  Current State:")
        print(f"    Phase: {self._detect_enhanced_game_phase()}")
        print(f"    Current weights: {self.current_weights}")
        
        if self._move_quality_history:
            avg_quality = sum(self._move_quality_history) / len(self._move_quality_history)
            print(f"    Average move quality: {avg_quality:.3f}")
    
    # Delegate missing methods to maintain compatibility
    def _score_move(self, state, move):
        """Delegate to phase-aware scoring."""
        game_phase = self._detect_enhanced_game_phase()
        return self._score_move_with_phase(state, move, game_phase)
    
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
    
    def _quick_move_score(self, state, move):
        """Quick move scoring for filtering."""
        cache_key = str(move) + str(hash(str(state.board)))
        if cache_key in self._move_score_cache:
            return self._move_score_cache[cache_key]
        
        # Fast scoring
        dest_x, dest_y = move[1]
        
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
    
    # Additional delegated methods (simplified implementations for compatibility)
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
    
    def _evaluate_position_quality_with_phase(self, state, player, game_phase):
        """Evaluate position quality with phase awareness."""
        score = 0.0
        count = 0
        
        for x in range(7):
            for y in range(7):
                if state.board[x][y] == player:
                    count += 1
                    
                    # Phase-specific position evaluation
                    if game_phase in ['opening']:
                        # Focus on center control
                        center_distance = abs(x - 3) + abs(y - 3)
                        score += (7 - center_distance) / 7.0
                    else:
                        # Focus on clustering and edge control
                        neighbors = self._count_adjacent_pieces(state, x, y, player)
                        score += neighbors * 0.2
                        
                        # Edge bonus in endgame
                        if game_phase == 'endgame':
                            edge_distance = min(x, 6-x, y, 6-y)
                            if edge_distance == 0:
                                score += 0.1
        
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
    
    def _select_best_expansion_move_with_phase(self, state, untried_moves, game_phase):
        """Select best move for expansion with phase awareness."""
        if len(untried_moves) == 1:
            return untried_moves[0]
        
        best_move = None
        best_score = -1
        
        for move in untried_moves:
            score = self._score_move_with_phase(state, move, game_phase)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move if best_move else untried_moves[0]
    
    def _tactical_simulation_with_phase(self, state, original_player, game_phase, max_depth=3):
        """Tactical simulation with phase-specific strategy."""
        sim_state = deepcopy(state)
        depth = 0
        
        while not sim_state.is_game_over() and depth < max_depth:
            moves = sim_state.get_all_possible_moves()
            if not moves:
                break
            
            # Phase-specific move selection
            move = self._select_simulation_move_with_phase(sim_state, moves, game_phase)
            sim_state.move_with_position(move)
            sim_state.toggle_player()
            depth += 1
        
        return self._evaluate_final_position(sim_state, original_player)
    
    def _select_simulation_move_with_phase(self, state, moves, game_phase):
        """Select move for simulation with phase-specific bias."""
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        
        # Limit moves for performance
        if len(moves) > 6:
            moves = self._get_top_moves_for_simulation(state, moves, 6)
        
        # Phase-specific scoring
        enhanced_scores = []
        for move in moves:
            base_score = self._score_move_with_phase(state, move, game_phase)
            tactical_bonus = self._get_tactical_bonus(state, move)
            
            # Phase-specific weighting
            if game_phase == 'endgame':
                enhanced_score = base_score * 0.8 + tactical_bonus * 0.2
            else:
                enhanced_score = base_score * 0.6 + tactical_bonus * 0.4
            
            enhanced_scores.append(max(0.1, enhanced_score))
        
        # Temperature-based selection
        temperature = {'opening': 1.0, 'midgame': 0.8, 'endgame': 0.6}.get(game_phase, 0.8)
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
    
    def _get_top_moves_for_simulation(self, state, moves, count):
        """Get top moves for simulation."""
        if len(moves) <= count:
            return moves
        
        game_phase = self._detect_enhanced_game_phase()
        scored_moves = [
            (move, self._score_move_with_phase(state, move, game_phase)) 
            for move in moves
        ]
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        return [move for move, _ in scored_moves[:count]]
    
    def _evaluate_final_position(self, state, original_player):
        """Evaluate final position."""
        if state.is_game_over():
            my_pieces = state.balls[original_player]
            opp_pieces = state.balls[-original_player]
            
            if my_pieces > opp_pieces:
                return 1.0
            elif my_pieces < opp_pieces:
                return 0.0
            else:
                return 0.5
        
        # Not terminal, use heuristic
        return self._evaluate_material_balance(state, original_player)
