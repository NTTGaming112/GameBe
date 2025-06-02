"""
MONTE CARLO DOMAIN - TOURNAMENT SYSTEM INTEGRATED
================================================

Hoàn toàn tích hợp tournament system với ba vòng thi đấu:
1. Vòng sơ loại: Đánh giá sơ bộ tất cả moves
2. Vòng mô phỏng: Simulation chi tiết cho top candidates
3. Vòng chung kết: Head-to-head với parallel processing

Hệ thống điểm số tổng hợp:
- Heuristic Score (H): s1×A + s2×B + s3×C - s4×P
- Tactical Score (T): Corner avoidance + No suicide + Max convert  
- Strategic Score (S): Tempo + Enemy isolation + Mobility reduction
- Simulation Score: Monte Carlo rollouts

Phase-adaptive weights: α×H + β×T + γ×S + δ×Sim
"""

import random
import math
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from .monte_carlo_base import MonteCarloBase
from app.ai.ataxx_state import Ataxx, PLAYER_ONE, PLAYER_TWO
from .constants import (
    TOURNAMENT_CONFIG, PHASE_BONUS_PENALTY, MCD_PHASE_WEIGHTS,
    ADJACENT_POSITIONS, JUMP_POSITIONS, TEMPERATURE_SCHEDULE, COMPONENT_WEIGHTS,
    PHASE_ADAPTIVE_HEURISTIC_COEFFS, WIN_BONUS_EARLY, WIN_BONUS_FULL_BOARD
)

def softmax_with_temperature(scores, temperature=1.0):
    """
    Apply softmax probability distribution with temperature control
    
    Args:
        scores: List of numerical scores
        temperature: Temperature parameter (higher = more random, lower = more deterministic)
                    - temperature > 1.0: More exploration (flatter distribution)
                    - temperature < 1.0: More exploitation (sharper distribution)
                    - temperature = 1.0: Standard softmax
                    - temperature → 0: Deterministic (max score gets probability 1)
                    - temperature → ∞: Uniform distribution
    
    Returns:
        List of probabilities summing to 1.0
    """

    temperature = TEMPERATURE_SCHEDULE[scores[0].get('phase', 'mid')] if isinstance(scores[0], dict) else temperature
    if not scores:
        return []
    
    if temperature <= 0:
        # Deterministic selection - max score gets probability 1
        max_idx = max(range(len(scores)), key=lambda i: scores[i])
        probs = [0.0] * len(scores)
        probs[max_idx] = 1.0
        return probs
    
    # Apply temperature scaling
    scaled_scores = [score / temperature for score in scores]
    
    # Subtract max for numerical stability
    max_score = max(scaled_scores)
    exp_scores = [math.exp(score - max_score) for score in scaled_scores]
    
    # Calculate softmax probabilities
    sum_exp = sum(exp_scores)
    if sum_exp == 0:
        # Fallback to uniform distribution
        return [1.0 / len(scores)] * len(scores)
    
    probabilities = [exp_score / sum_exp for exp_score in exp_scores]
    return probabilities

def sample_from_probabilities(items, probabilities):
    """
    Sample an item based on probability distribution
    
    Args:
        items: List of items to sample from
        probabilities: List of probabilities for each item
    
    Returns:
        Selected item
    """
    if not items or not probabilities:
        return None
    
    if len(items) != len(probabilities):
        raise ValueError("Items and probabilities must have same length")
    
    # Cumulative probability distribution
    cumulative = []
    total = 0.0
    for prob in probabilities:
        total += prob
        cumulative.append(total)
    
    # Random selection
    rand_val = random.random() * total
    for i, cum_prob in enumerate(cumulative):
        if rand_val <= cum_prob:
            return items[i]
    
    # Fallback to last item
    return items[-1]

def _parallel_simulation_worker(args):
    """Worker function cho parallel processing - Thread safe"""
    state_data, move, num_simulations, original_player = args
    
    if state_data is None or move is None:
        return 0.0
        
    # Reconstruct state
    state = Ataxx()
    state.player1_board = state_data['player1_board']
    state.player2_board = state_data['player2_board']
    state.balls = state_data['balls'].copy()
    state.turn_player = state_data['turn_player']
    
    total_score = 0.0
    
    for _ in range(num_simulations):
        new_state = deepcopy(state)
        new_state.move_with_position(move)
        new_state.toggle_player()
        
        # Use enhanced rollout simulation
        simulation_score = _rollout_simulation(new_state, original_player)
        total_score += simulation_score
       
            
    return total_score / num_simulations

def _rollout_simulation(state, original_player):
    """Enhanced rollout simulation with proper final evaluation"""
    simulation_state = deepcopy(state)
    depth = 0
    max_depth = 20
    
    while depth < max_depth and not simulation_state.is_game_over():
        moves = simulation_state.get_all_possible_moves()
        if not moves:
            simulation_state.toggle_player()
            moves = simulation_state.get_all_possible_moves()
            if not moves:
                break
                
        move = random.choice(moves)
        simulation_state.move_with_position(move)
        simulation_state.toggle_player()
        depth += 1
        
    # Use enhanced evaluation for final position
    return _evaluate_final_position_static(simulation_state, original_player)

def _evaluate_final_position_static(state, player):
    """Static version of final position evaluation"""
    num_own = state.balls[player]
    num_opp = state.balls[-player]
    score = num_own - num_opp
    
    if state.is_game_over():
        winner = state.get_winner()
        if winner == player:
            total_pieces = num_own + num_opp
            empty_spaces = 49 - total_pieces
            score += WIN_BONUS_FULL_BOARD if empty_spaces == 0 else WIN_BONUS_EARLY
        elif winner == -player:
            total_pieces = num_own + num_opp
            empty_spaces = 49 - total_pieces
            score -= WIN_BONUS_FULL_BOARD if empty_spaces == 0 else WIN_BONUS_EARLY
        elif winner == 100:  # Draw
            score = 0
    
    # Normalize score to [0, 1] range
    return (score + 549) / 1098

class MonteCarloDomain(MonteCarloBase):
    """
    Monte Carlo Domain với Tournament System hoàn chỉnh
    
    Triển khai tournament 3 vòng với hệ thống điểm tổng hợp:
    - Round 1: Preliminary evaluation
    - Round 2: Intensive simulation 
    - Round 3: Final head-to-head elimination
    """
    
    def __init__(self, state, exploration_factor=1.41, max_iterations=1000, 
                 time_limit=5.0, use_parallel=True, basic_simulations=300,
                 component_weights=None, **kwargs):
        super().__init__(state, **kwargs)
        
        # Debug: kiểm tra state
        if state is None:
            print("Warning: state is None in MonteCarloDomain constructor")
        else:
            print(f"MonteCarloDomain initialized with state type: {type(state)}")
        self.time_limit = time_limit
        self.use_parallel = use_parallel
        self.basic_simulations = basic_simulations
        
        # Set component weights with defaults from constants
        self.component_weights = component_weights or COMPONENT_WEIGHTS.copy()
        
        self.performance_stats = {
            'tournaments_run': 0,
            'parallel_evaluations': 0,
            'avg_tournament_time': 0.0,
            'total_moves_evaluated': 0
        }
        
    def get_move(self, time_limit=None):
        """Interface tương thích với base class"""
        return self.get_mcd_move(time_limit)
        
    def get_mcd_move(self, time_limit=None):
        """Main entry point - Tournament system move selection"""
        start_time = time.time()
        state = self.root_state
        
        if state is None:
            print("Error: root_state is None")
            return None
        
        moves = state.get_all_possible_moves()
        if not moves:
            return None
            
        if len(moves) == 1:
            return moves[0]
            
        # Run full tournament system
        best_move = self._execute_tournament_rounds(state, moves)
        
        # Update performance statistics
        self.performance_stats['tournaments_run'] += 1
        tournament_time = time.time() - start_time
        self.performance_stats['avg_tournament_time'] = (
            (self.performance_stats['avg_tournament_time'] * 
             (self.performance_stats['tournaments_run'] - 1) + tournament_time) /
            self.performance_stats['tournaments_run']
        )
        self.performance_stats['total_moves_evaluated'] += len(moves)
        
        return best_move
         
    def _execute_tournament_rounds(self, state, moves):
        """
        Execute complete tournament system
        
        🏆 Round 1: Preliminary evaluation of all moves
        🎯 Round 2: Intensive simulation for top K1 candidates
        🏁 Round 3: Final elimination with parallel processing
        """
        print(f"\n🏆 TOURNAMENT SYSTEM START - {len(moves)} moves competing")
        
        k1 = min(TOURNAMENT_CONFIG['K1'], len(moves))
        k2 = min(TOURNAMENT_CONFIG['K2'], k1)
        
        phase = self._detect_game_phase(state)
        weights = MCD_PHASE_WEIGHTS[phase]
        
        print(f"🎮 Game phase: {phase.upper()}")
        print(f"⚖️  Component weights: α={weights['alpha']:.1f} β={weights['beta']:.1f} "
              f"γ={weights['gamma']:.1f} δ={weights['delta']:.1f}")
        
        # Show phase-adaptive bonus/penalty values
        phase_config = PHASE_BONUS_PENALTY[phase]
        print(f"🎯 Phase bonuses: Clone={phase_config['clone_bonus']:.1f}, "
              f"Jump penalty={phase_config['jump_penalty']:.1f}")
        
        # Calculate simulation counts for each round
        round1_sims = int(self.basic_simulations * TOURNAMENT_CONFIG['ROUND1_SIM_RATIO'])
        round2_sims = int(self.basic_simulations * TOURNAMENT_CONFIG['ROUND2_SIM_RATIO'])
        round3_sims = int(self.basic_simulations * TOURNAMENT_CONFIG['ROUND3_SIM_RATIO'])
        
        print(f"🎯 Simulation counts: R1={round1_sims}, R2={round2_sims}, R3={round3_sims} "
              f"(base={self.basic_simulations})")
        
        # =================== ROUND 1: PRELIMINARY ===================
        print(f"\n📊 ROUND 1: Preliminary Evaluation ({len(moves)} moves, {round1_sims} sims each)")
        
        # Convert moves to candidate format for unified function
        round1_candidates = [(move, 0.0, 0.0, 0.0, 0.0, 0.0) for move in moves]
        
        # Use unified function for Round 1
        round1_results = self._execute_tournament_round(
            state, round1_candidates, weights,
            round_name="Round 1 (Preliminary)",
            sim_ratio=TOURNAMENT_CONFIG['ROUND1_SIM_RATIO'],
            is_first_round=True
        )
        
        # Sort và advance top K1
        round1_results.sort(key=lambda x: x[1], reverse=True)
        round2_candidates = round1_results[:k1]
        
        print(f"\n✅ ROUND 1 COMPLETE - Top {k1} moves advance:")
        for i, (move, score, sim) in enumerate(round2_candidates):
            print(f"  {i+1}. {move}: {score:.3f}")
            
        if k1 == 1:
            winner = round2_candidates[0][0]
            print(f"\n🥇 SINGLE WINNER: {winner}")
            return winner
        
        # =================== ROUND 2: COMBINED SCORING ===================
        print(f"\n🎯 ROUND 2: Combined Scoring & Simulation ({k1} moves, {round2_sims} sims each)")
        
        # Use unified function for Round 2
        round2_results = self._execute_tournament_round(
            state, round2_candidates, weights, 
            round_name="Round 2", 
            sim_ratio=TOURNAMENT_CONFIG['ROUND2_SIM_RATIO']
        )
        
        # Sort và advance top K2
        round2_results.sort(key=lambda x: x[1], reverse=True)
        round3_candidates = round2_results[:k2]
        
        print(f"\n✅ ROUND 2 COMPLETE - Top {k2} moves advance:")
        for i, (move, score, sim) in enumerate(round3_candidates):
            print(f"  {i+1}. {move}: {score:.3f}")
            
        if k2 == 1:
            winner = round3_candidates[0][0]
            print(f"\n🥇 SINGLE WINNER: {winner}")
            return winner
        
        # =================== ROUND 3: FINAL ===================
        print(f"\n🏁 ROUND 3: Final Elimination ({k2} moves, {round3_sims} sims each)")
        
        # Use unified function for Round 3
        final_scores = self._execute_tournament_round(
            state, round3_candidates, weights,
            round_name="Round 3 (Final)",
            sim_ratio=TOURNAMENT_CONFIG['ROUND3_SIM_RATIO']
        )
        
        # Determine tournament winner
        winner_move = max(final_scores, key=lambda x: x[1])[0]
        winner_score = max(final_scores, key=lambda x: x[1])[1]
        
        print(f"\n🥇 TOURNAMENT WINNER: {winner_move} (score: {winner_score:.3f})")
        return winner_move
    
    def _calculate_heuristic_score(self, state, move):
        """
        Heuristic Score: H = s1×A + s2×B + s3×C - s4×P
        
        A = Attack moves (capture enemy pieces)
        B = Friendly neighbors (adjacent friendly pieces)
        C = Clone bonus (reward for adjacent moves)
        P = Jump penalty (penalize long jumps)
        """ 
        from_pos, to_pos = move

        # Special handling for clone moves (from_pos is None)
        if from_pos is None:
            # This is a clone move, distance is 0
            distance = 0
        else:
            distance = self._manhattan_distance(from_pos, to_pos)
        
        # C: Attack component (captured pieces)
        temp_state = deepcopy(state)
        original_enemy = self._count_enemy_pieces(temp_state, state.turn_player)
        temp_state.move_with_position(move)
        new_enemy = self._count_enemy_pieces(temp_state, state.turn_player)
        A = original_enemy - new_enemy
        
        # B: Friendly neighbors component
        B = self._count_friendly_neighbors(state, to_pos, state.turn_player)
        
        # C: Clone bonus (reward for adjacent moves)
        C = 1.0 if distance <= 1 else 0.0

        # P: Jump penalty (penalize long jumps)
        P = 1.0 if distance > 1 else 0
        
        # Apply heuristic formula
        phase = self._detect_game_phase(state)
        heuristic_coeffs = PHASE_ADAPTIVE_HEURISTIC_COEFFS[phase]
        h_raw = (heuristic_coeffs['s1'] * A + 
                heuristic_coeffs['s2'] * B + 
                heuristic_coeffs['s3'] * C - 
                heuristic_coeffs['s4'] * P)

        # Sigmoid normalization
        return self._sigmoid_normalize(h_raw)
    
    def _calculate_tactical_score(self, state, move):
        """
        Tactical Score: Corner avoidance + No suicide + Max convert
        """
        from_pos, to_pos = move
        tactical_score = 0.5  # Base score
        
        # 1. Corner avoidance (tránh góc)
        if self._is_corner_position(to_pos):
            tactical_score -= 0.4
        elif self._is_edge_position(to_pos):
            tactical_score -= 0.1
            
        # 2. Suicide move detection
        if self._is_suicide_move(state, move):
            tactical_score -= 0.5
            
        # 3. Capture maximization
        captures = self._count_captures(state, move)
        tactical_score += captures * 0.15
        
        # 4. Center control bonus
        if self._is_center_position(to_pos):
            tactical_score += 0.2
            
        # 5. Mobility preservation
        mobility_preserved = self._check_mobility_preservation(state, move)
        tactical_score += mobility_preserved * 0.1

        return self._sigmoid_normalize(tactical_score)

    def _calculate_strategic_score(self, state, move):
        """
        Strategic Score: Tempo control + Enemy isolation + Mobility reduction
        """
        strategic_score = 0.0
        
        # 1. Tempo control (0-0.3)
        tempo_value = self._evaluate_tempo_control(state, move)
        strategic_score += tempo_value * 0.3
        
        # 2. Enemy isolation (0-0.4)
        isolation_value = self._evaluate_enemy_isolation(state, move)
        strategic_score += isolation_value * 0.4
        
        # 3. Mobility reduction (0-0.3)
        mobility_reduction = self._evaluate_mobility_reduction(state, move)
        strategic_score += mobility_reduction * 0.3

        return self._sigmoid_normalize(strategic_score)
    
    def _execute_tournament_round(self, state, candidates, weights, round_name="Round", sim_ratio=1.5, is_first_round=False):
        """
        Unified function for executing tournament rounds with combined scoring and parallel processing
        
        Args:
            state: Current game state
            candidates: List of candidate moves with their data
            weights: Phase-adaptive component weights (alpha, beta, gamma, delta)
            round_name: Name of the round for logging
            sim_ratio: Simulation ratio multiplier
            is_first_round: True if this is Round 1 (preliminary evaluation)
            
        Returns:
            List of (move, combined_score, simulation_score) tuples
        """
        num_candidates = len(candidates)
        
        # Decide parallel vs sequential processing
        use_parallel = (num_candidates >= TOURNAMENT_CONFIG['PARALLEL_THRESHOLD'] and self.use_parallel)
        
        if use_parallel:
            return self._parallel_tournament_round(state, candidates, weights, round_name, sim_ratio, is_first_round)
        else:
            return self._sequential_tournament_round(state, candidates, weights, round_name, sim_ratio, is_first_round)
    
    def _parallel_tournament_round(self, state, candidates, weights, round_name, sim_ratio, is_first_round=False):
        """Parallel processing for tournament rounds with combined scoring"""
        print(f"  🔄 Using PARALLEL processing for {round_name}")
        self.performance_stats['parallel_evaluations'] += 1
        
        # Prepare state data for worker processes
        state_data = {
            'player1_board': state.player1_board,
            'player2_board': state.player2_board,
            'balls': state.balls,
            'turn_player': state.turn_player
        }
        
        # Calculate simulation count for this round
        num_sims = int(self.basic_simulations * sim_ratio)
        
        # Prepare arguments - handle different candidate formats
        if is_first_round:  # Round 1: candidates are (move, 0.0, 0.0, 0.0, 0.0, 0.0) 
            args_list = [(state_data, move, num_sims, state.turn_player) 
                        for move, _, _, _, _, _ in candidates]
        elif len(candidates[0]) == 6:  # Round 2 format: (move, prev_score, h, t, s, prev_sim)
            args_list = [(state_data, move, num_sims, state.turn_player) 
                        for move, _, _, _, _, _ in candidates]
        else:  # Round 3 format: (move, combined_score, sim_score)
            args_list = [(state_data, move, num_sims, state.turn_player) 
                        for move, _, _ in candidates]
        
        results = []
        
        try:
            with ThreadPoolExecutor(max_workers=TOURNAMENT_CONFIG['MAX_WORKERS']) as executor:
                # Submit all simulation tasks
                future_to_move = {
                    executor.submit(_parallel_simulation_worker, args): candidates[i]
                    for i, args in enumerate(args_list)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_move):
                    candidate_data = future_to_move[future]
                    
                    try:
                        new_sim_score = future.result()
                        
                        # Extract move and previous simulation score
                        if is_first_round:  # Round 1 format
                            move, _, _, _, _, prev_sim = candidate_data
                        elif len(candidate_data) == 6:  # Round 2 format
                            move, prev_combined_score, prev_h, prev_t, prev_s, prev_sim = candidate_data
                        else:  # Round 3 format
                            move, prev_combined_score, prev_sim = candidate_data
                        
                        # Recalculate all components for fresh scoring
                        h = self._calculate_heuristic_score(state, move)
                        t = self._calculate_tactical_score(state, move)
                        s = self._calculate_strategic_score(state, move)
                        
                        # Handle simulation score combination
                        if is_first_round:
                            # For Round 1, just use the new simulation score
                            combined_sim = new_sim_score
                        else:
                            # Combine simulation scores (average of previous and new)
                            combined_sim = (prev_sim + new_sim_score) / 2
                        
                        # Calculate final combined score with phase weights
                        combined_score = (
                            weights['alpha'] * h +
                            weights['beta'] * t +
                            weights['gamma'] * s +
                            weights['delta'] * combined_sim
                        )
                        
                        results.append((move, combined_score, combined_sim))
                        print(f"    ✓ {move}: H={h:.2f} T={t:.2f} S={s:.2f} Sim={combined_sim:.3f} → {combined_score:.3f}")
                        
                    except Exception as e:
                        print(f"    ⚠ {candidate_data[0]}: parallel error, using fallback")
                        # Fallback to previous score
                        if is_first_round:
                            # For Round 1, use basic heuristic as fallback
                            move = candidate_data[0]
                            h = self._calculate_heuristic_score(state, move)
                            results.append((move, h, 0.0))
                        elif len(candidate_data) == 6:
                            results.append((candidate_data[0], candidate_data[1], candidate_data[5]))
                        else:
                            results.append((candidate_data[0], candidate_data[1], candidate_data[2]))
                        
        except Exception as e:
            print(f"  ⚠️ Parallel processing failed, falling back to sequential")
            return self._sequential_tournament_round(state, candidates, weights, round_name, sim_ratio, is_first_round)
        
        return results
    
    def _sequential_tournament_round(self, state, candidates, weights, round_name, sim_ratio, is_first_round=False):
        """Sequential processing for tournament rounds with combined scoring"""
        print(f"  🔄 Using SEQUENTIAL processing for {round_name}")
        
        # Calculate simulation count for this round
        num_sims = int(self.basic_simulations * sim_ratio)
        results = []
        
        for candidate_data in candidates:
            # Extract move and previous simulation score
            if is_first_round:  # Round 1 format
                move, _, _, _, _, prev_sim = candidate_data
            elif len(candidate_data) == 6:  # Round 2 format
                move, prev_combined_score, prev_h, prev_t, prev_s, prev_sim = candidate_data
            else:  # Round 3 format
                move, prev_combined_score, prev_sim = candidate_data
            
            # Run intensive simulation for this round
            new_sim_score = _rollout_simulation(state, move, num_sims)

            # Recalculate all components for fresh scoring
            h = self._calculate_heuristic_score(state, move)
            t = self._calculate_tactical_score(state, move)
            s = self._calculate_strategic_score(state, move)
            
            # Handle simulation score combination
            if is_first_round:
                # For Round 1, just use the new simulation score
                combined_sim = new_sim_score
            else:
                # Combine simulation scores (average of previous and new)
                combined_sim = (prev_sim + new_sim_score) / 2
            
            # Calculate final combined score with phase weights
            combined_score = (
                weights['alpha'] * h +
                weights['beta'] * t +
                weights['gamma'] * s +
                weights['delta'] * combined_sim
            )
            
            results.append((move, combined_score, combined_sim))
            print(f"    ✓ {move}: H={h:.2f} T={t:.2f} S={s:.2f} Sim={combined_sim:.3f} → {combined_score:.3f}")
            
        return results
    
    # ========================= HELPER METHODS =========================
    
    def _detect_game_phase(self, state):
        """Detect game phase based on piece count"""
        total_pieces = bin(state.player1_board | state.player2_board).count('1')
        
        if total_pieces < 8:
            return 'early'
        elif total_pieces < 20:
            return 'mid'
        else:
            return 'late'
    
    def _sigmoid_normalize(self, x):
        """Sigmoid normalization: σ(x) = 1/(1 + e^(-x/factor))"""
        return 1 / (1 + math.exp(-x / TOURNAMENT_CONFIG['SIGMOID_FACTOR']))
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance"""
        if pos1 is None or pos2 is None:
            return 0  # Return safe default
            
        # Handle both tuple (row, col) and integer formats
        if isinstance(pos1, tuple):
            row1, col1 = pos1
        else:
            row1, col1 = divmod(pos1, 7)
            
        if isinstance(pos2, tuple):
            row2, col2 = pos2
        else:
            row2, col2 = divmod(pos2, 7)
            
        return abs(row1 - row2) + abs(col1 - col2)
    
    def _count_enemy_pieces(self, state, player):
        """Count enemy pieces"""
        if player == PLAYER_ONE:
            return bin(state.player2_board).count('1')
        else:
            return bin(state.player1_board).count('1')
        
    def _count_friendly_neighbors(self, state, position, player):
        """
        Đếm số quân đồng minh xung quanh ô đích
        
        Args:
            state: Current game state
            position: Target position (row, col) hoặc int
            player: Current player (PLAYER_ONE hoặc PLAYER_TWO)
        
        Returns:
            Number of friendly pieces (0-8) around the target position
        """
        if position is None:
            return 0
        
        # Handle both tuple (row, col) and integer formats
        if isinstance(position, tuple):
            row, col = position
        else:
            row, col = divmod(position, 7)
        
        friendly_count = 0
        
        # Check all 8 adjacent positions (including diagonals)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip the center position itself
                    
                new_row, new_col = row + dr, col + dc
                
                # Check if position is within board bounds
                if 0 <= new_row < 7 and 0 <= new_col < 7:
                    new_bit = new_row * 7 + new_col
                    
                    # Check if there's a friendly piece at this position
                    if player == PLAYER_ONE:
                        if (state.player1_board >> new_bit) & 1:
                            friendly_count += 1
                    else:  # PLAYER_TWO
                        if (state.player2_board >> new_bit) & 1:
                            friendly_count += 1
        
        return friendly_count
    
    # def _calculate_blocking_value(self, state, position):
    #     """Calculate blocking value for position"""
    #     # Simplified: check if position blocks enemy expansion paths
    #     blocking_value = 0.0
        
    #     # Handle both tuple (row, col) and integer formats
    #     if isinstance(position, tuple):
    #         row, col = position
    #     else:
    #         row, col = divmod(position, 7)
        
    #     # Check adjacent positions for enemy pieces
    #     for dr, dc in ADJACENT_POSITIONS:
    #         new_row, new_col = row + dr, col + dc
    #         if 0 <= new_row < 7 and 0 <= new_col < 7:
    #             new_pos = new_row * 7 + new_col
    #             if state.turn_player == PLAYER_ONE:
    #                 if (state.player2_board >> new_pos) & 1:
    #                     blocking_value += 0.2
    #             else:
    #                 if (state.player1_board >> new_pos) & 1:
    #                     blocking_value += 0.2
        
    #     return min(blocking_value, 1.0)
    
    # def _calculate_position_penalty(self, position):
    #     """Calculate position penalty"""
    #     penalty = 0.0
        
    #     if self._is_corner_position(position):
    #         penalty += 0.8
    #     elif self._is_edge_position(position):
    #         penalty += 0.3
            
    #     return penalty

    def _is_corner_position(self, position):
        """Check if position is corner"""
        if position is None:
            return False
            
        # Handle both tuple (row, col) and integer formats
        if isinstance(position, tuple):
            row, col = position
            # Check corners for tuple format
            return (row == 0 or row == 6) and (col == 0 or col == 6)
        else:
            # Check corners for integer format
            corners = [0, 6, 42, 48]  # 7x7 board corners
            return position in corners
    
    def _is_edge_position(self, position):
        """Check if position is edge"""
        if position is None:
            return False
            
        # Handle both tuple (row, col) and integer formats
        if isinstance(position, tuple):
            row, col = position
        else:
            row, col = divmod(position, 7)
        
        return row == 0 or row == 6 or col == 0 or col == 6
    
    def _is_center_position(self, position):
        """Check if position is center"""
        if position is None:
            return False
            
        # Handle both tuple (row, col) and integer formats
        if isinstance(position, tuple):
            row, col = position
        else:
            row, col = divmod(position, 7)
        
        return 2 <= row <= 4 and 2 <= col <= 4
    
    def _is_suicide_move(self, state, move):
        """Check if move isolates our piece"""
        # Simplified suicide detection
        from_pos, to_pos = move
        
        # Check if destination will be isolated
        temp_state = deepcopy(state)
        temp_state.move_with_position(move)
        
        # Count friendly pieces around destination
        # Handle both tuple (row, col) and integer formats
        if isinstance(to_pos, tuple):
            row, col = to_pos
        else:
            row, col = divmod(to_pos, 7)
        
        friendly_neighbors = 0
        
        for dr, dc in ADJACENT_POSITIONS:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 7 and 0 <= new_col < 7:
                new_pos = new_row * 7 + new_col
                if state.turn_player == PLAYER_ONE:
                    if (temp_state.player1_board >> new_pos) & 1:
                        friendly_neighbors += 1
                else:
                    if (temp_state.player2_board >> new_pos) & 1:
                        friendly_neighbors += 1
        
        return friendly_neighbors == 0
    
    def _count_captures(self, state, move):
        """Count captured pieces from move"""
        temp_state = deepcopy(state)
        original_enemy = self._count_enemy_pieces(temp_state, state.turn_player)
        temp_state.move_with_position(move)
        new_enemy = self._count_enemy_pieces(temp_state, state.turn_player)
        return original_enemy - new_enemy
    
    def _check_mobility_preservation(self, state, move):
        """Check if move preserves our mobility"""
        try:
            # Count our moves before the move
            our_moves_before = len(state.get_all_possible_moves())
            
            # Simulate the move
            temp_state = deepcopy(state)
            temp_state.move_with_position(move)
            temp_state.toggle_player()  # Switch to enemy
            temp_state.toggle_player()  # Switch back to us
            
            # Count our moves after the move
            our_moves_after = len(temp_state.get_all_possible_moves())
            
            if our_moves_before == 0:
                return 0.0  # No mobility to preserve
            
            # Calculate mobility preservation ratio
            mobility_ratio = our_moves_after / our_moves_before
            
            # Bonus for maintaining or increasing mobility
            if mobility_ratio >= 1.0:
                return 1.0  # Perfect preservation or improvement
            else:
                return mobility_ratio  # Proportional to preserved mobility
                
        except Exception:
            return 0.5  # Fallback on error
    
    def _evaluate_tempo_control(self, state, move):
        """Evaluate tempo control value"""
        try:
            from_pos, to_pos = move
            tempo_score = 0.0
            
            # 1. Center control bonus (controlling central squares)
            if self._is_center_position(to_pos):
                tempo_score += 0.4
            
            # 2. Multiple threat creation
            temp_state = deepcopy(state)
            temp_state.move_with_position(move)
            
            # Count how many enemy pieces we can potentially capture next turn
            threat_count = 0
            if isinstance(to_pos, tuple):
                row, col = to_pos
            else:
                row, col = divmod(to_pos, 7)
            
            for dr, dc in ADJACENT_POSITIONS:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 7 and 0 <= new_col < 7:
                    new_bit = new_row * 7 + new_col
                    if state.turn_player == PLAYER_ONE:
                        if (state.player2_board >> new_bit) & 1:
                            threat_count += 1
                    else:
                        if (state.player1_board >> new_bit) & 1:
                            threat_count += 1
            
            tempo_score += min(threat_count * 0.15, 0.3)  # Cap at 0.3
            
            # 3. Initiative bonus for aggressive moves
            captures = self._count_captures(state, move)
            if captures > 0:
                tempo_score += 0.2  # Bonus for taking initiative
            
            # 4. Clone move tempo bonus (fast expansion)
            if from_pos is None:  # Clone move
                tempo_score += 0.1
            
            return min(tempo_score, 1.0)
            
        except Exception:
            return 0.5  # Fallback on error
    
    def _evaluate_enemy_isolation(self, state, move):
        """Evaluate enemy isolation potential using connectivity analysis"""
        try:
            from_pos, to_pos = move
            
            # Simulate the move
            temp_state = deepcopy(state)
            temp_state.move_with_position(move)
            
            # Count connected enemy groups before and after our move
            enemy_player = -state.turn_player
            
            # Get enemy positions
            enemy_positions = []
            for i in range(49):
                row, col = divmod(i, 7)
                if enemy_player == PLAYER_ONE:
                    if (temp_state.player1_board >> i) & 1:
                        enemy_positions.append((row, col))
                else:
                    if (temp_state.player2_board >> i) & 1:
                        enemy_positions.append((row, col))
            
            if not enemy_positions:
                return 1.0  # Perfect isolation (no enemies left)
            
            # Count connected components using flood-fill
            visited = set()
            components = 0
            
            def flood_fill(start_pos):
                """Flood fill to find connected enemy pieces"""
                stack = [start_pos]
                component_size = 0
                
                while stack:
                    pos = stack.pop()
                    if pos in visited:
                        continue
                    
                    visited.add(pos)
                    component_size += 1
                    
                    # Check adjacent positions
                    row, col = pos
                    for dr, dc in ADJACENT_POSITIONS:
                        new_row, new_col = row + dr, col + dc
                        if (0 <= new_row < 7 and 0 <= new_col < 7 and 
                            (new_row, new_col) in enemy_positions and 
                            (new_row, new_col) not in visited):
                            stack.append((new_row, new_col))
                
                return component_size
            
            # Count components and their sizes
            component_sizes = []
            for pos in enemy_positions:
                if pos not in visited:
                    size = flood_fill(pos)
                    component_sizes.append(size)
                    components += 1
            
            # Calculate isolation score
            if components == 0:
                return 1.0
            elif components == 1:
                # Single large group - low isolation
                return 0.2
            else:
                # Multiple small groups - good isolation
                # More components and smaller average size = better isolation
                avg_component_size = sum(component_sizes) / len(component_sizes)
                isolation_score = min(components * 0.2, 0.8)  # More components = better
                isolation_score += max(0, (3 - avg_component_size) * 0.1)  # Smaller groups = better
                
                return min(isolation_score, 1.0)
            
        except Exception:
            return 0.5  # Fallback on error
    
    def _evaluate_mobility_reduction(self, state, move):
        """Evaluate enemy mobility reduction"""
        # Count enemy moves before/after
        temp_state = deepcopy(state)
        temp_state.toggle_player()
        enemy_moves_before = len(temp_state.get_all_possible_moves())
        
        temp_state.toggle_player()
        temp_state.move_with_position(move)
        temp_state.toggle_player()
        enemy_moves_after = len(temp_state.get_all_possible_moves())
        
        if enemy_moves_before == 0:
            return 1.0
        return (enemy_moves_before - enemy_moves_after) / enemy_moves_before