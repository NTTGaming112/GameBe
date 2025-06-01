"""
MONTE CARLO DOMAIN - TOURNAMENT SYSTEM INTEGRATED
================================================

Hoàn toàn tích hợp tournament system với ba vòng thi đấu:
1. Vòng sơ loại: Đánh giá sơ bộ tất cả moves
2. Vòng mô phỏng: Simulation chi tiết cho top candidates
3. Vòng chung kết: Head-to-head với parallel processing

Hệ thống điểm số tổng hợp:
- Heuristic Score (H): s1×C + s2×A + s3×B - s4×P
- Tactical Score (T): Corner avoidance + No suicide + Max convert  
- Strategic Score (S): Tempo + Enemy isolation + Mobility reduction
- Simulation Score: Monte Carlo rollouts

Phase-adaptive weights: α×H + β×T + γ×S + δ×Sim
"""

import random
import math
import time
from copy import deepcopy
from collections import OrderedDict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from .monte_carlo_base import MonteCarloBase
from app.ai.ataxx_state import Ataxx, PLAYER_ONE, PLAYER_TWO
from .constants import (
    TOURNAMENT_CONFIG, 
    HEURISTIC_COEFFS, PHASE_BONUS_PENALTY, MCD_PHASE_WEIGHTS,
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
        
        # Add evaluation cache
        self.eval_cache = {}
        
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
        
    def get_probabilistic_move(self, temperature=1.0, use_tournament=True):
        """
        Get move using probabilistic selection with softmax and temperature
        
        Args:
            temperature: Temperature for softmax (higher = more random)
            use_tournament: Whether to use tournament system for scoring
        
        Returns:
            Selected move based on probability distribution
        """
        state = self.root_state
        
        if state is None:
            print("Error: root_state is None")
            return None
        
        moves = state.get_all_possible_moves()
        if not moves:
            return None
            
        if len(moves) == 1:
            return moves[0]
        
        # Get move scores
        if use_tournament:
            # Use simplified tournament scoring
            scores = self._get_move_scores_tournament(state, moves)
        else:
            # Use basic heuristic scoring
            scores = self._get_move_scores_basic(state, moves)
        
        # Apply softmax with temperature
        probabilities = softmax_with_temperature(scores, temperature)
        
        # Sample move based on probabilities
        selected_move = sample_from_probabilities(moves, probabilities)
        
        print(f"🎯 Probabilistic selection (T={temperature:.2f}):")
        for i, (move, score, prob) in enumerate(zip(moves, scores, probabilities)):
            marker = "👉" if move == selected_move else "  "
            print(f"{marker} {move}: score={score:.3f}, prob={prob:.3f}")
        
        return selected_move
    
    def _get_move_scores_tournament(self, state, moves):
        """Get move scores using simplified tournament system"""
        phase = self._detect_game_phase(state)
        weights = MCD_PHASE_WEIGHTS[phase]
        
        scores = []
        for move in moves:
            # Calculate all score components
            heuristic_score = self._calculate_heuristic_score(state, move)
            tactical_score = self._calculate_tactical_score(state, move)
            strategic_score = self._calculate_strategic_score(state, move)
            simulation_score = self._run_preliminary_simulation(state, move)
            
            # Combined score với phase weights
            combined_score = (
                weights['alpha'] * heuristic_score +
                weights['beta'] * tactical_score +
                weights['gamma'] * strategic_score +
                weights['delta'] * simulation_score
            )
            
            scores.append(combined_score)
        
        return scores
    
    def _get_move_scores_basic(self, state, moves):
        """Get move scores using basic heuristic evaluation"""
        scores = []
        for move in moves:
            heuristic_score = self._calculate_heuristic_score(state, move)
            tactical_score = self._calculate_tactical_score(state, move)
            combined_score = (heuristic_score + tactical_score) / 2
            scores.append(combined_score)
        
        return scores
    
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
        
        round1_results = []
        for i, move in enumerate(moves):
            # Calculate all score components
            heuristic_score = self._calculate_heuristic_score(state, move)
            tactical_score = self._calculate_tactical_score(state, move)
            strategic_score = self._calculate_strategic_score(state, move)
            simulation_score = self._run_preliminary_simulation(state, move)
            
            # Combined score với phase weights
            combined_score = (
                weights['alpha'] * heuristic_score +
                weights['beta'] * tactical_score +
                weights['gamma'] * strategic_score +
                weights['delta'] * simulation_score
            )
            
            round1_results.append((move, combined_score, heuristic_score, 
                                 tactical_score, strategic_score, simulation_score))
            
            print(f"  Move {i+1:2d} {move}: H={heuristic_score:.2f} T={tactical_score:.2f} "
                  f"S={strategic_score:.2f} Sim={simulation_score:.2f} → {combined_score:.3f}")
        
        # Sort và advance top K1
        round1_results.sort(key=lambda x: x[1], reverse=True)
        round2_candidates = round1_results[:k1]
        
        print(f"\n✅ ROUND 1 COMPLETE - Top {k1} moves advance:")
        for i, (move, score, h, t, s, sim) in enumerate(round2_candidates):
            print(f"  {i+1}. {move}: {score:.3f}")
            
        if k1 == 1:
            winner = round2_candidates[0][0]
            print(f"\n🥇 SINGLE WINNER: {winner}")
            return winner
        
        # =================== ROUND 2: SIMULATION ===================
        print(f"\n🎯 ROUND 2: Intensive Simulation ({k1} moves, {round2_sims} sims each)")
        
        round2_results = []
        for move, prev_score, h, t, s, prev_sim in round2_candidates:
            # More intensive simulation
            intensive_sim = self._run_intensive_simulation(state, move)
            
            # Combine simulation scores
            combined_sim = (prev_sim + intensive_sim) / 2
            
            # Recalculate final score
            final_score = (
                weights['alpha'] * h +
                weights['beta'] * t +
                weights['gamma'] * s +
                weights['delta'] * combined_sim
            )
            
            round2_results.append((move, final_score, combined_sim))
            print(f"  {move}: intensive_sim={intensive_sim:.3f} → final={final_score:.3f}")
        
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
        
        # Decide parallel vs sequential
        use_parallel = (len(round3_candidates) >= TOURNAMENT_CONFIG['PARALLEL_THRESHOLD'] 
                       and self.use_parallel)
        
        if use_parallel:
            final_scores = self._parallel_final_round(state, round3_candidates, weights)
        else:
            final_scores = self._sequential_final_round(state, round3_candidates, weights)
        
        # Determine tournament winner
        winner_move = max(final_scores, key=lambda x: x[1])[0]
        winner_score = max(final_scores, key=lambda x: x[1])[1]
        
        print(f"\n🥇 TOURNAMENT WINNER: {winner_move} (score: {winner_score:.3f})")
        return winner_move
    
    def _calculate_heuristic_score(self, state, move):
        """
        Heuristic Score: H = s1×C + s2×A + s3×B - s4×P
        
        C = Clone moves (1-step moves)
        A = Attack moves (capture enemy pieces)
        B = Clone bonus (reward clone/adjacent moves)
        P = Jump penalty (penalize long jumps)
        """
        # Debug: kiểm tra cấu trúc move
        if move is None:
            return 0.0
        
        # Kiểm tra xem move có phải là tuple không
        if not isinstance(move, (tuple, list)) or len(move) != 2:
            print(f"Warning: Invalid move format: {move}, type: {type(move)}")
            return 0.0
            
        from_pos, to_pos = move
        
        # Kiểm tra positions có hợp lệ không
        if to_pos is None:
            print(f"Warning: to_pos is None in move: {move}")
            return 0.0
            
        # Special handling for clone moves (from_pos is None)
        if from_pos is None:
            # This is a clone move, distance is 0
            distance = 0
        else:
            distance = self._manhattan_distance(from_pos, to_pos)

        
        # C: Clone component
        # Special handling for clone moves (from_pos is None)
        if from_pos is None:
            # This is a clone move, distance is 0
            distance = 0
        else:
            distance = self._manhattan_distance(from_pos, to_pos)
        C = 1.0 if distance <= 1 else 0.0
        
        # A: Attack component (captured pieces)
        temp_state = deepcopy(state)
        original_enemy = self._count_enemy_pieces(temp_state, state.turn_player)
        temp_state.move_with_position(move)
        new_enemy = self._count_enemy_pieces(temp_state, state.turn_player)
        A = original_enemy - new_enemy
        
        # B: Clone bonus (reward clone moves)
        phase = self._detect_game_phase(state)
        B = self._calculate_clone_bonus(from_pos, to_pos, phase)
        
        # P: Jump penalty (penalize long jumps)
        P = self._calculate_jump_penalty(from_pos, to_pos, phase)
        
        # Apply heuristic formula
        h_raw = (HEURISTIC_COEFFS['s1'] * C + 
                HEURISTIC_COEFFS['s2'] * A + 
                HEURISTIC_COEFFS['s3'] * B - 
                HEURISTIC_COEFFS['s4'] * P)
        
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
        
        return max(0.0, min(1.0, tactical_score))
    
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
        
        return max(0.0, min(1.0, strategic_score))
    
    def _run_preliminary_simulation(self, state, move):
        """Quick simulation cho Round 1"""
        num_sims = int(self.basic_simulations * TOURNAMENT_CONFIG['ROUND1_SIM_RATIO'])
        total_score = 0.0
        original_player = state.turn_player
        
        for _ in range(num_sims):
            temp_state = deepcopy(state)
            temp_state.move_with_position(move)
            temp_state.toggle_player()
            
            score = _rollout_simulation(temp_state, original_player)
            total_score += score
            
        return total_score / num_sims
    
    def _run_intensive_simulation(self, state, move):
        """Intensive simulation cho Round 2"""
        num_sims = int(self.basic_simulations * TOURNAMENT_CONFIG['ROUND2_SIM_RATIO'])
        total_score = 0.0
        original_player = state.turn_player
        
        for _ in range(num_sims):
            temp_state = deepcopy(state)
            temp_state.move_with_position(move)
            temp_state.toggle_player()
            
            # Longer, smarter rollout
            score = self._smart_rollout(temp_state, original_player)
            total_score += score
            
        return total_score / num_sims
    
    def _parallel_final_round(self, state, candidates, weights):
        """Parallel processing cho Round 3 with combined scoring"""
        print("  🔄 Using PARALLEL processing for final round")
        self.performance_stats['parallel_evaluations'] += 1
        
        # Prepare state data
        state_data = {
            'player1_board': state.player1_board,
            'player2_board': state.player2_board,
            'balls': state.balls,
            'turn_player': state.turn_player
        }
        
        # Prepare arguments
        num_sims = int(self.basic_simulations * TOURNAMENT_CONFIG['ROUND3_SIM_RATIO'])
        args_list = [(state_data, move, num_sims, state.turn_player) 
                    for move, _, _ in candidates]
        
        final_scores = []
        
        try:
            with ThreadPoolExecutor(max_workers=TOURNAMENT_CONFIG['MAX_WORKERS']) as executor:
                future_to_move = {
                    executor.submit(_parallel_simulation_worker, args): (candidates[i][0], candidates[i][2])
                    for i, args in enumerate(args_list)
                }
                
                for future in as_completed(future_to_move):
                    move, prev_sim_score = future_to_move[future]
                    try:
                        final_sim_score = future.result()
                        
                        # Recalculate all components for final scoring
                        h = self._calculate_heuristic_score(state, move)
                        t = self._calculate_tactical_score(state, move)
                        s = self._calculate_strategic_score(state, move)
                        
                        # Average previous and final simulation scores
                        combined_sim = (prev_sim_score + final_sim_score) / 2
                        
                        # Calculate final combined score
                        final_combined_score = (
                            weights['alpha'] * h +
                            weights['beta'] * t +
                            weights['gamma'] * s +
                            weights['delta'] * combined_sim
                        )
                        
                        final_scores.append((move, final_combined_score))
                        print(f"    ✓ {move}: H={h:.2f} T={t:.2f} S={s:.2f} Sim={combined_sim:.3f} → {final_combined_score:.3f}")
                    except Exception as e:
                        print(f"    ⚠ {move}: parallel error, using fallback")
                        final_scores.append((move, 0.5))
                        
        except Exception as e:
            print(f"  ⚠️ Parallel processing failed, falling back to sequential")
            return self._sequential_final_round(state, candidates, weights)
        
        return final_scores
    
    def _sequential_final_round(self, state, candidates, weights):
        """Sequential processing cho Round 3 with combined scoring"""
        print("  🔄 Using SEQUENTIAL processing for final round")
        final_scores = []
        
        for move, prev_combined_score, prev_sim_score in candidates:
            # Get final intensive simulation
            final_sim_score = self._run_final_simulation(state, move)
            
            # Recalculate all components for final scoring
            h = self._calculate_heuristic_score(state, move)
            t = self._calculate_tactical_score(state, move) 
            s = self._calculate_strategic_score(state, move)
            
            # Average previous and final simulation scores
            combined_sim = (prev_sim_score + final_sim_score) / 2
            
            # Calculate final combined score
            final_combined_score = (
                weights['alpha'] * h +
                weights['beta'] * t +
                weights['gamma'] * s +
                weights['delta'] * combined_sim
            )
            
            final_scores.append((move, final_combined_score))
            print(f"    ✓ {move}: H={h:.2f} T={t:.2f} S={s:.2f} Sim={combined_sim:.3f} → {final_combined_score:.3f}")
            
        return final_scores
    
    def _run_final_simulation(self, state, move):
        """Final intensive simulation"""
        num_sims = int(self.basic_simulations * TOURNAMENT_CONFIG['ROUND3_SIM_RATIO'])
        total_score = 0.0
        original_player = state.turn_player
        
        for _ in range(num_sims):
            temp_state = deepcopy(state)
            temp_state.move_with_position(move)
            temp_state.toggle_player()
            
            score = self._smart_rollout(temp_state, original_player)
            total_score += score
            
        return total_score / num_sims
    
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
    
    def _calculate_blocking_value(self, state, position):
        """Calculate blocking value for position"""
        # Simplified: check if position blocks enemy expansion paths
        blocking_value = 0.0
        
        # Handle both tuple (row, col) and integer formats
        if isinstance(position, tuple):
            row, col = position
        else:
            row, col = divmod(position, 7)
        
        # Check adjacent positions for enemy pieces
        for dr, dc in ADJACENT_POSITIONS:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 7 and 0 <= new_col < 7:
                new_pos = new_row * 7 + new_col
                if state.turn_player == PLAYER_ONE:
                    if (state.player2_board >> new_pos) & 1:
                        blocking_value += 0.2
                else:
                    if (state.player1_board >> new_pos) & 1:
                        blocking_value += 0.2
        
        return min(blocking_value, 1.0)
    
    def _calculate_position_penalty(self, position):
        """Calculate position penalty"""
        penalty = 0.0
        
        if self._is_corner_position(position):
            penalty += 0.8
        elif self._is_edge_position(position):
            penalty += 0.3
            
        return penalty
    
    def _calculate_clone_bonus(self, from_pos, to_pos, phase):
        """Calculate phase-adaptive clone bonus"""
        phase_config = PHASE_BONUS_PENALTY[phase]
        
        # Clone moves (from_pos is None) get full bonus
        if from_pos is None:
            return phase_config['clone_bonus']
        
        # Adjacent moves (distance 1) get partial bonus
        distance = self._manhattan_distance(from_pos, to_pos)
        if distance <= 1:
            return phase_config['clone_bonus'] * 0.7
        
        # Jump moves get no clone bonus
        return 0.0
    
    def _calculate_jump_penalty(self, from_pos, to_pos, phase):
        """Calculate phase-adaptive jump penalty"""
        phase_config = PHASE_BONUS_PENALTY[phase]
        
        # Clone moves (from_pos is None) have no jump penalty
        if from_pos is None:
            return 0.0
        
        # Calculate distance and apply penalty
        distance = self._manhattan_distance(from_pos, to_pos)
        if distance <= 1:
            return 0.0  # No penalty for adjacent moves
        else:
            # Jump moves get penalty based on distance and phase
            penalty_factor = min(distance / 2.0, 1.0)  # Scale by distance
            return phase_config['jump_penalty'] * penalty_factor

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
    
    def _smart_rollout(self, state, original_player):
        """Smarter rollout với heuristic guidance and enhanced evaluation"""
        simulation_state = deepcopy(state)
        depth = 0
        max_depth = 25
        
        while depth < max_depth and not simulation_state.is_game_over():
            moves = simulation_state.get_all_possible_moves()
            if not moves:
                simulation_state.toggle_player()
                moves = simulation_state.get_all_possible_moves()
                if not moves:
                    break
            
            # Use heuristic guidance for move selection
            if len(moves) > 3:
                move_scores = []
                for move in moves:
                    score = self._quick_move_heuristic(simulation_state, move)
                    move_scores.append((move, score))
                
                move_scores.sort(key=lambda x: x[1], reverse=True)
                # Select từ top 3 moves
                top_moves = [move for move, _ in move_scores[:3]]
                selected_move = random.choice(top_moves)
            else:
                selected_move = random.choice(moves)
            
            simulation_state.move_with_position(selected_move)
            simulation_state.toggle_player()
            depth += 1
        
        # Use enhanced final position evaluation
        return self._evaluate_final_position(simulation_state, original_player)
    
    def _quick_move_heuristic(self, state, move):
        """Quick heuristic cho smart rollout"""
        from_pos, to_pos = move
        score = 0.5
        
        # Prefer center
        if self._is_center_position(to_pos):
            score += 0.3
        
        # Prefer captures
        captures = self._count_captures(state, move)
        score += captures * 0.2
        
        # Avoid corners
        if self._is_corner_position(to_pos):
            score -= 0.4
        
        return score
    
    def _get_cached_eval(self, state, player):
        """Get cached evaluation if exists"""
        state_key = (state.player1_board, state.player2_board, player)
        return self.eval_cache.get(state_key)
    
    def _cache_eval(self, state, player, score):
        """Cache evaluation result"""
        state_key = (state.player1_board, state.player2_board, player)
        self.eval_cache[state_key] = score
        
    def _evaluate_final_position(self, state, player):
        """Enhanced final position evaluation with caching"""
        cached_score = self._get_cached_eval(state, player)
        if cached_score is not None:
            return cached_score
        
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
        score = (score + 549) / 1098
        
        self._cache_eval(state, player, score)
        return score
    
    def analyze_temperature_effects(self, temperatures=[0.1, 0.5, 1.0, 2.0, 5.0]):
        """
        Analyze the effects of different temperature values on move selection
        
        Args:
            temperatures: List of temperature values to test
        """
        state = self.root_state
        if state is None:
            return
            
        moves = state.get_all_possible_moves()
        if not moves:
            return
            
        print(f"\n🌡️  Temperature Analysis for {len(moves)} moves:")
        
        # Get base scores
        scores = self._get_move_scores_basic(state, moves)
        
        print("\nMove Scores:")
        for i, (move, score) in enumerate(zip(moves, scores)):
            print(f"  {i+1}. {move}: {score:.3f}")
        
        print("\nProbability Distributions by Temperature:")
        for temp in temperatures:
            probabilities = softmax_with_temperature(scores, temp)
            print(f"\nTemperature = {temp}:")
            
            # Show probabilities
            for i, (move, prob) in enumerate(zip(moves, probabilities)):
                bar_length = int(prob * 20)  # Scale for visualization
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {move}: {prob:.3f} |{bar}|")
            
            # Calculate entropy (measure of randomness)
            entropy = -sum(p * math.log(p) if p > 0 else 0 for p in probabilities)
            max_entropy = math.log(len(moves))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            print(f"  Entropy: {entropy:.2f} (normalized: {normalized_entropy:.2f})")
            
            # Most likely move
            max_prob_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
            print(f"  Most likely: {moves[max_prob_idx]} ({probabilities[max_prob_idx]:.3f})")
    
    def get_top_moves_with_probabilities(self, top_k=5, temperature=1.0):
        """
        Get top K moves with their probabilities
        
        Args:
            top_k: Number of top moves to return
            temperature: Temperature for softmax
            
        Returns:
            List of (move, score, probability) tuples
        """
        state = self.root_state
        if state is None:
            return []
            
        moves = state.get_all_possible_moves()
        if not moves:
            return []
        
        # Get scores and probabilities
        scores = self._get_move_scores_basic(state, moves)
        probabilities = softmax_with_temperature(scores, temperature)
        
        # Combine and sort by probability
        move_data = list(zip(moves, scores, probabilities))
        move_data.sort(key=lambda x: x[2], reverse=True)  # Sort by probability
        
        return move_data[:top_k]
    
    def get_performance_stats(self):
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        stats['eval_cache_size'] = len(self.eval_cache)
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'tournaments_run': 0,
            'parallel_evaluations': 0,
            'avg_tournament_time': 0.0,
            'total_moves_evaluated': 0
        }
        # Clear evaluation cache
        self.eval_cache.clear()
        
    def clear_eval_cache(self):
        """Clear evaluation cache manually"""
        self.eval_cache.clear()
