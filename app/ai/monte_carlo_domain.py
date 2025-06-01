import random
import math
from copy import deepcopy
from collections import OrderedDict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time

from .monte_carlo_base import MonteCarloBase
from app.ai.ataxx_state import Ataxx, PLAYER_ONE, PLAYER_TWO
from app.ai.constants import (ADJACENT_POSITIONS, JUMP_POSITIONS, MOVE_WEIGHTS, 
                             PLAYER_ONE, PLAYER_TWO, PHASE_WEIGHTS, 
                             TEMPERATURE_SCHEDULE, COMPONENT_WEIGHTS)

def _evaluate_move_for_multiprocessing(state_data, move, num_simulations, original_player):
    """Standalone function for multiprocessing move evaluation."""
    # Reconstruct state from data
    state = Ataxx()
    state.player1_board = state_data['player1_board']
    state.player2_board = state_data['player2_board']
    state.balls = state_data['balls'].copy()
    state.turn_player = state_data['turn_player']
    
    total_score = 0.0
    
    for _ in range(num_simulations):
        # Create new state with the move applied
        new_state = deepcopy(state)
        new_state.move_with_position(move)
        new_state.toggle_player()
        
        # Run basic simulation
        simulation_score = _run_basic_simulation(new_state, original_player)
        total_score += simulation_score
    
    return total_score / num_simulations

def _run_basic_simulation(state, original_player):
    """Basic simulation for multiprocessing."""
    simulation_state = deepcopy(state)
    depth = 0
    max_depth = 15
    
    while depth < max_depth and not simulation_state.is_game_over():
        moves = simulation_state.get_all_possible_moves()
        if not moves:
            simulation_state.toggle_player()
            moves = simulation_state.get_all_possible_moves()
            if not moves:
                break
        
        # Random move selection
        move = random.choice(moves)
        simulation_state.move_with_position(move)
        simulation_state.toggle_player()
        depth += 1
    
    # Evaluate final state
    if simulation_state.is_game_over():
        winner = simulation_state.get_winner()
        if winner == original_player:
            return 1.0
        elif winner == -original_player:
            return 0.0
        else:
            return 0.5
    
    # Piece advantage evaluation
    my_pieces = simulation_state.balls.get(original_player, 0)
    opp_pieces = simulation_state.balls.get(-original_player, 0)
    
    if my_pieces + opp_pieces == 0:
        return 0.5
    
    return my_pieces / (my_pieces + opp_pieces)

def _run_simulation_batch_for_move(state_data, move, num_simulations, original_player):
    """Run a batch of simulations for a specific move (for multiprocessing)."""
    # Reconstruct state from data
    state = Ataxx()
    state.player1_board = state_data['player1_board']
    state.player2_board = state_data['player2_board']
    state.balls = state_data['balls'].copy()
    state.turn_player = state_data['turn_player']
    
    total_score = 0.0
    completed_simulations = 0
    
    for _ in range(num_simulations):
        try:
            # Apply the move to create new state
            new_state = deepcopy(state)
            new_state.move_with_position(move)
            new_state.toggle_player()
            
            # Run simulation
            score = _run_basic_simulation(new_state, original_player)
            total_score += score
            completed_simulations += 1
            
        except Exception as e:
            # Skip failed simulations but continue with others
            continue
    
    # Return average score and number of completed simulations
    if completed_simulations > 0:
        return total_score / completed_simulations, completed_simulations
    else:
        return 0.5, 0  # Neutral score if all simulations failed

class MonteCarloDomain(MonteCarloBase):
    def __init__(self, state, component_weights=None, **kwargs):
        super().__init__(state, **kwargs)
        self.phase_weights = PHASE_WEIGHTS
        self.temperature_schedule = TEMPERATURE_SCHEDULE
        self.component_weights = component_weights or COMPONENT_WEIGHTS
        self.eval_cache = OrderedDict()
        self.max_cache_size = 1000
        self.center_mask = sum(1 << (r * 7 + c) for r, c in [(3,3), (3,4), (4,3), (4,4)])
        self.near_center_mask = sum(1 << (r * 7 + c) for r, c in [
            (2,2), (2,3), (2,4), (2,5), (3,2), (3,5), (4,2), (4,5), (5,2), (5,3), (5,4), (5,5)])
        self.corners_mask = sum(1 << (r * 7 + c) for r, c in [(0,0), (0,6), (6,0), (6,6)])
        self.adjacent_masks = {}
        self.jump_masks = {}
        for x in range(7):
            for y in range(7):
                bit = x * 7 + y
                adj_mask = 0
                for dx, dy in ADJACENT_POSITIONS:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < 7 and 0 <= new_y < 7:
                        adj_mask |= 1 << (new_x * 7 + new_y)
                self.adjacent_masks[bit] = adj_mask
                jump_mask = 0
                for dx, dy in JUMP_POSITIONS:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < 7 and 0 <= new_y < 7:
                        jump_mask |= 1 << (new_x * 7 + new_y)
                self.jump_masks[bit] = jump_mask

    def _ultra_fast_move_selection(self, state, moves):
        """
        Optimized fast move selection with improved scoring and selection strategy.
        
        Improvements:
        1. More efficient move evaluation
        2. Better candidate filtering
        3. Adaptive evaluation depth based on move count
        4. Improved softmax selection with phase-appropriate temperature
        """
        if len(moves) <= 1:
            return moves[0] if moves else None
        
        my_player = state.current_player()
        phase = self._detect_game_phase(state)
        
        # Adaptive evaluation strategy based on number of moves
        if len(moves) <= 3:
            # Few moves: evaluate all thoroughly
            max_evaluate = len(moves)
            use_softmax = False
        elif len(moves) <= 8:
            # Moderate moves: evaluate top candidates
            max_evaluate = min(6, len(moves))
            use_softmax = True
        else:
            # Many moves: quick filtering then detailed evaluation
            max_evaluate = min(5, len(moves))
            use_softmax = True
        
        # Phase-appropriate quick filtering for large move sets
        if len(moves) > 10:
            moves = self._quick_filter_moves(state, moves, phase, max_candidates=12)
        
        # Evaluate top candidates with comprehensive scoring
        move_scores = {}
        best_score = -1.0
        best_move = moves[0]
        
        for i in range(min(max_evaluate, len(moves))):
            move = moves[i]
            score = self._calculate_comprehensive_move_score(state, move, phase)
            move_scores[move] = score
            
            if score > best_score:
                best_score = score
                best_move = move
        
        # Selection strategy based on phase and move count
        if use_softmax and len(move_scores) > 1:
            temperature = self._get_selection_temperature(phase, len(moves))
            selected_move = self._softmax_selection(move_scores, temperature)
            return selected_move if selected_move else best_move
        
        return best_move

    def _calculate_structured_move_score(self, state, move, phase):
        undo_info = state.apply_move_with_undo(move)
        next_state = Ataxx()
        next_state.player1_board = state.player1_board
        next_state.player2_board = state.player2_board
        next_state.balls = state.balls.copy()
        next_state.turn_player = state.turn_player
        next_state.moves = state.moves.copy()
        
        # Calculate component scores
        heuristic_score = self._calculate_heuristic_score(next_state, phase)
        strategic_score = self._calculate_strategic_score(state, move, next_state)
        tactical_score = self._calculate_tactical_score(state, move, next_state)
        
        state.undo_move(undo_info)
        
        # Normalize scores
        h_norm = self._sigmoid(heuristic_score)
        s_norm = self._sigmoid(strategic_score)
        t_norm = tactical_score  # Already normalized in tactical methods
        
        # Combine with weighted average (Tactical is very important for immediate safety)
        return (self.component_weights['heuristic'] * h_norm + 
                self.component_weights['strategic'] * s_norm +
                self.component_weights['tactical'] * t_norm) / 3.0

    def _calculate_heuristic_score(self, state, phase):
        cached_score = self._get_cached_eval(state, state.current_player())
        if cached_score is not None:
            return cached_score
        weights = self.phase_weights[phase]
        centrality = self._calculate_centrality_score(state)
        accessibility = self._calculate_accessibility_score(state)
        blocking = self._calculate_blocking_score(state)
        penalty = self._calculate_penalty_score(state)
        score = (weights['s1'] * centrality + 
                 weights['s2'] * accessibility + 
                 weights['s3'] * blocking - 
                 weights['s4'] * penalty)
        self._cache_eval(state, state.current_player(), score)
        return score

    def _calculate_centrality_score(self, state):
        my_board = state.player1_board if state.current_player() == PLAYER_ONE else state.player2_board
        opp_board = state.player2_board if state.current_player() == PLAYER_ONE else state.player1_board
        my_centrality = bin(my_board & self.center_mask).count('1') * 3 + \
                        bin(my_board & self.near_center_mask).count('1')
        opp_centrality = bin(opp_board & self.center_mask).count('1') * 3 + \
                         bin(opp_board & self.near_center_mask).count('1')
        return my_centrality - opp_centrality

    def _calculate_accessibility_score(self, state):
        return len(state.get_all_possible_moves())

    def _calculate_blocking_score(self, state):
        my_player = state.current_player()
        opp_player = -my_player
        opp_pieces = [(i // 7, i % 7) for i in range(49) if 
                      (state.player2_board if my_player == PLAYER_ONE else state.player1_board) & (1 << i)]
        blocking_value = 0
        for piece_pos in opp_pieces:
            bit = piece_pos[0] * 7 + piece_pos[1]
            available_moves = bin(self.adjacent_masks[bit] & ~(state.player1_board | state.player2_board)).count('1')
            available_moves += bin(self.jump_masks[bit] & ~(state.player1_board | state.player2_board)).count('1')
            if available_moves == 0:
                blocking_value += 5
            elif available_moves <= 2:
                blocking_value += 3
            elif available_moves <= 5:
                blocking_value += 1
        return blocking_value

    def _calculate_penalty_score(self, state):
        my_board = state.player1_board if state.current_player() == PLAYER_ONE else state.player2_board
        penalty = bin(my_board & self.corners_mask).count('1') * 2
        my_pieces = [(i // 7, i % 7) for i in range(49) if my_board & (1 << i)]
        for piece in my_pieces:
            bit = piece[0] * 7 + piece[1]
            if not (self.adjacent_masks[bit] & my_board):
                penalty += 1
        return penalty

    def _calculate_strategic_score(self, current_state, move, next_state):
        score = 0
        score += self._tempo_control_score(current_state, next_state)
        score += self._enemy_isolation_score(current_state, next_state)
        score += self._mobility_reduction_score(current_state, next_state)
        return score

    def _tempo_control_score(self, current_state, next_state):
        my_player = current_state.current_player()
        current_balance = current_state.balls[my_player] - current_state.balls[-my_player]
        next_balance = next_state.balls[my_player] - next_state.balls[-my_player]
        return next_balance - current_balance

    def _enemy_isolation_score(self, current_state, next_state):
        opp_player = -current_state.current_player()
        opp_pieces = [(i // 7, i % 7) for i in range(49) if 
                      (next_state.player2_board if opp_player == PLAYER_TWO else next_state.player1_board) & (1 << i)]
        if not opp_pieces:
            return 5
        visited = set()
        groups = 0
        for piece in opp_pieces:
            if piece not in visited:
                groups += 1
                queue = [piece]
                visited.add(piece)
                while queue:
                    current = queue.pop(0)
                    bit = current[0] * 7 + current[1]
                    for dx, dy in ADJACENT_POSITIONS + JUMP_POSITIONS:
                        new_x, new_y = current[0] + dx, current[1] + dy
                        neighbor = (new_x, new_y)
                        if 0 <= new_x < 7 and 0 <= new_y < 7 and neighbor in opp_pieces and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
        return groups * 1.5

    def _mobility_reduction_score(self, current_state, next_state):
        next_state_copy = Ataxx()
        next_state_copy.player1_board = next_state.player1_board
        next_state_copy.player2_board = next_state.player2_board
        next_state_copy.balls = next_state.balls.copy()
        next_state_copy.turn_player = -next_state.current_player()
        move_count = len(next_state_copy.get_all_possible_moves())
        if move_count == 0:
            return 10
        elif move_count <= 3:
            return 3
        elif move_count <= 8:
            return 1
        return 0

    def _sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0 if x < 0 else 1

    def _sigmoid_normalize(self, score, center=0.0, scale=1.0):
        """
        Sigmoid normalization to map scores to (0,1) range.
        
        Args:
            score: Raw score to normalize
            center: Center point of sigmoid (default 0.0)
            scale: Scale factor for sigmoid steepness (default 1.0)
        
        Returns:
            Normalized score in (0,1) range
        """
        return self._sigmoid((score - center) / scale)

    def _adaptive_sigmoid_normalize(self, score, phase):
        """
        Adaptive sigmoid normalization based on game phase.
        Different phases have different score distributions.
        
        Args:
            score: Raw score to normalize
            phase: Game phase ('opening', 'midgame', 'endgame')
        
        Returns:
            Normalized score in (0,1) range
        """
        if phase == 'opening':
            # Opening: wider distribution, less aggressive normalization
            return self._sigmoid_normalize(score, center=0.0, scale=2.0)
        elif phase == 'midgame':
            # Midgame: balanced normalization
            return self._sigmoid_normalize(score, center=0.0, scale=1.5)
        else:  # endgame
            # Endgame: tighter distribution, more aggressive normalization
            return self._sigmoid_normalize(score, center=0.0, scale=1.0)

    def _detect_game_phase(self, state):
        total_pieces = state.balls[PLAYER_ONE] + state.balls[PLAYER_TWO]
        empty_spaces = bin(~(state.player1_board | state.player2_board) & ((1 << 49) - 1)).count('1')
        total_positions = 49
        fill_ratio = total_pieces / (total_positions - empty_spaces + total_pieces)
        if total_pieces <= 8 or fill_ratio <= 0.3:
            return 'opening'
        elif total_pieces >= 25 or fill_ratio >= 0.7:
            return 'endgame'
        return 'midgame'

    def _cache_eval(self, state, player, score):
        state_key = (state.player1_board, state.player2_board, state.current_player(), player)
        self.eval_cache[state_key] = score
        if len(self.eval_cache) > self.max_cache_size:
            self.eval_cache.popitem(last=False)

    def _get_cached_eval(self, state, player):
        state_key = (state.player1_board, state.player2_board, state.current_player(), player)
        return self.eval_cache.get(state_key)
    
    # =============================================================================
    # TOURNAMENT LAYERING METHODS
    # =============================================================================
    
    def calculate_tournament_simulations(self, state, base_simulations=None):
        """
        Optimized calculation of simulation counts for tournament rounds.
        Reduced overhead while maintaining effectiveness.
        """
        phase = self._detect_game_phase(state)
        
        # Use basic_simulations from the instance if base_simulations not provided
        if base_simulations is None:
            base_simulations = getattr(self, 'basic_simulations', 200)  # Reduced default
        
        # Simplified ratios for better performance
        if phase == "opening":
            # Faster evaluation in opening
            s1_ratio, s2_ratio, s3_ratio = 0.10, 0.20, 0.25
        elif phase == "midgame":
            # Balanced approach
            s1_ratio, s2_ratio, s3_ratio = 0.15, 0.25, 0.30
        else:  # endgame
            # More intensive evaluation in critical endgame
            s1_ratio, s2_ratio, s3_ratio = 0.20, 0.30, 0.40
        
        # Calculate actual simulation counts with reasonable minimums
        S1 = max(3, int(base_simulations * s1_ratio))
        S2 = max(5, int(base_simulations * s2_ratio))
        S3 = max(8, int(base_simulations * s3_ratio))
        
        return (S1, S2, S3)
    
    def _remove_duplicate_moves(self, moves):
        """Remove moves with same destination to avoid redundant calculations."""
        seen_destinations = set()
        unique_moves = []
        
        for move in moves:
            destination = move[1]  # (x, y) position
            if destination not in seen_destinations:
                unique_moves.append(move)
                seen_destinations.add(destination)
        
        return unique_moves
    
    def _get_move_cache_key(self, state, move):
        """Generate cache key for move evaluation."""
        board_hash = hash(tuple(tuple(row) for row in state.board))
        return (board_hash, move, state.current_player())
    
    def _run_single_simulation(self, state, original_player):
        """Run a single Monte Carlo simulation from given state."""
        simulation_state = deepcopy(state)
        depth = 0
        max_depth = 15
        
        while depth < max_depth and not simulation_state.is_game_over():
            moves = simulation_state.get_all_possible_moves()
            if not moves:
                # No moves available, switch player
                simulation_state.toggle_player()
                moves = simulation_state.get_all_possible_moves()
                if not moves:
                    break  # Game truly over
            
            # Random move selection for simulation speed
            move = random.choice(moves)
            simulation_state.move_with_position(move)
            simulation_state.toggle_player()
            depth += 1
        
        # Evaluate final state
        return self._evaluate_final_position(simulation_state, original_player)
    
    def _evaluate_moves_sequential(self, state, moves, simulations_per_move):
        """Evaluate moves sequentially (fallback method)."""
        results = {}
        
        for move in moves:
            # Apply move temporarily
            move_state = deepcopy(state)
            move_state.move_with_position(move)
            move_state.toggle_player()
            
            # Run simulations for this move
            total_reward = 0.0
            for _ in range(simulations_per_move):
                reward = self._run_single_simulation(move_state, state.current_player())
                total_reward += reward

            results[move] = total_reward

        return results
    
    def _get_move_with_tournament_layering(self, state, base_simulations=None):
        """Tournament layering approach for move selection with 3 rounds."""
        moves = state.get_all_possible_moves()
        if not moves:
            return None
        
        if len(moves) == 1:
            return moves[0]
        
        # Remove duplicate destinations to avoid redundant calculations
        unique_moves = self._remove_duplicate_moves(moves)
        
        # Calculate simulation budgets based on game phase and base simulations
        S1, S2, S3 = self.calculate_tournament_simulations(state, base_simulations)
        k1 = min(5, len(unique_moves))  # Top moves for round 2
        k2 = min(3, len(unique_moves))  # Top moves for round 3
        
        print(f"Tournament: {len(unique_moves)} unique moves, S1={S1}, S2={S2}, S3={S3}")
        
        # ROUND 1: Evaluate all moves with S1 parallel simulations each
        print(f"Round 1: Evaluating {len(unique_moves)} moves with {S1} parallel simulations each...")
        round1_sim_scores = self._evaluate_moves_with_parallel_sims(state, unique_moves, S1)
        
        # Add structured evaluation for Round 1
        phase = self._detect_game_phase(state)
        round1_results = []
        for move, sim_score in round1_sim_scores.items():
            structured_score = self._calculate_comprehensive_move_score(state, move, phase)
            # Combine: 60% simulation, 40% heuristic in Round 1
            combined_score = 0.6 * sim_score + 0.4 * structured_score
            round1_results.append((move, combined_score, sim_score, structured_score, S1))
        
        # Sort by combined score and select top k1 moves
        round1_results.sort(key=lambda x: x[1], reverse=True)
        round2_candidates = round1_results[:k1]
        
        print(f"Round 1 complete. Top {k1} moves advance to Round 2:")
        for i, (move, combined_score, sim_score, struct_score, _) in enumerate(round2_candidates):
            print(f"  {i+1}. Move {move}: combined={combined_score:.3f} (sim={sim_score:.3f}, struct={struct_score:.3f})")
        
        # ROUND 2: Additional S2 parallel simulations for top k1 moves
        print(f"Round 2: {k1} moves with {S2} additional parallel simulations...")
        round2_results = []
        
        for move, prev_combined, prev_sim_score, prev_struct_score, prev_sims in round2_candidates:
            # Run additional parallel simulations for this move
            move_state = deepcopy(state)
            move_state.move_with_position(move)
            move_state.toggle_player()
            
            additional_score = self._run_parallel_simulations_for_move(move_state, move, S2)
            
            # Combine with previous simulation results
            total_sim_reward = prev_sim_score * prev_sims + additional_score * S2
            total_sims = prev_sims + S2
            avg_sim_score = total_sim_reward / total_sims
            
            # Update combined score: 65% simulation, 35% heuristic in Round 2
            combined_score = 0.65 * avg_sim_score + 0.35 * prev_struct_score
            
            round2_results.append((move, combined_score, avg_sim_score, prev_struct_score, total_sims))
            print(f"  Move {move}: combined={combined_score:.3f} (sim={avg_sim_score:.3f}, struct={prev_struct_score:.3f}, sims={total_sims})")
        
        # Sort and select top k2 moves
        round2_results.sort(key=lambda x: x[1], reverse=True)
        round3_candidates = round2_results[:k2]
        
        print(f"Round 2 complete. Top {k2} moves advance to Round 3:")
        for i, (move, score, _, _, _) in enumerate(round3_candidates):
            print(f"  {i+1}. Move {move}: score={score:.3f}")
        
        # ROUND 3: Final S3 simulations for top k2 moves
        print(f"Round 3: {k2} moves with {S3} additional simulations...")
        final_results = []
        
        for move, prev_combined, prev_sim_score, prev_struct_score, prev_sims in round3_candidates:
            # Run final simulations
            move_state = deepcopy(state)
            move_state.move_with_position(move)
            move_state.toggle_player()
            
            final_total = 0.0
            for _ in range(S3):
                reward = self._run_single_simulation(move_state, state.current_player())
                final_total += reward
            
            final_avg_sim = final_total
            
            # Final score calculation: combine all simulation results
            total_sim_reward = prev_sim_score * prev_sims + final_avg_sim * S3
            total_sims = prev_sims + S3
            final_avg_sim_score = total_sim_reward / total_sims
            
            # Final combined score: 70% simulation, 30% heuristic
            combined_score = 0.70 * final_avg_sim_score + 0.30 * prev_struct_score
            
            final_results.append((move, combined_score, final_avg_sim_score, prev_struct_score, total_sims))
            
            print(f"  {move}: combined={combined_score:.3f}, sim={final_avg_sim_score:.3f}, "
                  f"struct={prev_struct_score:.3f}, total_sims={total_sims}")
        
        # Select best move
        final_results.sort(key=lambda x: x[1], reverse=True)
        best_move = final_results[0][0]
        best_score = final_results[0][1]
        
        print(f"Tournament Winner: {best_move} (score: {best_score:.3f})")
        return best_move
    
    def get_mcd_move(self, time_limit=None, base_simulations=None):
        """
        Optimized main method to get the best move using improved tournament layering.
        
        Improvements:
        1. Adaptive strategy based on move count and game phase
        2. Faster evaluation for simple positions
        3. Better time management
        4. Reduced computational overhead
        """
        state = self.root_state
        moves = state.get_all_possible_moves()
        
        if not moves:
            return None
        
        if len(moves) == 1:
            return moves[0]
        
        # Remove duplicate destinations to avoid redundant calculations
        unique_moves = self._remove_duplicate_moves(moves)
        
        # Adaptive strategy based on complexity
        if len(unique_moves) <= 3:
            # Simple position: use fast comprehensive scoring
            best_move = None
            best_score = -1.0
            phase = self._detect_game_phase(state)
            
            for move in unique_moves:
                score = self._calculate_comprehensive_move_score(state, move, phase)
                if score > best_score:
                    best_score = score
                    best_move = move
            
            return best_move
        
        elif len(unique_moves) <= 8:
            # Moderate complexity: simplified tournament
            return self._simplified_tournament_selection(state, unique_moves, base_simulations)
        
        else:
            # High complexity: full tournament with optimizations
            return self._get_move_with_optimized_tournament_layering(state, unique_moves, base_simulations)
    
    # =============================================================================
    # PARALLEL PROCESSING METHODS
    # =============================================================================
    
    def _evaluate_moves_parallel(self, state, moves, simulations_per_move, max_workers=None):
        """Evaluate moves using parallel processing."""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(moves), 8)
        
        move_scores = {}
        
        # For small workloads, use sequential processing
        if len(moves) <= 2 or simulations_per_move <= 10:
            return self._evaluate_moves_sequential(state, moves, simulations_per_move)
        
        # Prepare state data for multiprocessing
        state_data = {
            'player1_board': state.player1_board,
            'player2_board': state.player2_board,
            'balls': state.balls,
            'turn_player': state.turn_player
        }
        
        original_player = state.current_player()
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks
                future_to_move = {}
                for move in moves:
                    future = executor.submit(
                        _evaluate_move_for_multiprocessing,
                        state_data,
                        move,
                        simulations_per_move,
                        original_player
                    )
                    future_to_move[future] = move
                
                # Collect results
                for future in as_completed(future_to_move):
                    move = future_to_move[future]
                    try:
                        score = future.result(timeout=30)
                        move_scores[move] = score
                    except Exception as e:
                        print(f"Error evaluating move {move}: {e}")
                        move_scores[move] = self._quick_move_evaluation(state, move)
        
        except Exception as e:
            print(f"Parallel processing failed: {e}, falling back to sequential")
            return self._evaluate_moves_sequential(state, moves, simulations_per_move)
        
        return move_scores
    
    def _evaluate_move_batch(self, state, move, num_simulations):
        """Evaluate a single move with multiple simulations (for parallel execution)."""
        total_score = 0.0
        original_player = state.current_player()
        
        for _ in range(num_simulations):
            # Create new state with the move applied
            new_state = deepcopy(state)
            new_state.move_with_position(move)
            new_state.toggle_player()
            
            # Run simulation
            simulation_score = self._run_single_simulation(new_state, original_player)
            total_score += simulation_score
        
        return total_score
    
    def _adaptive_parallel_strategy(self, state, moves, total_time_limit=None):
        """Adaptively choose parallel vs sequential based on workload and time."""
        num_moves = len(moves)
        cpu_count = mp.cpu_count()
        
        # Calculate adaptive simulation counts
        if total_time_limit:
            # Time-based adaptation
            base_simulations = max(5, min(50, int(total_time_limit * 10)))
        else:
            # Move-count based adaptation
            if num_moves <= 5:
                base_simulations = 25
            elif num_moves <= 10:
                base_simulations = 15
            else:
                base_simulations = 10
        
        # Decide parallel vs sequential
        estimated_time_per_simulation = 0.01  # seconds
        total_estimated_time = num_moves * base_simulations * estimated_time_per_simulation
        
        use_parallel = (
            num_moves >= 3 and
            cpu_count >= 2 and
            total_estimated_time > 0.5  # Only parallelize if >0.5 seconds
        )
        
        if use_parallel:
            max_workers = min(cpu_count, num_moves, 6)  # Conservative worker count
            return self._evaluate_moves_parallel(state, moves, base_simulations, max_workers)
        else:
            return self._evaluate_moves_sequential(state, moves, base_simulations)
    
    def _parallel_tournament_evaluation(self, state, moves, tournament_config):
        """Run tournament evaluation with parallel processing for each round."""
        s1_sims, s2_sims, s3_sims = tournament_config
        
        # Round 1: Parallel evaluation with basic simulations
        print(f"Tournament Round 1: {len(moves)} moves, {s1_sims} simulations each")
        round1_scores = self._evaluate_moves_parallel(state, moves, s1_sims)
        
        # Select top 60% for round 2
        sorted_moves = sorted(round1_scores.items(), key=lambda x: x[1], reverse=True)
        round2_moves = [move for move, _ in sorted_moves[:max(1, int(len(moves) * 0.6))]]
        
        # Round 2: More intensive evaluation
        print(f"Tournament Round 2: {len(round2_moves)} moves, {s2_sims} simulations each")
        round2_scores = self._evaluate_moves_parallel(state, round2_moves, s2_sims)
        
        # Select top 40% for round 3
        sorted_moves = sorted(round2_scores.items(), key=lambda x: x[1], reverse=True)
        round3_moves = [move for move, _ in sorted_moves[:max(1, int(len(round2_moves) * 0.4))]]
        
        # Round 3: Final intensive evaluation
        print(f"Tournament Round 3: {len(round3_moves)} moves, {s3_sims} simulations each")
        round3_scores = self._evaluate_moves_parallel(state, round3_moves, s3_sims)
        
        # Combine with structured evaluation
        final_scores = {}
        for move in round3_moves:
            simulation_score = round3_scores[move]
            phase = self._detect_game_phase(state)
            structured_score = self._calculate_comprehensive_move_score(state, move, phase)
            # Weight: 70% simulation, 30% heuristic
            final_scores[move] = 0.7 * simulation_score + 0.3 * structured_score
        
        return final_scores
    
    def _threaded_simulation_batch(self, state, moves, simulations_per_move, num_threads=4):
        """Use threading for I/O-bound or lightweight simulation batches."""
        import threading
        from queue import Queue
        
        results = {}
        results_lock = threading.Lock()
        task_queue = Queue()
        
        # Fill task queue
        for move in moves:
            task_queue.put(move)
        
        def worker():
            while True:
                try:
                    move = task_queue.get_nowait()
                    score = self._evaluate_move_batch(state, move, simulations_per_move)
                    
                    with results_lock:
                        results[move] = score
                    
                    task_queue.task_done()
                except:
                    break
        
        # Start worker threads
        threads = []
        for _ in range(min(num_threads, len(moves))):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()
            threads.append(t)
        
        # Wait for completion
        task_queue.join()
        
        return results
    
    def get_move_parallel(self, time_limit=None, parallel_mode='auto', base_simulations=None):
        """
        Get the best move using parallel processing.
        
        Args:
            time_limit: Maximum time to spend on move selection
            parallel_mode: 'auto', 'process', 'thread', 'sequential'
            base_simulations: Base number of simulations to use for tournament layering
        """
        state = self.root_state
        moves = state.get_all_possible_moves()
        
        if not moves:
            return None
        
        if len(moves) == 1:
            return moves[0]
        
        # Remove duplicates
        moves = self._remove_duplicate_moves(moves)
        
        start_time = time.time()
        
        try:
            if parallel_mode == 'auto':
                # Adaptive strategy
                move_scores = self._adaptive_parallel_strategy(state, moves, time_limit)
            elif parallel_mode == 'process':
                # Force multiprocessing
                tournament_config = self.calculate_tournament_simulations(state, self.basic_simulations)
                move_scores = self._parallel_tournament_evaluation(state, moves, tournament_config)
            elif parallel_mode == 'thread':
                # Force threading
                simulations = sum(self.calculate_tournament_simulations(state, self.basic_simulations)) // 3
                move_scores = self._threaded_simulation_batch(state, moves, simulations)
            else:
                # Sequential fallback
                return self.get_mcd_move(time_limit)
            
            # Select best move
            if move_scores:
                best_move = max(move_scores.items(), key=lambda x: x[1])[0]
                elapsed = time.time() - start_time
                print(f"Parallel move selection completed in {elapsed:.2f}s")
                return best_move
            
        except Exception as e:
            print(f"Parallel processing error: {e}, falling back to sequential")
        
        # Fallback to sequential
        return self.get_mcd_move(time_limit)
    
    def _quick_move_evaluation(self, state, move):
        """Quick evaluation for fallback when parallel processing fails."""
        new_state = deepcopy(state)
        new_state.move_with_position(move)
        
        # Use comprehensive move scoring for quick evaluation
        player = state.current_player()
        phase = self._detect_game_phase(state)
        return self._calculate_comprehensive_move_score(state, move, phase)
    
    # =============================================================================
    # PARALLEL SIMULATION PER MOVE
    # =============================================================================
    
    def _run_parallel_simulations_for_move(self, state, move, num_simulations, max_workers=None):
        """Run multiple simulations in parallel for a single move."""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), num_simulations, 8)
        
        # For small simulation counts, use sequential to avoid overhead
        if num_simulations <= 4:
            return self._run_sequential_simulations_for_move(state, move, num_simulations)
        
        original_player = state.current_player()
        
        # Prepare state data for multiprocessing
        state_data = {
            'player1_board': state.player1_board,
            'player2_board': state.player2_board,
            'balls': state.balls,
            'turn_player': state.turn_player
        }
        
        try:
            # Split simulations across workers
            simulations_per_worker = max(1, num_simulations // max_workers)
            remaining_sims = num_simulations % max_workers
            
            simulation_batches = []
            for i in range(max_workers):
                batch_size = simulations_per_worker
                if i < remaining_sims:
                    batch_size += 1
                if batch_size > 0:
                    simulation_batches.append(batch_size)
            
            with ProcessPoolExecutor(max_workers=len(simulation_batches)) as executor:
                # Submit simulation batches
                futures = []
                for batch_size in simulation_batches:
                    future = executor.submit(
                        _run_simulation_batch_for_move,
                        state_data,
                        move,
                        batch_size,
                        original_player
                    )
                    futures.append(future)
                
                # Collect results
                total_score = 0.0
                completed_simulations = 0
                
                for future in as_completed(futures):
                    try:
                        batch_score, batch_count = future.result(timeout=15)
                        total_score += batch_score * batch_count
                        completed_simulations += batch_count
                    except Exception as e:
                        print(f"Simulation batch failed: {e}")
                        # Continue with other batches
                
                if completed_simulations > 0:
                    return total_score / completed_simulations
                else:
                    # Fallback to sequential if all batches failed
                    return self._run_sequential_simulations_for_move(state, move, min(5, num_simulations))
                    
        except Exception as e:
            print(f"Parallel simulation failed: {e}, falling back to sequential")
            return self._run_sequential_simulations_for_move(state, move, num_simulations)
    
    def _run_sequential_simulations_for_move(self, state, move, num_simulations):
        """Run simulations sequentially for a single move (fallback method)."""
        total_score = 0.0
        original_player = state.current_player()
        
        for _ in range(num_simulations):
            # Apply move to create new state
            new_state = deepcopy(state)
            new_state.move_with_position(move)
            new_state.toggle_player()
            
            # Run simulation
            score = self._run_single_simulation(new_state, original_player)
            total_score += score
        
        return total_score
    
    def _evaluate_moves_with_parallel_sims(self, state, moves, simulations_per_move):
        """Evaluate each move using parallel simulations per move."""
        move_scores = {}
        total_moves = len(moves)
        
        print(f"Evaluating {total_moves} moves with {simulations_per_move} parallel simulations each")
        
        for i, move in enumerate(moves):
            print(f"Move {i+1}/{total_moves}: {move}")
            
            # Run parallel simulations for this specific move
            score = self._run_parallel_simulations_for_move(state, move, simulations_per_move)
            move_scores[move] = score
            
            print(f"  Score: {score:.4f}")
        
        return move_scores

    # =============================================================================
    # TACTICAL EVALUATION (T) METHODS
    # =============================================================================
    
    def _calculate_tactical_score(self, current_state, move, next_state):
        """Calculate tactical evaluation score for a move."""
        corner_score = self._corner_avoidance_score(current_state, move, next_state)
        suicide_score = self._no_suicide_move_score(current_state, move, next_state)
        convert_score = self._max_convert_score(current_state, move, next_state)
        
        # Weighted combination of tactical components
        tactical_score = (
            0.4 * corner_score +     # Corner avoidance is critical
            0.4 * suicide_score +    # Avoiding suicide moves is critical
            0.2 * convert_score      # Max convert is beneficial but less critical
        )
        
        return tactical_score
    
    def _corner_avoidance_score(self, current_state, move, next_state):
        """Prevent getting trapped in corners - negative score for corner moves."""
        from_pos, to_pos = move
        
        # Define corner positions (0,0), (0,6), (6,0), (6,6)
        corners = [(0, 0), (0, 6), (6, 0), (6, 6)]
        
        penalty = 0.0
        
        # Heavy penalty for moving TO a corner
        if to_pos in corners:
            penalty += 10.0
            
        # Check if move creates a trapped piece in corner
        if to_pos in corners:
            # Check mobility from corner position
            corner_bit = to_pos[0] * 7 + to_pos[1]
            my_player = current_state.current_player()
            
            # Create test state to see available moves from corner
            test_state = Ataxx()
            test_state.player1_board = next_state.player1_board
            test_state.player2_board = next_state.player2_board
            test_state.balls = next_state.balls.copy()
            test_state.turn_player = my_player
            
            # Count available moves from this corner position
            available_moves = bin(self.adjacent_masks[corner_bit] & 
                                ~(test_state.player1_board | test_state.player2_board)).count('1')
            available_moves += bin(self.jump_masks[corner_bit] & 
                                 ~(test_state.player1_board | test_state.player2_board)).count('1')
            
            # Additional penalty if corner piece has very limited mobility
            if available_moves <= 2:
                penalty += 5.0
            elif available_moves <= 4:
                penalty += 3.0
        
        # Penalty for moving close to corners when it's risky
        corner_adjacent = [(0, 1), (1, 0), (1, 1),  # Near (0,0)
                          (0, 5), (1, 6), (1, 5),   # Near (0,6)
                          (5, 0), (6, 1), (5, 1),   # Near (6,0)
                          (5, 6), (6, 5), (5, 5)]   # Near (6,6)
        
        if to_pos in corner_adjacent:
            penalty += 2.0
            
        # Return normalized score (higher is better, so negate penalty)
        return max(0.0, 1.0 - penalty / 20.0)
    
    def _no_suicide_move_score(self, current_state, move, next_state):
        """Avoid moves that lead to immediate disadvantage or piece loss."""
        from_pos, to_pos = move
        my_player = current_state.current_player()
        opp_player = -my_player
        
        penalty = 0.0
        
        # Check if move exposes our pieces to enemy capture
        # Simulate opponent's response to see if they can capture us immediately
        
        # 1. Check if the moved piece can be immediately captured
        to_bit = to_pos[0] * 7 + to_pos[1]
        opp_board = next_state.player2_board if my_player == PLAYER_ONE else next_state.player1_board
        
        # Can opponent immediately capture our new piece?
        enemy_pieces_around = bin(self.adjacent_masks[to_bit] & opp_board).count('1')
        if enemy_pieces_around > 0:
            # Check if opponent can actually make a capturing move
            for dx, dy in ADJACENT_POSITIONS + JUMP_POSITIONS:
                enemy_x, enemy_y = to_pos[0] + dx, to_pos[1] + dy
                if (0 <= enemy_x < 7 and 0 <= enemy_y < 7 and 
                    opp_board & (1 << (enemy_x * 7 + enemy_y))):
                    # Enemy can potentially capture, add penalty
                    penalty += 3.0
                    break
        
        # 2. Check if move leaves our other pieces vulnerable
        my_board = next_state.player1_board if my_player == PLAYER_ONE else next_state.player2_board
        my_pieces = [(i // 7, i % 7) for i in range(49) if my_board & (1 << i)]
        
        vulnerable_pieces = 0
        for piece_pos in my_pieces:
            piece_bit = piece_pos[0] * 7 + piece_pos[1]
            # Count enemy pieces that can reach this piece
            enemy_threats = bin(self.adjacent_masks[piece_bit] & opp_board).count('1')
            
            # Count our supporting pieces
            our_support = bin(self.adjacent_masks[piece_bit] & my_board).count('1') - 1  # -1 for the piece itself
            
            # If enemy threats > our support, piece is vulnerable
            if enemy_threats > our_support and enemy_threats > 0:
                vulnerable_pieces += 1
        
        # Penalty for creating vulnerable pieces
        penalty += vulnerable_pieces * 2.0
        
        # 3. Check if move reduces our mobility significantly
        current_mobility = len(current_state.get_all_possible_moves())
        
        # Create temporary state to check next mobility
        temp_state = Ataxx()
        temp_state.player1_board = next_state.player1_board  
        temp_state.player2_board = next_state.player2_board
        temp_state.balls = next_state.balls.copy()
        temp_state.turn_player = my_player
        next_mobility = len(temp_state.get_all_possible_moves())
        
        mobility_reduction = current_mobility - next_mobility
        if mobility_reduction > current_mobility * 0.5:  # Lost more than 50% mobility
            penalty += 5.0
        elif mobility_reduction > current_mobility * 0.3:  # Lost more than 30% mobility
            penalty += 3.0
        
        # Return normalized score (higher is better)
        return max(0.0, 1.0 - penalty / 15.0)
    
    def _max_convert_score(self, current_state, move, next_state):
        """Maximize immediate piece captures/conversions."""
        from_pos, to_pos = move
        my_player = current_state.current_player()
        opp_player = -my_player
        
        # Count pieces we gain from this move
        current_my_pieces = current_state.balls[my_player]
        next_my_pieces = next_state.balls[my_player]
        pieces_gained = next_my_pieces - current_my_pieces
        
        # Count opponent pieces we captured/converted
        current_opp_pieces = current_state.balls[opp_player]
        next_opp_pieces = next_state.balls[opp_player]
        pieces_captured = current_opp_pieces - next_opp_pieces
        
        # Calculate conversion score
        to_bit = to_pos[0] * 7 + to_pos[1]
        opp_board = current_state.player2_board if my_player == PLAYER_ONE else current_state.player1_board
        
        # Direct captures from the target position
        direct_captures = bin(self.adjacent_masks[to_bit] & opp_board).count('1')
        
        # Bonus for moves that capture multiple pieces
        capture_bonus = 0.0
        if direct_captures >= 3:
            capture_bonus = 3.0
        elif direct_captures == 2:
            capture_bonus = 2.0
        elif direct_captures == 1:
            capture_bonus = 1.0
        
        # Consider potential follow-up captures
        follow_up_potential = 0.0
        my_board = next_state.player1_board if my_player == PLAYER_ONE else next_state.player2_board
        
        # Check if this move sets up future captures
        for dx, dy in ADJACENT_POSITIONS:
            check_x, check_y = to_pos[0] + dx, to_pos[1] + dy
            if (0 <= check_x < 7 and 0 <= check_y < 7):
                check_bit = check_x * 7 + check_y
                if opp_board & (1 << check_bit):
                    # Count how many of our pieces would be adjacent to this opponent piece
                    our_adjacent = bin(self.adjacent_masks[check_bit] & my_board).count('1')
                    if our_adjacent >= 2:  # Multiple pieces can potentially capture
                        follow_up_potential += 0.5
        
        # Normalize score based on pieces gained and strategic value
        base_score = (pieces_gained + pieces_captured) / 8.0  # Max ~8 pieces could be gained
        bonus_score = (capture_bonus + follow_up_potential) / 5.0  # Normalize bonuses
        
        return min(1.0, base_score + bonus_score)
    
    # =============================================================================
    # ENHANCED SCORING SYSTEM - NEW IMPLEMENTATION
    # =============================================================================

    def _calculate_move_heuristic_score(self, state, move, phase):
        """
        Calculate heuristic score using the formula S(m, s, p) = s1×C(m,s) + s2×A(m,s) + s3×B(m) - s4×P(m,s)
        
        Where:
        • C(m, s): Number of opponent pieces captured when executing move m in state s  
        • A(m, s): Number of ally pieces around the target square of move m
        • B(m): Bonus for Clone move (1 if Clone, 0 if Jump)
        • P(m, s): Penalty for Jump move - number of ally pieces around source square if Jump
        • s1, s2, s3, s4: Dynamic weights according to game phase p
        """
        from_pos, to_pos = move
        my_player = state.current_player()
        my_board = state.player1_board if PLAYER_ONE else state.player2_board
        opp_board = state.player2_board if PLAYER_ONE else state.player1_board
        
        # Get phase-based weights for heuristic formula
        weights = self._get_phase_weights(phase)
        s1, s2, s3, s4 = weights['s1'], weights['s2'], weights['s3'], weights['s4']
        
        # C(m, s): Captured opponent pieces
        to_bit = to_pos[0] * 7 + to_pos[1]
        captures = bin(self.adjacent_masks[to_bit] & opp_board).count('1')
        
        # A(m, s): Allied pieces around target
        allied_around_target = bin(self.adjacent_masks[to_bit] & my_board).count('1')
        
        # B(m): Clone bonus (1 if clone, 0 if jump)  
        clone_bonus = 1 if from_pos is None else 0
        
        # P(m, s): Jump penalty (allied pieces around source)
        jump_penalty = 0
        if from_pos is not None:  # Jump move
            from_bit = from_pos[0] * 7 + from_pos[1]
            jump_penalty = bin(self.adjacent_masks[from_bit] & my_board).count('1')
        
        # Calculate S(m, s, p) using the formula
        heuristic_score = (s1 * captures + 
                          s2 * allied_around_target + 
                          s3 * clone_bonus - 
                          s4 * jump_penalty)
        
        return heuristic_score
    
    def _get_phase_weights(self, phase):
        """Get dynamic weights based on game phase."""
        if phase == 'opening':
            return {'s1': 1.2, 's2': 0.8, 's3': 1.5, 's4': 0.6}  # Favor expansion and clone moves
        elif phase == 'midgame':
            return {'s1': 1.0, 's2': 1.0, 's3': 1.0, 's4': 1.0}  # Balanced approach
        else:  # endgame
            return {'s1': 1.5, 's2': 1.2, 's3': 0.7, 's4': 1.3}  # Favor captures and consolidation

    def _calculate_enhanced_tactical_score(self, state, move, phase):
        """
        Enhanced tactical evaluation with four main components:
        1. Corner Avoidance: Avoid dangerous corner positions
        2. Blocking: Limit opponent mobility  
        3. Centrality: Control center positions
        4. Allied Safety: Avoid exposing many pieces to enemy attack
        """
        from_pos, to_pos = move
        my_player = state.current_player()
        my_board = state.player1_board if my_player == PLAYER_ONE else state.player2_board
        opp_board = state.player2_board if my_player == PLAYER_ONE else state.player1_board
        
        # 1. Corner Avoidance Score
        corner_score = self._corner_avoidance_tactical_score(move, state)
        
        # 2. Blocking Score  
        blocking_score = self._blocking_tactical_score(move, state)
        
        # 3. Centrality Score
        centrality_score = self._centrality_tactical_score(move, state)
        
        # 4. Allied Safety Score - penalize moves that expose many allies
        safety_penalty = 0.0
        if self._would_lose_many_allies(move, state):
            safety_penalty = -5.0  # Heavy penalty for exposing allies
        
        # Weight components based on phase
        if phase == 'opening':
            # Early game: prioritize centrality and avoid corners, moderate safety concern
            weights = {'corner': 0.4, 'blocking': 0.15, 'centrality': 0.3, 'safety': 0.15}
        elif phase == 'midgame':
            # Mid game: balanced approach with more blocking and safety
            weights = {'corner': 0.25, 'blocking': 0.35, 'centrality': 0.25, 'safety': 0.15}
        else:  # endgame
            # Late game: focus on blocking, corner safety, and protecting pieces
            weights = {'corner': 0.3, 'blocking': 0.4, 'centrality': 0.1, 'safety': 0.2}
        
        tactical_score = (weights['corner'] * corner_score + 
                         weights['blocking'] * blocking_score + 
                         weights['centrality'] * centrality_score +
                         weights['safety'] * safety_penalty)
        
        return tactical_score
    
    def _corner_avoidance_tactical_score(self, move, state):
        """Tactical corner avoidance - prevent getting trapped."""
        from_pos, to_pos = move
        corners = [(0, 0), (0, 6), (6, 0), (6, 6)]
        
        score = 0.0
        
        # Heavy penalty for moving TO a corner
        if to_pos in corners:
            score -= 8.0
            
            # Additional penalty if corner has limited escape routes
            to_bit = to_pos[0] * 7 + to_pos[1]
            my_player = state.current_player()
            opp_board = state.player2_board if my_player == PLAYER_ONE else state.player1_board
            
            # Check escape routes from corner
            escape_routes = bin(self.adjacent_masks[to_bit] & 
                               ~(state.player1_board | state.player2_board)).count('1')
            escape_routes += bin(self.jump_masks[to_bit] & 
                                ~(state.player1_board | state.player2_board)).count('1')
            
            if escape_routes <= 3:
                score -= 5.0  # Very dangerous corner
            elif escape_routes <= 6:
                score -= 2.0  # Somewhat dangerous
        
        # Penalty for moving adjacent to corners
        corner_adjacent = [(0, 1), (1, 0), (1, 1), (0, 5), (1, 6), (1, 5), 
                          (5, 0), (6, 1), (5, 1), (5, 6), (6, 5), (5, 5)]
        if to_pos in corner_adjacent:
            score -= 2.0
        
        return score
    
    def _blocking_tactical_score(self, move, state):
        """Tactical blocking - limit opponent mobility."""
        from_pos, to_pos = move
        
        # Apply move temporarily to evaluate blocking effect
        temp_state = deepcopy(state)
        temp_state.move_with_position(move)
        temp_state.toggle_player()  # Switch to opponent
        
        # Count opponent's available moves after our move
        opp_moves = len(temp_state.get_all_possible_moves())
        
        # Score based on mobility reduction
        if opp_moves == 0:
            return 10.0  # Complete blockade
        elif opp_moves <= 3:
            return 6.0   # Severe restriction
        elif opp_moves <= 8:
            return 3.0   # Moderate restriction
        elif opp_moves <= 15:
            return 1.0   # Light restriction
        else:
            return 0.0   # No significant restriction
    
    def _centrality_tactical_score(self, move, state):
        """Tactical centrality - control important board positions."""
        from_pos, to_pos = move
        
        # Define center positions with different values
        center_core = [(3, 3)]  # Most valuable
        center_ring1 = [(2, 3), (3, 2), (3, 4), (4, 3)]  # High value
        center_ring2 = [(2, 2), (2, 4), (4, 2), (4, 4)]  # Medium value
        
        score = 0.0
        
        # Bonus for occupying center positions
        if to_pos in center_core:
            score += 4.0
        elif to_pos in center_ring1:
            score += 2.5
        elif to_pos in center_ring2:
            score += 1.5
        
        # Additional bonus for controlling center area
        to_bit = to_pos[0] * 7 + to_pos[1]
        my_player = state.current_player()
        my_board = state.player1_board if my_player == PLAYER_ONE else state.player2_board
        
        # Count how many center positions we can influence from target position
        center_influence = 0
        all_center = center_core + center_ring1 + center_ring2
        for center_pos in all_center:
            center_bit = center_pos[0] * 7 + center_pos[1]
            if (self.adjacent_masks[to_bit] | self.jump_masks[to_bit]) & (1 << center_bit):
                center_influence += 1
        
        score += center_influence * 0.3
        
        return score
    
    def _would_lose_many_allies(self, move, state):
        """Helper function to check if move exposes many allied pieces."""
        from_pos, to_pos = move
        my_player = state.current_player()
        
        # Apply move and check for vulnerabilities
        temp_state = deepcopy(state)
        temp_state.move_with_position(move)
        
        my_board = temp_state.player1_board if my_player == PLAYER_ONE else temp_state.player2_board
        opp_board = temp_state.player2_board if my_player == PLAYER_ONE else temp_state.player1_board
        
        vulnerable_count = 0
        my_pieces = [(i // 7, i % 7) for i in range(49) if my_board & (1 << i)]
        
        for piece_pos in my_pieces:
            piece_bit = piece_pos[0] * 7 + piece_pos[1]
            # Count enemy threats around this piece
            enemy_threats = bin(self.adjacent_masks[piece_bit] & opp_board).count('1')
            # Count our support around this piece (excluding the piece itself)
            our_support = bin(self.adjacent_masks[piece_bit] & my_board).count('1') - 1
            
            # If threats > support, piece is vulnerable
            if enemy_threats > our_support and enemy_threats > 0:
                vulnerable_count += 1
        
        return vulnerable_count >= 2  # True if losing 2+ allies

    def _calculate_comprehensive_move_score(self, state, move, phase):
        """
        Optimized comprehensive move evaluation with improved balance and performance.
        
        Improvements:
        1. Simplified component calculation to reduce overhead
        2. Better normalization with bounded ranges
        3. More balanced phase-based weighting
        4. Faster computation without losing accuracy
        """
        
        # Fast pre-computation of move info
        from_pos, to_pos = move
        my_player = state.current_player()
        my_board = state.player1_board if my_player == PLAYER_ONE else state.player2_board
        opp_board = state.player2_board if my_player == PLAYER_ONE else state.player1_board
        to_bit = to_pos[0] * 7 + to_pos[1]
        
        # 1. Immediate rewards (captures and positioning)
        captures = bin(self.adjacent_masks[to_bit] & opp_board).count('1')
        clone_bonus = 1.0 if from_pos is None else 0.0
        
        # 2. Tactical safety check (simplified)
        corner_penalty = 0.0
        corners = [(0, 0), (0, 6), (6, 0), (6, 6)]
        if to_pos in corners:
            corner_penalty = -3.0
        
        # 3. Strategic positioning
        center_bonus = 0.0
        center_positions = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)]
        if to_pos in center_positions:
            center_bonus = 2.0 if to_pos == (3, 3) else 1.0
        
        # 4. Mobility consideration (simplified)
        allied_support = bin(self.adjacent_masks[to_bit] & my_board).count('1')
        mobility_bonus = min(allied_support * 0.5, 2.0)
        
        # 5. Phase-based component weights (simplified and balanced)
        if phase == 'opening':
            # Early game: favor expansion and safe positioning
            capture_weight = 2.0
            clone_weight = 1.5
            corner_weight = 2.0
            center_weight = 1.2
            mobility_weight = 0.8
        elif phase == 'midgame':
            # Mid game: balanced approach with emphasis on captures
            capture_weight = 3.0
            clone_weight = 1.0
            corner_weight = 2.5
            center_weight = 1.0
            mobility_weight = 1.0
        else:  # endgame
            # Late game: prioritize captures and safety
            capture_weight = 4.0
            clone_weight = 0.5
            corner_weight = 3.0
            center_weight = 0.5
            mobility_weight = 1.2
        
        # 6. Calculate final score with proper scaling
        base_score = (capture_weight * captures + 
                     clone_weight * clone_bonus + 
                     corner_weight * corner_penalty + 
                     center_weight * center_bonus + 
                     mobility_weight * mobility_bonus)
        
        # 7. Normalize to [0, 1] range with improved sigmoid
        # Expected range: -9 to +12, center at 1.5
        normalized_score = self._sigmoid((base_score - 1.5) / 3.0)
        
        return normalized_score

    def _calculate_move_strategic_score(self, state, move, phase):
        """
        Strategic evaluation for long-term positioning with three main components:
        1. Tempo Control: Maintain initiative and piece advantage
        2. Enemy Isolation: Fragment opponent's forces
        3. Reduce Enemy Mobility: Limit opponent's options
        """
        from_pos, to_pos = move
        my_player = state.current_player()
        
        # Apply move temporarily to evaluate strategic impact
        temp_state = deepcopy(state)
        temp_state.move_with_position(move)
        
        # 1. Tempo Control Score
        tempo_score = self._tempo_control_strategic_score(state, temp_state, phase)
        
        # 2. Enemy Isolation Score  
        isolation_score = self._enemy_isolation_strategic_score(state, temp_state, phase)
        
        # 3. Enemy Mobility Reduction Score
        mobility_score = self._reduce_enemy_mobility_strategic_score(state, temp_state, phase)
        
        # Weight components based on phase
        if phase == 'opening':
            # Early game: focus on tempo and positioning
            weights = {'tempo': 0.5, 'isolation': 0.2, 'mobility': 0.3}
        elif phase == 'midgame':
            # Mid game: balanced strategic approach
            weights = {'tempo': 0.4, 'isolation': 0.3, 'mobility': 0.3}
        else:  # endgame
            # Late game: focus on isolation and mobility control
            weights = {'tempo': 0.2, 'isolation': 0.4, 'mobility': 0.4}
        
        strategic_score = (weights['tempo'] * tempo_score + 
                          weights['isolation'] * isolation_score + 
                          weights['mobility'] * mobility_score)
        
        return strategic_score
    
    def _tempo_control_strategic_score(self, current_state, next_state, phase):
        """Evaluate tempo control - maintaining initiative and piece advantage."""
        my_player = current_state.current_player()
        
        # 1. Piece advantage change
        old_advantage = current_state.balls[my_player] - current_state.balls[-my_player]
        new_advantage = next_state.balls[my_player] - next_state.balls[-my_player]
        advantage_gain = new_advantage - old_advantage
        
        score = advantage_gain * 2.0  # Base tempo score
        
        # 2. Board control evaluation
        my_board = next_state.player1_board if my_player == PLAYER_ONE else next_state.player2_board
        opp_board = next_state.player2_board if my_player == PLAYER_ONE else next_state.player1_board
        
        # Count controlled territory (influenced squares)
        my_influence = 0
        opp_influence = 0
        
        for x in range(7):
            for y in range(7):
                bit = x * 7 + y
                if not ((my_board | opp_board) & (1 << bit)):  # Empty square
                    # Check influence from both players
                    my_inf = bin((self.adjacent_masks[bit] | self.jump_masks[bit]) & my_board).count('1')
                    opp_inf = bin((self.adjacent_masks[bit] | self.jump_masks[bit]) & opp_board).count('1')
                    
                    if my_inf > opp_inf:
                        my_influence += 1
                    elif opp_inf > my_inf:
                        opp_influence += 1
        
        territory_control = my_influence - opp_influence
        score += territory_control * 0.5
        
        # 3. Phase-specific tempo adjustments
        if phase == 'opening':
            # Bonus for early expansion
            if advantage_gain > 0:
                score += 1.0
        elif phase == 'endgame':
            # Critical tempo in endgame
            score *= 1.5
        
        return score
    
    def _enemy_isolation_strategic_score(self, current_state, next_state, phase):
        """Evaluate enemy isolation - fragmenting opponent's forces."""
        opp_player = -current_state.current_player()
        opp_board = next_state.player2_board if opp_player == PLAYER_TWO else next_state.player1_board
        
        # Get opponent pieces positions
        opp_pieces = [(i // 7, i % 7) for i in range(49) if opp_board & (1 << i)]
        
        if not opp_pieces:
            return 5.0  # Maximum isolation if no opponent pieces
        
        # Count connected components using flood-fill
        visited = set()
        groups = 0
        largest_group = 0
        
        for piece in opp_pieces:
            if piece not in visited:
                groups += 1
                group_size = self._flood_fill_group_size(piece, opp_pieces, visited)
                largest_group = max(largest_group, group_size)
        
        # Score based on fragmentation
        total_pieces = len(opp_pieces)
        if total_pieces == 0:
            return 5.0
        
        # More groups = better isolation
        fragmentation_score = groups * 2.0
        
        # Smaller largest group = better isolation  
        if total_pieces > 0:
            largest_group_ratio = largest_group / total_pieces
            fragmentation_score += (1.0 - largest_group_ratio) * 3.0
        
        # Phase adjustments
        if phase == 'midgame' or phase == 'endgame':
            fragmentation_score *= 1.2  # More important in later phases
        
        return min(fragmentation_score, 8.0)  # Cap the score
    
    def _reduce_enemy_mobility_strategic_score(self, current_state, next_state, phase):
        """Evaluate reduction in enemy mobility options."""
        # Switch to opponent's turn to count their moves
        next_state.toggle_player()
        opp_moves = len(next_state.get_all_possible_moves())
        next_state.toggle_player()  # Switch back
        
        # Compare with estimated current opponent mobility
        current_state.toggle_player() 
        current_opp_moves = len(current_state.get_all_possible_moves())
        current_state.toggle_player()
        
        mobility_reduction = current_opp_moves - opp_moves
        
        # Score based on mobility restriction
        if opp_moves == 0:
            return 10.0  # Complete mobility elimination
        elif opp_moves <= 3:
            return 8.0   # Severe restriction
        elif opp_moves <= 8:
            return 5.0   # Significant restriction
        elif mobility_reduction > 5:
            return 3.0   # Good reduction
        elif mobility_reduction > 2:
            return 1.0   # Minor reduction
        else:
            return 0.0   # No significant reduction
    
    def _flood_fill_group_size(self, start_piece, all_pieces, visited):
        """Helper function to calculate connected group size using flood-fill."""
        if start_piece in visited:
            return 0
        
        queue = [start_piece]
        visited.add(start_piece)
        group_size = 1
        
        while queue:
            current = queue.pop(0)
            current_bit = current[0] * 7 + current[1]
            
            # Check adjacent and jump positions for connected pieces
            for dx, dy in ADJACENT_POSITIONS + JUMP_POSITIONS:
                neighbor_x, neighbor_y = current[0] + dx, current[1] + dy
                neighbor = (neighbor_x, neighbor_y)
                
                if (0 <= neighbor_x < 7 and 0 <= neighbor_y < 7 and 
                    neighbor in all_pieces and neighbor not in visited):
                    visited.add(neighbor)
                    queue.append(neighbor)
                    group_size += 1
        
        return group_size

    def _softmax_selection(self, move_scores, temperature=1.0):
        """Apply softmax for probabilistic move selection."""
        if not move_scores:
            return None
        
        moves, scores = zip(*move_scores.items())
        
        # Apply temperature scaling
        scaled_scores = [s / temperature for s in scores]
        
        # Numerical stability - subtract max
        max_score = max(scaled_scores)
        exp_scores = [math.exp(s - max_score) for s in scaled_scores]
        sum_exp = sum(exp_scores)
        
        if sum_exp == 0:
            return moves[0]
        
        # Calculate probabilities
        probabilities = [exp_s / sum_exp for exp_s in exp_scores]
        
        # Sample based on probabilities
        rand_val = random.random()
        cumulative = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if rand_val <= cumulative:
                return moves[i]
        
        return moves[-1]  # Fallback
    
    def _quick_filter_moves(self, state, moves, phase, max_candidates=12):
        """
        Quick filtering of moves to reduce evaluation overhead for large move sets.
        Uses simple heuristics to eliminate obviously poor moves.
        """
        if len(moves) <= max_candidates:
            return moves
        
        my_player = state.current_player()
        my_board = state.player1_board if my_player == PLAYER_ONE else state.player2_board
        opp_board = state.player2_board if my_player == PLAYER_ONE else state.player1_board
        
        # Quick scoring for filtering
        move_quick_scores = []
        corners = [(0, 0), (0, 6), (6, 0), (6, 6)]
        
        for move in moves:
            from_pos, to_pos = move
            to_bit = to_pos[0] * 7 + to_pos[1]
            
            # Quick score components
            captures = bin(self.adjacent_masks[to_bit] & opp_board).count('1')
            corner_penalty = -5.0 if to_pos in corners else 0.0
            clone_bonus = 2.0 if from_pos is None else 0.0
            
            quick_score = captures * 3.0 + clone_bonus + corner_penalty
            move_quick_scores.append((move, quick_score))
        
        # Sort by quick score and return top candidates
        move_quick_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_quick_scores[:max_candidates]]
    
    def _get_selection_temperature(self, phase, num_moves):
        """
        Get appropriate temperature for softmax selection based on phase and move count.
        Lower temperature = more deterministic, higher temperature = more exploratory.
        """
        base_temp = {
            'opening': 1.2,    # More exploration in opening
            'midgame': 0.8,    # Balanced approach
            'endgame': 0.5     # More deterministic in endgame
        }.get(phase, 0.8)
        
        # Adjust based on number of moves
        if num_moves <= 3:
            return base_temp * 0.7  # Less randomness with few options
        elif num_moves >= 15:
            return base_temp * 1.3  # More exploration with many options
        else:
            return base_temp

    def _simplified_tournament_selection(self, state, moves, base_simulations=None):
        """
        Simplified tournament for moderate complexity positions.
        Uses 2-round approach with optimized simulation counts.
        """
        S1, S2, _ = self.calculate_tournament_simulations(state, base_simulations)
        phase = self._detect_game_phase(state)
        k1 = min(4, len(moves))  # Top moves for round 2
        
        print(f"Simplified Tournament: {len(moves)} moves, S1={S1}, S2={S2}")
        
        # Round 1: Quick evaluation of all moves
        round1_results = []
        for move in moves:
            # Combine fast heuristic with limited simulations
            heuristic_score = self._calculate_comprehensive_move_score(state, move, phase)
            sim_score = self._run_sequential_simulations_for_move(state, move, S1)
            # Weight: 50% heuristic, 50% simulation for speed
            combined_score = 0.5 * heuristic_score + 0.5 * sim_score
            round1_results.append((move, combined_score, sim_score, heuristic_score))
        
        # Sort and select top k1 moves
        round1_results.sort(key=lambda x: x[1], reverse=True)
        round2_candidates = round1_results[:k1]
        
        print(f"Round 1 complete. Top {k1} moves advance to Round 2:")
        for i, (move, combined_score, sim_score, heur_score) in enumerate(round2_candidates):
            print(f"  {i+1}. Move {move}: combined={combined_score:.3f}")
        
        # Round 2: More intensive evaluation for top candidates
        final_results = []
        for move, _, prev_sim, prev_heur in round2_candidates:
            # Additional simulations
            additional_sim = self._run_sequential_simulations_for_move(state, move, S2)
            avg_sim = (prev_sim + additional_sim) / 2.0
            
            # Final score: 70% simulation, 30% heuristic
            final_score = 0.7 * avg_sim + 0.3 * prev_heur
            final_results.append((move, final_score))
        
        # Select best move
        final_results.sort(key=lambda x: x[1], reverse=True)
        best_move = final_results[0][0]
        
        print(f"Simplified Tournament Winner: {best_move}")
        return best_move
    
    def _get_move_with_optimized_tournament_layering(self, state, moves, base_simulations=None):
        """
        Optimized tournament layering for complex positions.
        Reduced computational overhead while maintaining quality.
        """
        S1, S2, S3 = self.calculate_tournament_simulations(state, base_simulations)
        k1 = min(6, len(moves))  # Top moves for round 2
        k2 = min(3, len(moves))  # Top moves for round 3
        
        print(f"Optimized Tournament: {len(moves)} moves, S1={S1}, S2={S2}, S3={S3}")
        
        # Pre-filter moves to remove obviously poor candidates
        phase = self._detect_game_phase(state)
        if len(moves) > 12:
            moves = self._quick_filter_moves(state, moves, phase, max_candidates=10)
            print(f"Pre-filtered to {len(moves)} candidates")
        
        # Round 1: Evaluate all moves with minimal simulations
        round1_results = []
        for move in moves:
            heuristic_score = self._calculate_comprehensive_move_score(state, move, phase)
            # Use smaller simulation count for first round
            sim_score = self._run_sequential_simulations_for_move(state, move, max(3, S1//2))
            combined_score = 0.6 * sim_score + 0.4 * heuristic_score
            round1_results.append((move, combined_score, sim_score, heuristic_score, max(3, S1//2)))
        
        # Sort and select top k1 moves
        round1_results.sort(key=lambda x: x[1], reverse=True)
        round2_candidates = round1_results[:k1]
        
        print(f"Round 1 complete. Top {k1} moves advance:")
        for i, (move, score, _, _, _) in enumerate(round2_candidates):
            print(f"  {i+1}. Move {move}: score={score:.3f}")
        
        # Round 2: More simulations for top candidates
        round2_results = []
        for move, _, prev_sim, prev_heur, prev_count in round2_candidates:
            additional_sim = self._run_sequential_simulations_for_move(state, move, S2)
            total_sim = (prev_sim * prev_count + additional_sim * S2) / (prev_count + S2)
            combined_score = 0.65 * total_sim + 0.35 * prev_heur
            round2_results.append((move, combined_score, total_sim, prev_heur, prev_count + S2))
        
        # Sort and select top k2 moves
        round2_results.sort(key=lambda x: x[1], reverse=True)
        round3_candidates = round2_results[:k2]
        
        print(f"Round 2 complete. Top {k2} moves advance:")
        for i, (move, score, _, _, _) in enumerate(round3_candidates):
            print(f"  {i+1}. Move {move}: score={score:.3f}")
        
        # Round 3: Final intensive evaluation
        final_results = []
        for move, _, prev_sim, prev_heur, prev_count in round3_candidates:
            final_sim = self._run_sequential_simulations_for_move(state, move, S3)
            total_sim = (prev_sim * prev_count + final_sim * S3) / (prev_count + S3)
            final_score = 0.75 * total_sim + 0.25 * prev_heur
            final_results.append((move, final_score))
            
            print(f"  {move}: final_score={final_score:.3f}")
        
        # Select best move
        final_results.sort(key=lambda x: x[1], reverse=True)
        best_move = final_results[0][0]
        best_score = final_results[0][1]
        
        print(f"Optimized Tournament Winner: {best_move} (score: {best_score:.3f})")
        return best_move