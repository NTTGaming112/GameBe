import random
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from .monte_carlo_node import MonteCarloNode
from .monte_carlo_base import MonteCarloBase
from app.ai.ataxx_state import Ataxx
from .constants import PLAYER_ONE, WIN_BONUS_EARLY, WIN_BONUS_FULL_BOARD, ADJACENT_POSITIONS, PHASE_ADAPTIVE_COEFFS, MCD_PHASE_WEIGHTS

class MonteCarloDomain(MonteCarloBase):
   
    def __init__(self, state, time_limit=None, **kwargs):
        super().__init__(state, **kwargs)
        self.time_limit = time_limit or self.max_time
        self.max_iterations = self.basic_simulations
        self.max_workers = 4
        self.heuristic_cache = {}
        self.simulation_cache = {}
        
        self.performance_stats = {
            'iterations_run': 0,
            'avg_iteration_time': 0.0,
            'total_nodes_expanded': 0,
            'parallel_iterations': 0
        }

    def get_move(self, time_limit=None):
        return self.get_mcts_move(time_limit)

    def get_mcts_move(self, time_limit=None):
        start_time = time.time()
        time_limit = self.time_limit if time_limit is None else time_limit

        root = MonteCarloNode(self.root_state, mcd_instance=self)
        
        if not self.root_state.get_all_possible_moves():
            return None

        if len(self.root_state.get_all_possible_moves()) == 1:
            return self.root_state.get_all_possible_moves()[0]

        iteration_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            while iteration_count < self.max_iterations and (time.time() - start_time < time_limit):
                batch_size = min(self.max_workers, self.max_iterations - iteration_count)
                for _ in range(batch_size):
                    future = executor.submit(self._run_iteration, root)
                    futures.append(future)
                
                for future in as_completed(futures):
                    if future.result():
                        iteration_count += 1
                        self.performance_stats['iterations_run'] += 1
                
                futures.clear()
                self.performance_stats['parallel_iterations'] += 1

                if root.children and iteration_count > 50:
                    max_visits = max(c.visits for c in root.children)
                    avg_visits = sum(c.visits for c in root.children) / len(root.children)
                    if max_visits > 2 * avg_visits:
                        print(f"Dừng sớm tại lần lặp {iteration_count}: tìm thấy nước đi vượt trội")
                        break

                if iteration_count % 50 == 0 and iteration_count > 0:
                    elapsed = time.time() - start_time
                    win_rate = root.ep / root.visits if root.visits > 0 else 0
                    print(f"  {iteration_count}/{self.max_iterations} mô phỏng, {elapsed:.1f}s, root_wr={win_rate:.3f}")

        iteration_time = time.time() - start_time
        self.performance_stats['avg_iteration_time'] = (
            (self.performance_stats['avg_iteration_time'] * 
             (self.performance_stats['iterations_run'] - 1) + iteration_time) /
            self.performance_stats['iterations_run']
        )
        self.performance_stats['total_nodes_expanded'] += len(root.children)

        if not root.children:
            valid_moves = self.root_state.get_all_possible_moves()
            return random.choice(valid_moves) if valid_moves else None
        
        return max(root.children, key=lambda c: c.visits).move

    def _run_iteration(self, root):
        state = Ataxx()
        state.player1_board = self.root_state.player1_board
        state.player2_board = self.root_state.player2_board
        state.balls = self.root_state.balls.copy()
        state.turn_player = self.root_state.turn_player
        node = root
        undo_stack = []
        
        while (not node.untried_moves or len(node.untried_moves) == 0) and node.children:
            node = node.select_child()
            if node.move:
                undo_stack.append(state.apply_move_with_undo(node.move))

        if node.untried_moves and len(node.untried_moves) > 0:
            expanded_node = node.expand()
            if expanded_node is None:
                for undo_info in reversed(undo_stack):
                    state.undo_move(undo_info)
                return False
            node = expanded_node
            if node.move:
                undo_stack.append(state.apply_move_with_undo(node.move))

        result = self._simulate(node.state, self.root_state.turn_player)
        
        for undo_info in reversed(undo_stack):
            state.undo_move(undo_info)

        node.backpropagate(result)
        
        return True

    def _simulate(self, state, player):
        return self._rollout_simulation(state, player)

    def _rollout_simulation(self, state, original_player):
        undo_stack = []
        depth = 0
        max_depth = 5
        
        while depth < max_depth and not state.is_game_over():
            moves = state.get_all_possible_moves()
            if not moves:
                state.toggle_player()
                moves = state.get_all_possible_moves()
                if not moves:
                    break
            
            move = self._select_move_with_heuristic(state, moves)
            undo_stack.append(state.apply_move_with_undo(move))
            state.toggle_player()
            depth += 1
        
        reward = self._evaluate_final_position(state, original_player)
        
        for undo_info in reversed(undo_stack):
            state.undo_move(undo_info)
        
        return reward

    def _select_move_with_heuristic(self, state, moves):
        # Lọc bỏ move None hoặc không hợp lệ
        valid_moves = [move for move in moves if move is not None and isinstance(move, tuple)]
        if not valid_moves:
            print("Warning: No valid moves found, returning random move")
            return random.choice(moves) if moves else None

        state_key = (state.player1_board, state.player2_board, state.turn_player)
        phase = self._detect_game_phase(state)
        weights = MCD_PHASE_WEIGHTS[phase]
        heuristic_scores = []
        
        def evaluate_move(move):
            cache_key = (state_key, move, phase)
            if cache_key in self.heuristic_cache:
                return self.heuristic_cache[cache_key]
            
            try:
                h = self._calculate_heuristic_score(state, move)
                t = self._calculate_tactical_score(state, move)
                s = self._calculate_strategic_score(state, move)
                sim = self._calculate_simulation_score(state, move, original_player=state.turn_player)
                combined_score = (
                    weights['alpha'] * h +
                    weights['beta'] * t +
                    weights['gamma'] * s +
                    weights['delta'] * sim
                )
            except Exception as e:
                print(f"Error evaluating move {move}: {e}")
                combined_score = 0.0  # Fallback score
            
            self.heuristic_cache[cache_key] = combined_score
            return combined_score

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_move = {executor.submit(evaluate_move, move): move for move in valid_moves}
            for future in as_completed(future_to_move):
                try:
                    score = future.result()
                    heuristic_scores.append((future_to_move[future], score))
                except Exception as e:
                    print(f"Parallel evaluation error for move {future_to_move[future]}: {e}")
                    heuristic_scores.append((future_to_move[future], 0.0))

        if not heuristic_scores:
            print("Warning: No scores computed, returning random valid move")
            return random.choice(valid_moves)

        # Sắp xếp theo score thay vì move để tránh lỗi so sánh
        sorted_scores = sorted(heuristic_scores, key=lambda x: x[1], reverse=True)
        moves, scores = zip(*sorted_scores)
        probabilities = self._softmax(list(scores))
        return random.choices(moves, weights=probabilities, k=1)[0]

    def _calculate_simulation_score(self, state, move, original_player):
        state_key = (state.player1_board, state.player2_board, state.turn_player)
        sim_cache_key = (state_key, move)
        if sim_cache_key in self.simulation_cache:
            return self.simulation_cache[sim_cache_key]
        
        temp_state = Ataxx()
        temp_state.player1_board = state.player1_board
        temp_state.player2_board = state.player2_board
        temp_state.balls = state.balls.copy()
        temp_state.turn_player = state.turn_player
        
        undo_info = temp_state.apply_move_with_undo(move)
        temp_state.toggle_player()
        
        sim_depth = 10
        num_sims = 3  # Giảm để tăng tốc
        total_score = 0.0
        
        for _ in range(num_sims):
            sim_state = Ataxx()
            sim_state.player1_board = temp_state.player1_board
            sim_state.player2_board = temp_state.player2_board
            sim_state.balls = temp_state.balls.copy()
            sim_state.turn_player = temp_state.turn_player
            reward = self._short_simulation(sim_state, original_player, sim_depth)
            total_score += reward
        
        temp_state.undo_move(undo_info)
        
        combined_sim = total_score / num_sims
        self.simulation_cache[sim_cache_key] = combined_sim
        if len(self.simulation_cache) > 10000:
            self.simulation_cache.pop(next(iter(self.simulation_cache)))
        
        return combined_sim

    def _short_simulation(self, state, original_player, max_depth):
        undo_stack = []
        depth = 0
        
        while depth < max_depth and not state.is_game_over():
            moves = state.get_all_possible_moves()
            if not moves:
                state.toggle_player()
                moves = state.get_all_possible_moves()
                if not moves:
                    break
            
            move = self._select_move_with_heuristic(state, moves)
            undo_stack.append(state.apply_move_with_undo(move))
            state.toggle_player()
            depth += 1
        
        reward = self._evaluate_final_position(state, original_player)
        
        for undo_info in reversed(undo_stack):
            state.undo_move(undo_info)
        
        return reward

    def _evaluate_final_position(self, state, player):
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
                empty_slots = 49 - total_pieces
                score += WIN_BONUS_FULL_BOARD if empty_slots == 0 else WIN_BONUS_EARLY
            elif winner == -player:
                total_pieces = num_own + num_opp
                empty_slots = 49 - total_pieces
                score -= WIN_BONUS_FULL_BOARD if empty_slots == 0 else WIN_BONUS_EARLY
            elif winner == 100:
                score = 0
                
        score = (score + 549) / 1098
        self._cache_eval(state, player, score)
        return score

    def get_structured_prior(self, node):
        moves = node.state.get_all_possible_moves()
        if not moves:
            return 0.0
        
        state_key = (node.state.player1_board, node.state.player2_board, node.state.turn_player)
        phase = self._detect_game_phase(node.state)
        weights = MCD_PHASE_WEIGHTS[phase]
        scores = []
        
        for move in moves:
            if move is None or not isinstance(move, tuple):
                continue
            cache_key = (state_key, move, phase)
            if cache_key in self.heuristic_cache:
                combined_score = self.heuristic_cache[cache_key]
            else:
                h = self._calculate_heuristic_score(node.state, move)
                t = self._calculate_tactical_score(node.state, move)
                s = self._calculate_strategic_score(node.state, move)
                sim = self._calculate_simulation_score(node.state, move, original_player=node.state.turn_player)
                combined_score = (
                    weights['alpha'] * h +
                    weights['beta'] * t +
                    weights['gamma'] * s +
                    weights['delta'] * sim
                )
                self.heuristic_cache[cache_key] = combined_score
            scores.append(combined_score)
        
        return max(scores) if scores else 0.0

    def _softmax(self, scores, temperature=1.0):
        if not scores:
            return []
        scaled_scores = [score / temperature for score in scores]
        max_score = max(scaled_scores)
        exp_scores = [math.exp(score - max_score) for score in scaled_scores]
        sum_exp = sum(exp_scores)
        if sum_exp == 0:
            return [1.0 / len(scores)] * len(scores)
        return [exp_score / sum_exp for exp_score in exp_scores]

    def _detect_game_phase(self, state):
        total_pieces = bin(state.player1_board | state.player2_board).count('1')
        if total_pieces < 8:
            return 'early'
        elif total_pieces < 20:
            return 'mid'
        return 'late'

    def _calculate_heuristic_score(self, state, move):
        from_pos, to_pos = move
        distance = 0 if from_pos is None else self._manhattan_distance(from_pos, to_pos)
        
        temp_state = Ataxx()
        temp_state.player1_board = state.player1_board
        temp_state.player2_board = state.player2_board
        temp_state.balls = state.balls.copy()
        temp_state.turn_player = state.turn_player
        original_enemy = self._count_enemy_pieces(temp_state, state.turn_player)
        temp_state.move_with_position(move)
        new_enemy = self._count_enemy_pieces(temp_state, state.turn_player)
        A = original_enemy - new_enemy
        
        B = self._count_friendly_neighbors(state, to_pos, state.turn_player)
        C = 1.0 if distance <= 1 else 0.0
        P = 1.0 if distance > 1 else 0
        
        phase = self._detect_game_phase(state)
        heuristic_coeffs = PHASE_ADAPTIVE_COEFFS['heuristic'][phase]
        h_raw = (heuristic_coeffs['s1'] * A + 
                 heuristic_coeffs['s2'] * B + 
                 heuristic_coeffs['s3'] * C - 
                 heuristic_coeffs['s4'] * P)
        return self._sigmoid_normalize(h_raw)

    def _calculate_tactical_score(self, state, move):
        from_pos, to_pos = move
        tactical_score = 0.5
        
        if self._is_corner_position(to_pos):
            tactical_score -= 0.4
        elif self._is_edge_position(to_pos):
            tactical_score -= 0.1
            
        if self._is_suicide_move(state, move):
            tactical_score -= 0.5
            
        captures = self._count_captures(state, move)
        tactical_score += captures * 0.15
        
        if self._is_center_position(to_pos):
            tactical_score += 0.2
            
        mobility_preserved = self._check_mobility_preservation(state, move)
        tactical_score += mobility_preserved * 0.1

        return self._sigmoid_normalize(tactical_score)

    def _calculate_strategic_score(self, state, move):
        strategic_score = 0.0
        
        tempo_value = self._evaluate_tempo_control(state, move)
        strategic_score += tempo_value * 0.3
        
        isolation_value = self._evaluate_enemy_isolation(state, move)
        strategic_score += isolation_value * 0.4
        
        mobility_reduction = self._evaluate_mobility_reduction(state, move)
        strategic_score += mobility_reduction * 0.3

        return self._sigmoid_normalize(strategic_score)

    def _manhattan_distance(self, pos1, pos2):
        if pos1 is None or pos2 is None:
            return 0
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
        return bin(state.player2_board if player == PLAYER_ONE else state.player1_board).count('1')

    def _count_friendly_neighbors(self, state, position, player):
        if position is None:
            return 0
        if isinstance(position, tuple):
            row, col = position
        else:
            row, col = divmod(position, 7)
        
        friendly_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 7 and 0 <= new_col < 7:
                    new_bit = new_row * 7 + new_col
                    board = state.player1_board if player == PLAYER_ONE else state.player2_board
                    if (board >> new_bit) & 1:
                        friendly_count += 1
        return friendly_count

    def _is_corner_position(self, position):
        if position is None:
            return False
        if isinstance(position, tuple):
            row, col = position
            return (row == 0 or row == 6) and (col == 0 or col == 6)
        corners = [0, 6, 42, 48]
        return position in corners

    def _is_edge_position(self, position):
        if position is None:
            return False
        if isinstance(position, tuple):
            row, col = position
        else:
            row, col = divmod(position, 7)
        return row == 0 or row == 6 or col == 0 or col == 6

    def _is_center_position(self, position):
        if position is None:
            return False
        if isinstance(position, tuple):
            row, col = position
        else:
            row, col = divmod(position, 7)
        return 2 <= row <= 4 and 2 <= col <= 4

    def _is_suicide_move(self, state, move):
        from_pos, to_pos = move
        temp_state = Ataxx()
        temp_state.player1_board = state.player1_board
        temp_state.player2_board = state.player2_board
        temp_state.balls = state.balls.copy()
        temp_state.turn_player = state.turn_player
        temp_state.move_with_position(move)
        if isinstance(to_pos, tuple):
            row, col = to_pos
        else:
            row, col = divmod(to_pos, 7)
        
        friendly_neighbors = 0
        for dr, dc in ADJACENT_POSITIONS:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 7 and 0 <= new_col < 7:
                new_pos = new_row * 7 + new_col
                board = temp_state.player1_board if state.turn_player == PLAYER_ONE else temp_state.player2_board
                if (board >> new_pos) & 1:
                    friendly_neighbors += 1
        return friendly_neighbors == 0

    def _count_captures(self, state, move):
        temp_state = Ataxx()
        temp_state.player1_board = state.player1_board
        temp_state.player2_board = state.player2_board
        temp_state.balls = state.balls.copy()
        temp_state.turn_player = state.turn_player
        original_enemy = self._count_enemy_pieces(temp_state, state.turn_player)
        temp_state.move_with_position(move)
        new_enemy = self._count_enemy_pieces(temp_state, state.turn_player)
        return original_enemy - new_enemy

    def _check_mobility_preservation(self, state, move):
        our_moves_before = len(state.get_all_possible_moves())
        temp_state = Ataxx()
        temp_state.player1_board = state.player1_board
        temp_state.player2_board = state.player2_board
        temp_state.balls = state.balls.copy()
        temp_state.turn_player = state.turn_player
        temp_state.move_with_position(move)
        temp_state.toggle_player()
        temp_state.toggle_player()
        our_moves_after = len(temp_state.get_all_possible_moves())
        if our_moves_before == 0:
            return 0.0
        mobility_ratio = our_moves_after / our_moves_before
        return 1.0 if mobility_ratio >= 1.0 else mobility_ratio


    def _evaluate_tempo_control(self, state, move):
        from_pos, to_pos = move
        tempo_score = 0.0
        if self._is_center_position(to_pos):
            tempo_score += 0.4
        temp_state = Ataxx()
        temp_state.player1_board = state.player1_board
        temp_state.player2_board = state.player2_board
        temp_state.balls = state.balls.copy()
        temp_state.turn_player = state.turn_player
        temp_state.move_with_position(move)
        if isinstance(to_pos, tuple):
            row, col = to_pos
        else:
            row, col = divmod(to_pos, 7)
        threat_count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 7 and 0 <= new_col < 7:
                    new_bit = new_row * 7 + new_col
                    enemy_board = state.player2_board if state.turn_player == PLAYER_ONE else state.player1_board
                    if (enemy_board >> new_bit) & 1:
                        threat_count += 1
        tempo_score += min(threat_count * 0.15, 0.3)
        captures = self._count_captures(state, move)
        if captures > 0:
            tempo_score += 0.2
        if from_pos is None:
            tempo_score += 0.1
        return min(tempo_score, 1.0)


    def _evaluate_enemy_isolation(self, state, move):
        from_pos, to_pos = move
        temp_state = Ataxx()
        temp_state.player1_board = state.player1_board
        temp_state.player2_board = state.player2_board
        temp_state.balls = state.balls.copy()
        temp_state.turn_player = state.turn_player
        temp_state.move_with_position(move)
        
        enemy_player = -state.turn_player if state.turn_player in [PLAYER_ONE, -PLAYER_ONE] else PLAYER_ONE
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
            return 1.0
        
        visited = set()
        components = 0
        def flood_fill(start_pos):
            stack = [start_pos]
            component_size = 0
            while stack:
                pos = stack.pop()
                if pos in visited:
                    continue
                visited.add(pos)
                component_size += 1
                row, col = pos
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        new_row, new_col = row + dr, col + dc
                        if (0 <= new_row < 7 and 0 <= new_col < 7 and 
                            (new_row, new_col) in enemy_positions and 
                            (new_row, new_col) not in visited):
                            stack.append((new_row, new_col))
            return component_size
        
        component_sizes = []
        for pos in enemy_positions:
            if pos not in visited:
                size = flood_fill(pos)
                component_sizes.append(size)
                components += 1
        
        if components == 0:
            return 1.0
        elif components == 1:
            return 0.2
        else:
            avg_component_size = sum(component_sizes) / len(component_sizes)
            isolation_score = min(components * 0.2, 0.8)
            isolation_score += max(0, (3 - avg_component_size) * 0.1)
            return min(isolation_score, 1.0)


    def _evaluate_mobility_reduction(self, state, move):
        temp_state = Ataxx()
        temp_state.player1_board = state.player1_board
        temp_state.player2_board = state.player2_board
        temp_state.balls = state.balls.copy()
        temp_state.turn_player = state.turn_player
        temp_state.toggle_player()
        enemy_moves_before = len(temp_state.get_all_possible_moves())
        temp_state.toggle_player()
        temp_state.move_with_position(move)
        temp_state.toggle_player()
        enemy_moves_after = len(temp_state.get_all_possible_moves())
        if enemy_moves_before == 0:
            return 1.0
        return (enemy_moves_before - enemy_moves_after) / enemy_moves_before


    def _sigmoid_normalize(self, x):
        return 1 / (1 + math.exp(-x))