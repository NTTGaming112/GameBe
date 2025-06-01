
import random
import time
import concurrent.futures
from collections import OrderedDict

from .monte_carlo_node import MonteCarloNode
from app.ai.constants import WIN_BONUS_EARLY, WIN_BONUS_FULL_BOARD, PLAYER_ONE, PLAYER_TWO
from app.ai.ataxx_state import Ataxx

class MonteCarloBase:
    def __init__(self, state, **kwargs):
        self.root_state = state
        self.basic_simulations = kwargs.get('basic_simulations', 300)
        self.exploration = kwargs.get('exploration', 1.414)
        self.max_time = kwargs.get('max_time', 1.0)
        self.time_limit = kwargs.get('time_limit', 50)
        self.eval_cache = OrderedDict()
        self.max_cache_size = 1000

    def calculate_simulations(self, state):
        total_pieces = state.balls[PLAYER_ONE] + state.balls[PLAYER_TWO]
        return int(self.basic_simulations * (1 + 0.1 * total_pieces))

    def get_play(self):
        
        return self.get_move()

    def get_move(self, time_limit=None):
        if time_limit is None:
            time_limit = self.time_limit
        start_time = time.time()
        root = MonteCarloNode(self.root_state, mcd_instance=self if hasattr(self, '_calculate_structured_move_score') else None)
        simulations = self.calculate_simulations(self.root_state)
        batch_size = min(100, simulations // 4)
        simulation_count = 0

        def run_simulation():
            state = Ataxx()
            state.player1_board = self.root_state.player1_board
            state.player2_board = self.root_state.player2_board
            state.balls = self.root_state.balls.copy()
            state.turn_player = self.root_state.turn_player
            state.moves = self.root_state.moves.copy()
            state.position_history = self.root_state.position_history.copy()
            node = root
            undo_stack = []
            
            # Selection and expansion
            while (not node.untried_moves or len(node.untried_moves) == 0) and node.children:
                node = node.select_child()
                if node.move:  # Only apply move if it's not None
                    undo_stack.append(state.apply_move_with_undo(node.move))
                    
            if node.untried_moves and len(node.untried_moves) > 0:
                expanded_node = node.expand()
                if expanded_node is None:
                    # Expansion failed - undo any moves we applied
                    for undo_info in reversed(undo_stack):
                        state.undo_move(undo_info)
                    return None, None
                node = expanded_node
                if node.move:  # Only apply move if it's not None
                    undo_stack.append(state.apply_move_with_undo(node.move))
                    
            result = self._simulate(state, state.current_player())
            
            # Undo all moves
            for undo_info in reversed(undo_stack):
                state.undo_move(undo_info)
            return node, result

        while simulation_count < simulations and (time_limit is None or time.time() - start_time < time_limit):
            for _ in range(min(batch_size, simulations - simulation_count)):
                node, result = run_simulation()
                if node is None:
                    continue
                while node:
                    node.update(result)
                    node = node.parent
                    result = 1 - result
                simulation_count += 1
                if time_limit and time.time() - start_time >= time_limit:
                    break

        if not root.children:
            # If no children were expanded, try to get any valid move from current state
            valid_moves = self.root_state.get_all_possible_moves()
            if valid_moves:
                return random.choice(valid_moves)
            return None
        return max(root.children, key=lambda c: c.visits).move

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
                empty_spaces = 49 - total_pieces
                score += WIN_BONUS_FULL_BOARD if empty_spaces == 0 else WIN_BONUS_EARLY
            elif winner == -player:
                total_pieces = num_own + num_opp
                empty_spaces = 49 - total_pieces
                score -= WIN_BONUS_FULL_BOARD if empty_spaces == 0 else WIN_BONUS_EARLY
            elif winner == 100:  # Draw
                score = 0
        score = (score + 549) / 1098
        self._cache_eval(state, player, score)
        return score

    def _simulate(self, state, player):
        undo_stack = []
        simulation_depth = 0
        max_simulation_depth = 20
        moves_buffer = []
        consecutive_passes = 0
        
        while not state.is_game_over() and simulation_depth < max_simulation_depth and consecutive_passes < 2:
            moves_buffer.clear()
            moves_buffer.extend(state.get_all_possible_moves())
            if not moves_buffer:
                # No valid moves, pass turn
                state.toggle_player()
                consecutive_passes += 1
                simulation_depth += 1
                continue
            
            consecutive_passes = 0  # Reset pass counter when valid move is made
            move = self._ultra_fast_move_selection(state, moves_buffer) if hasattr(self, '_ultra_fast_move_selection') and simulation_depth < 2 and len(moves_buffer) > 1 else random.choice(moves_buffer)
            undo_stack.append(state.apply_move_with_undo(move))
            simulation_depth += 1
            
        reward = self._evaluate_final_position(state, player)
        for undo_info in reversed(undo_stack):
            state.undo_move(undo_info)
        return reward

    def _cache_eval(self, state, player, score):
        state_key = (state.player1_board, state.player2_board, state.current_player(), player)
        self.eval_cache[state_key] = score
        if len(self.eval_cache) > self.max_cache_size:
            self.eval_cache.popitem(last=False)

    def _get_cached_eval(self, state, player):
        state_key = (state.player1_board, state.player2_board, state.current_player(), player)
        return self.eval_cache.get(state_key)