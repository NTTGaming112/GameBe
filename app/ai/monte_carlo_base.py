
import random
import time
from collections import OrderedDict
import threading

from .monte_carlo_node import MonteCarloNode
from .constants import WIN_BONUS_EARLY, WIN_BONUS_FULL_BOARD, PLAYER_ONE, PLAYER_TWO
from .ataxx_state import Ataxx

class MonteCarloBase:
    def __init__(self, state, **kwargs):
        self.root_state = state
        self.basic_simulations = kwargs.get('basic_simulations', 300)
        self.exploration = kwargs.get('exploration', 1.414)
        self.max_time = kwargs.get('max_time', 1.0)
        self.time_limit = kwargs.get('time_limit', 50)
        self.eval_cache = OrderedDict()
        self.max_cache_size = 1000
        self.cache_lock = threading.Lock()

    def get_play(self):
        
        return self.get_move()

    def get_move(self, time_limit=None):
        if time_limit is None:
            time_limit = self.time_limit
        start_time = time.time()

        root = MonteCarloNode(self.root_state, mcd_instance=self)
        simulations = self.basic_simulations
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
            
            # Selection
            while (not node.untried_moves or len(node.untried_moves) == 0) and node.children:
                node = node.select_child()
                if node.move:  # Only apply move if it's not None
                    undo_stack.append(state.apply_move_with_undo(node.move))

            # Expansion
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

            # Simulation       
            result = self._simulate(state, self.root_state.turn_player)
            
            # Undo all moves
            for undo_info in reversed(undo_stack):
                state.undo_move(undo_info)

            # Backpropagation
            node.backpropagate(result)
            
            return True
        
        # Main simulation loop
        while simulation_count < simulations and (time_limit is None or time.time() - start_time < time_limit):
            if run_simulation():
                simulation_count += 1
                
            # Progress logging
            if simulation_count % 100 == 0 and simulation_count > 0:
                elapsed = time.time() - start_time
                win_rate = root.ep / root.visits if root.visits > 0 else 0
                print(f"  {simulation_count}/{simulations} sims, {elapsed:.1f}s, root_wr={win_rate:.3f}")
                
            if time_limit and time.time() - start_time >= time_limit:
                break
        
        # Final move selection
        if not root.children:
            # If no children were expanded, try to get any valid move from current state
            valid_moves = self.root_state.get_all_possible_moves()
            if not valid_moves:
                # No valid moves, return None
                return None
            # If there are valid moves, return one randomly
            return random.choice(valid_moves)
        
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
        max_simulation_depth = 50
        moves_buffer = []
        
        while not state.is_game_over() and simulation_depth < max_simulation_depth:
            moves_buffer.clear()
            moves_buffer.extend(state.get_all_possible_moves())
            if not moves_buffer:
                # No valid moves, pass turn
                state.toggle_player()
                simulation_depth += 1
                continue

            move = random.choice(moves_buffer)
            undo_stack.append(state.apply_move_with_undo(move))
            simulation_depth += 1
            
        reward = self._evaluate_final_position(state, player)
        for undo_info in reversed(undo_stack):
            state.undo_move(undo_info)
        return reward

    def _cache_eval(self, state, player, score):
        """Thread-safe cache evaluation with proper OrderedDict usage"""
        state_key = (state.player1_board, state.player2_board, state.current_player(), player)
        
        with self.cache_lock:
            self.eval_cache[state_key] = score
            # Fix: OrderedDict.popitem() syntax
            if len(self.eval_cache) > self.max_cache_size:
                self.eval_cache.popitem(last=False)  # Remove oldest item

    def _get_cached_eval(self, state, player):
        """Thread-safe get cached evaluation"""
        state_key = (state.player1_board, state.player2_board, state.current_player(), player)
        with self.cache_lock:
            return self.eval_cache.get(state_key)