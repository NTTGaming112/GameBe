#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
from .uct import Utc
from .ataxx_state import Ataxx
from copy import copy, deepcopy
import numpy as np
import sys
import logging
import math
import random

utc = Utc()

def argmax(arr):
    return arr.index(max(arr))

def gumbel_softmax(arr, temperature):
    arr = [v/temperature for v in arr]
    mx = max(arr)
    arr = [math.exp(v - mx) for v in arr]
    s = sum(arr)
    return [v/s for v in arr]

class MCTS:
    def __init__(self, state, root_turn, turn=1, act=None):
        self.state = deepcopy(state)
        self.turn = turn
        self.root_turn = root_turn
        self.successors = []
        self.valid_moves = [] if state.is_game_over() else state.get_all_possible_moves()
        self.wins = 0
        self.visits = 0
        self.act = act
        self.virtual_loss = 0
        
        # UCT fractional parameters
        self.c_base = 19652    # Base exploration constant
        self.c_init = 1.25     # Initial exploration constant
        self.gamma = 0.5       # Fractional power for UCT
        
        # Optimized weights for better performance
        self.piece_value = 1.0        # Base value for each piece
        self.center_value = 0.25      # Center control
        self.corner_value = 0.3       # Corner control
        self.edge_value = 0.15        # Edge control
        self.capture_value = 0.4      # Capture importance
        self.mobility_value = 0.03    # Mobility importance
        self.territory_value = 0.2    # Territory control

    def _get_uct_score(self, parent_visits, node_visits, node_wins):
        """Calculate UCT score with fractional power."""
        if node_visits == 0:
            return float('inf')
            
        # UCT fractional formula
        exploitation = node_wins / node_visits
        exploration = self.c_init * math.pow(parent_visits, self.gamma) / (1 + node_visits)
        
        return exploitation + exploration

    def select(self, c, expand_threshold=1):
        if self.act is not None and self.visits < expand_threshold:
            return self.rollout()

        if self.valid_moves:
            r = self.expand()
            self.visits += 1
            self.wins += r
            return r

        if self.state.is_game_over():
            r = self.wins / (self.visits + 1e-9)
            self.wins += r
            self.visits += 1
            return r

        fx = lambda x: x if self.turn == 1 else 1-x
        
        if not self.successors:
            return self.rollout()
            
        # Progressive widening with UCT fractional
        N = min(4, len(self.successors))
        sorted_successors = sorted(self.successors, 
                                 key=lambda s: self._get_uct_score(self.visits, s.visits, s.wins), 
                                 reverse=True)[:N]
        
        # Apply virtual loss
        for s in sorted_successors:
            s.virtual_loss += 1
            
        # Select best child using UCT fractional
        best_score = float('-inf')
        best_child = None
        
        for child in sorted_successors:
            score = self._get_uct_score(self.visits, child.visits, child.wins - child.virtual_loss)
            if score > best_score:
                best_score = score
                best_child = child
                
        r = best_child.select(c, expand_threshold)
        
        # Remove virtual loss
        for s in sorted_successors:
            s.virtual_loss -= 1
            
        self.wins += r
        self.visits += 1
        return r

    def rollout(self):
        sim_game = deepcopy(self.state)
        turn = self.root_turn * self.turn
        t = 0
        
        # Early termination if game is clearly won/lost
        if sim_game.balls[1] == 0 or sim_game.balls[-1] == 0:
            winner = sim_game.get_winner() * self.root_turn
            return 1 if winner > 0 else 0 if winner < 0 else 0.5
            
        while not sim_game.is_game_over() and t < 400:
            moves = sim_game.get_all_possible_moves()
            if not moves:
                break
                
            # Fast rollout with basic evaluation
            move_scores = []
            for move in moves:
                score = 0
                if move[0] == 'c':
                    # Copy moves are generally better
                    score += 0.1
                # Add basic position evaluation
                x, y = move[1] if move[0] == 'c' else move[2]
                center = 3
                dist = abs(x - center) + abs(y - center)
                score += (6 - dist) * 0.05
                move_scores.append((move, score))
                
            # Select move with highest score
            move = max(move_scores, key=lambda x: x[1])[0]
            sim_game.move_with_position(move)
            sim_game.toggle_player()
            t += 1

        # Final evaluation
        r = sim_game.get_winner() * self.root_turn
        if r != 0:
            r = 1 if r > 0 else 0
        else:
            eval_score = self._evaluate_position(sim_game)
            r = 0.5 + 0.5 * (1 if eval_score > 0 else -1 if eval_score < 0 else 0)
            
        self.wins += r
        self.visits += 1
        return r

    def _evaluate_position(self, game):
        """Fast position evaluation for rollouts."""
        score = 0
        player = game.current_player()
        opponent = -player
        
        # Basic piece count difference
        piece_diff = game.balls[player] - game.balls[opponent]
        score += piece_diff * self.piece_value
        
        # Basic territory control
        for i in range(7):
            for j in range(7):
                if game.board[i][j] == player:
                    # Distance from center
                    center = 3
                    dist = abs(i - center) + abs(j - center)
                    score += (6 - dist) * self.center_value
                    
                    # Corner control
                    if (i, j) in [(0,0), (0,6), (6,0), (6,6)]:
                        score += self.corner_value
                    # Edge control
                    elif i == 0 or i == 6 or j == 0 or j == 6:
                        score += self.edge_value
                        
        return score

    def _calculate_territory(self, game, player):
        """Calculate territory control for a player."""
        territory = 0
        for i in range(7):
            for j in range(7):
                if game.board[i][j] == player:
                    # Count adjacent empty squares as controlled territory
                    for dx, dy in [(-1,1), (0,1), (1,1), (-1,0), (1,0), (-1,-1), (0,-1), (1,-1)]:
                        nx, ny = i + dx, j + dy
                        if 0 <= nx < 7 and 0 <= ny < 7 and game.board[nx][ny] == 0:
                            territory += 1
        return territory

    def _get_opponent_moves(self, game, opponent):
        """Get all possible moves for the opponent."""
        # Temporarily switch to opponent's turn
        current_player = game.current_player()
        game.turn_player = opponent
        moves = game.get_all_possible_moves()
        # Switch back
        game.turn_player = current_player
        return moves

    def _counter_minimax(self, game, player):
        """Evaluate position for countering Minimax strategies."""
        score = 0
        opponent = -player
        
        # Minimax tends to focus on immediate captures
        # Counter by maintaining piece safety
        safe_pieces = 0
        for i in range(7):
            for j in range(7):
                if game.board[i][j] == player:
                    if self._is_piece_safe(game, i, j):
                        safe_pieces += 1
        score += safe_pieces * 0.1
        
        # Minimax likes to control center
        # Counter by creating multiple threats
        threats = self._count_threats(game, player)
        score += threats * 0.15
        
        # Minimax can be predictable in early game
        # Counter by maintaining flexibility
        if game.balls[player] + game.balls[opponent] < 15:
            flexible_moves = self._count_flexible_moves(game, player)
            score += flexible_moves * 0.1
            
        return score

    def _is_piece_safe(self, game, x, y):
        """Check if a piece is safe from immediate capture."""
        player = game.board[x][y]
        opponent = -player
        
        # Check adjacent squares
        for dx, dy in [(-1,1), (0,1), (1,1), (-1,0), (1,0), (-1,-1), (0,-1), (1,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 7 and 0 <= ny < 7:
                if game.board[nx][ny] == opponent:
                    return False
        return True

    def _count_threats(self, game, player):
        """Count number of threatening positions."""
        threats = 0
        for i in range(7):
            for j in range(7):
                if game.board[i][j] == 0:
                    # Check if this position threatens opponent pieces
                    for dx, dy in [(-1,1), (0,1), (1,1), (-1,0), (1,0), (-1,-1), (0,-1), (1,-1)]:
                        nx, ny = i + dx, j + dy
                        if 0 <= nx < 7 and 0 <= ny < 7:
                            if game.board[nx][ny] == -player:
                                threats += 1
                                break
        return threats

    def _count_flexible_moves(self, game, player):
        """Count number of flexible moves available."""
        moves = game.get_all_possible_moves()
        flexible_moves = 0
        
        for move in moves:
            if move[0] == 'c':
                # Copy moves are more flexible
                flexible_moves += 1
            else:
                # Jump moves that maintain options
                x, y = move[2]
                if self._maintains_options(game, x, y):
                    flexible_moves += 1
                    
        return flexible_moves

    def _maintains_options(self, game, x, y):
        """Check if a position maintains future move options."""
        options = 0
        for dx, dy in [(-1,1), (0,1), (1,1), (-1,0), (1,0), (-1,-1), (0,-1), (1,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 7 and 0 <= ny < 7:
                if game.board[nx][ny] == 0:
                    options += 1
        return options >= 3  # At least 3 future options

    def expand(self):
        if not self.valid_moves:
            return self.rollout()
            
        # Progressive widening: only expand top N moves
        N = min(3, len(self.valid_moves))
        moves_to_expand = self.valid_moves[:N]
        mv = moves_to_expand.pop(0)
        
        next_state = deepcopy(self.state)
        next_state.move_with_position(mv)
        next_state.toggle_player()
        new_node = MCTS(next_state, self.root_turn, -self.turn, mv)
        self.successors.append(new_node)
        return new_node.rollout()

class MctsAI:
    def __init__(self, c=2, num_sim=5000):
        self.name = 'MCTS'
        self.num_sim = num_sim
        self.c = c
        self.root = None
        self.expand_threshold = 1
        self.tree = {}  # Cache for tree nodes
        self.max_tree_size = 10000  # Limit tree size to prevent memory issues

    def _get_node_key(self, state):
        """Generate a unique key for a game state."""
        return hash(str(state.board) + str(state.balls) + str(state.turn_player))

    def _prune_tree(self):
        """Prune old nodes to maintain tree size limit."""
        if len(self.tree) > self.max_tree_size:
            # Remove least visited nodes
            sorted_nodes = sorted(self.tree.items(), key=lambda x: x[1].visits)
            nodes_to_remove = len(self.tree) - self.max_tree_size
            for key, _ in sorted_nodes[:nodes_to_remove]:
                del self.tree[key]

    def search(self, state, turn):
        # Generate key for current state
        state_key = self._get_node_key(state)
        
        # Try to find existing node
        if state_key in self.tree:
            self.root = self.tree[state_key]
        else:
            # Create new root node
            self.root = MCTS(state, turn)
            self.tree[state_key] = self.root

        # Run simulations
        for _ in range(self.num_sim):
            self.root.select(self.c, self.expand_threshold)
            
        # Prune tree if needed
        self._prune_tree()

        return [(t.visits, t.act, t.wins/t.visits) for t in self.root.successors]

    def _is_same_state(self, state1, state2):
        return (state1.board == state2.board and 
                state1.balls == state2.balls and 
                state1.turn_player == state2.turn_player)

    def play(self, state, turn, temperature=0, debug=False):
        act_visits = self.search(state, turn)
        if not act_visits:
            return None
            
        visits = [t[0] for t in act_visits]
        actions = [t[1] for t in act_visits]
        
        if temperature == 0:
            mx = max(visits)
            id = [i for i, v in enumerate(visits) if v == mx]
            act = random.choice(id)
            # Update root to the selected child
            if self.root and act < len(self.root.successors):
                next_state = deepcopy(self.root.state)
                next_state.move_with_position(actions[act])
                next_state.toggle_player()
                next_key = self._get_node_key(next_state)
                if next_key in self.tree:
                    self.root = self.tree[next_key]
                else:
                    self.root = None
        else:
            p = gumbel_softmax(visits, temperature)
            acc_p = [p[0]]
            for i in range(1, len(p)):
                acc_p.append(acc_p[i-1] + p[i])
            r = random.random()
            act = len(p) - 1
            while act > 0:
                if r >= acc_p[act-1]:
                    break
                act -= 1
            # Update root to the selected child
            if self.root and act < len(self.root.successors):
                next_state = deepcopy(self.root.state)
                next_state.move_with_position(actions[act])
                next_state.toggle_player()
                next_key = self._get_node_key(next_state)
                if next_key in self.tree:
                    self.root = self.tree[next_key]
                else:
                    self.root = None

        if debug:
            print(f"Move confidence: {act_visits[act][2]:.3f}")
        return actions[act]

class MonteCarloTreeSearch:
    def __init__(self, state, **kwargs):
        self.initial_state = state
        self.mcts_ai = MctsAI(
            c=kwargs.get('exploration_weight', 2),
            num_sim=kwargs.get('number_simulations', 200)
        )

    def get_play(self):
        return self.mcts_ai.play(
            self.initial_state,
            self.initial_state.current_player(),
            temperature=0
        )