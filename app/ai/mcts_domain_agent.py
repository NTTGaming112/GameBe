import numpy as np
from heuristics import evaluate, heuristic
from constants import DEFAULT_MCTS_DOMAIN_ITERATIONS

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_moves = state.get_legal_moves()
        self.heuristic_score = 0

class MCTSDomainAgent:
    def __init__(self, iterations=DEFAULT_MCTS_DOMAIN_ITERATIONS, tournament_params=None):
        self.iterations = iterations
        self.c = 1.41
        self.pb_c = 0.1
        self.domain_knowledge = evaluate
        
        self.tournament_params = tournament_params or {
            'round1': {'simulations': iterations, 'keep_top': 5},
            'round2': {'simulations': int(iterations * 1.5), 'keep_top': 3},
            'round3': {'simulations': iterations * 2}
        }
        
        self.heavy_playout_threshold = 0.7
        self.heavy_playout_depth = 10

    def ucb1(self, node, parent_visits):
        if node.visits == 0:
            return float('inf')
        
        pb_bonus = self.pb_c * node.heuristic_score / (node.visits + 1)
        exploitation = node.value / node.visits
        exploration = self.c * np.sqrt(np.log(parent_visits) / node.visits)
        
        return exploitation + exploration + pb_bonus

    def select(self, node):
        while node.untried_moves == [] and node.children:
            node = max(node.children, key=lambda c: self.ucb1(c, node.visits))
        return node

    def expand(self, node):
        if node.untried_moves:
            move = node.untried_moves.pop(0)
            new_state = node.state.copy()
            new_state.make_move(move)
            child = MCTSNode(new_state, parent=node, move=move)
            
            try:
                child.heuristic_score = heuristic(child.state, move, node.state.current_player)
            except (TypeError, ValueError):
                child.heuristic_score = self.domain_knowledge(child.state, node.state.current_player)
            
            node.children.append(child)
            return child
        return node

    def heavy_playout(self, state):
        sim_state = state.copy()
        player = state.current_player
        depth = 0
        
        while not sim_state.is_game_over() and depth < self.heavy_playout_depth:
            moves = sim_state.get_legal_moves()
            if not moves:
                break
                
            # Convert moves to list if needed
            moves_list = list(moves) if not isinstance(moves, list) else moves
            
            if np.random.random() < self.heavy_playout_threshold and len(moves_list) > 1:
                try:
                    scores = [heuristic(sim_state, move, player) for move in moves_list]
                    max_score = max(scores)
                    best_moves = [move for move, score in zip(moves_list, scores) if score == max_score]
                    move = best_moves[np.random.randint(len(best_moves))]
                except (TypeError, ValueError):
                    move = moves_list[np.random.randint(len(moves_list))]
            else:
                move = moves_list[np.random.randint(len(moves_list))]
                
            sim_state.make_move(move)
            depth += 1
            
        if sim_state.is_game_over():
            result = sim_state.get_winner()
            return 1 if result == player else 0 if result == 0 else -1
        return self.domain_knowledge(sim_state, player)

    def tournament_round(self, node, simulations, keep_top=None):
        while node.untried_moves:
            self.expand(node)
            
        for child in node.children:
            for _ in range(simulations):
                result = self.heavy_playout(child.state)
                self.backpropagate(child, result, node.state.current_player)
                
        if keep_top is not None and len(node.children) > keep_top:
            return sorted(node.children, key=lambda c: c.visits, reverse=True)[:keep_top]
        return node.children

    def backpropagate(self, node, result, root_player):
        while node:
            node.visits += 1
            if node.state.current_player == root_player:
                node.value += result
            else:
                node.value += 1 - result
            node = node.parent

    def get_move(self, state):
        if not state.get_legal_moves():
            return None
            
        root = MCTSNode(state)
        root_player = state.current_player
        
        params = self.tournament_params
        
        candidates = self.tournament_round(root, params['round1']['simulations'])
        
        if len(candidates) > params['round1']['keep_top']:
            candidates = self.tournament_round(root,
                                             params['round2']['simulations'],
                                             params['round1']['keep_top'])
        
        if len(candidates) > params['round2']['keep_top']:
            candidates = self.tournament_round(root,
                                             params['round3']['simulations'],
                                             params['round2']['keep_top'])
        
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.visits).move