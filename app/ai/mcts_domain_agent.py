import numpy as np
from ataxx_state import AtaxxState
from heuristics import evaluate

class MCTSDomainAgent:
    def __init__(self, iterations=1000):
        self.iterations = iterations
        self.c = 1.41
        self.domain_knowledge = evaluate

    def ucb1(self, node, parent_visits):
        if node['visits'] == 0:
            return float('inf')
        return (node['value'] / node['visits']) + self.c * np.sqrt(np.log(parent_visits) / node['visits'])

    def simulate(self, state):
        current = state.copy()
        while not current.is_game_over():
            moves = current.get_legal_moves()
            if not moves:
                break
            move = moves[np.random.randint(len(moves))]
            current.make_move(move)
        return self.domain_knowledge(current, 1)

    def tournament_mcts(self, state):
        root = {'move': None, 'value': 0, 'visits': 0, 'children': {}}
        legal_moves = state.get_legal_moves()
        
        # Round 1: Distribute S1 rollouts equally
        S1 = self.iterations // 5
        for move in legal_moves:
            root['children'][move] = {'move': move, 'value': 0, 'visits': 0, 'children': {}}
            for _ in range(S1 // len(legal_moves) if len(legal_moves) > 0 else 0):
                new_state = state.copy()
                new_state.make_move(move)
                value = self.simulate(new_state)
                root['children'][move]['value'] += value
                root['children'][move]['visits'] += 1
                root['value'] += value
                root['visits'] += 1

        # Round 2: Keep top k1=5 moves, assign S2 rollouts
        k1 = 5
        S2 = self.iterations // 2
        move_scores = [(move, node['value'] / node['visits'] if node['visits'] > 0 else 0) 
                       for move, node in root['children'].items()]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        top_k1_moves = [move for move, _ in move_scores[:k1]]
        
        for move in top_k1_moves:
            for _ in range(S2 // k1 if k1 > 0 else 0):
                new_state = state.copy()
                new_state.make_move(move)
                value = self.simulate(new_state)
                root['children'][move]['value'] += value
                root['children'][move]['visits'] += 1
                root['value'] += value
                root['visits'] += 1

        # Round 3: Keep top k2=3 moves, assign S3 rollouts
        k2 = 3
        S3 = self.iterations - S1 - S2
        move_scores = [(move, node['value'] / node['visits'] if node['visits'] > 0 else 0) 
                       for move, node in root['children'].items() if move in top_k1_moves]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        top_k2_moves = [move for move, _ in move_scores[:k2]]
        
        for move in top_k2_moves:
            for _ in range(S3 // k2 if k2 > 0 else 0):
                new_state = state.copy()
                new_state.make_move(move)
                value = self.simulate(new_state)
                root['children'][move]['value'] += value
                root['children'][move]['visits'] += 1
                root['value'] += value
                root['visits'] += 1

        # Select the best move
        best_move = max(root['children'].items(), 
                        key=lambda x: x[1]['value'] / x[1]['visits'] if x[1]['visits'] > 0 else 0)[0]
        return best_move

    def get_move(self, state):
        return self.tournament_mcts(state)