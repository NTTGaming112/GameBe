import numpy as np
from heuristics import evaluate, heuristic, sigmoid
from constants import DEFAULT_MCTS_DOMAIN_ITERATIONS

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = list(state.get_legal_moves())
        self.heuristic_score = 0.5  
        self._state_hash = self._compute_state_hash()
    
    def _compute_state_hash(self):
        """Tính hash cho state để so sánh nhanh"""
        return hash((self.state.board.tobytes(), self.state.current_player))
    
    def state_equals(self, other_state):
        """So sánh state với state khác"""
        return (self.state.current_player == other_state.current_player and 
                np.array_equal(self.state.board, other_state.board))

class MCTSDomainAgent:
    def __init__(self, iterations=DEFAULT_MCTS_DOMAIN_ITERATIONS, tournament_params=None):
        self.iterations = iterations
        self.c = 1.41
        self.pb_c = 0.1
        self.domain_knowledge = evaluate
        self.heuristic_scale = 5.0
        self.evaluation_scale = 2.0

        self.tournament_params = tournament_params or {
            'round1': {'simulations': iterations, 'keep_top': 5},
            'round2': {'simulations': int(iterations * 1.5), 'keep_top': 3},
            'round3': {'simulations': iterations * 2}
        }
        
        self.heavy_playout_threshold = 0.7
        self.heavy_playout_depth = 50
        
        self.root = None
        self.previous_states = []
        self.max_tree_depth = 50

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
                raw_heuristic = heuristic(move, node.state, node.state.current_player)
                child.heuristic_score = sigmoid(raw_heuristic, self.heuristic_scale)
            except (TypeError, ValueError):
                child.heuristic_score = self.domain_knowledge(child.state, node.state.current_player)
            
            node.children.append(child)
            return child
        return node

    def find_reusable_subtree(self, target_state):
        """Find a reusable subtree in the MCTS tree that matches the target state."""
        if not self.root:
            return None
        
        queue = [self.root]
        visited = set()
        
        while queue:
            node = queue.pop(0)
            
            if id(node) in visited:
                continue
            visited.add(id(node))
            
            if node.state_equals(target_state):
                node.parent = None
                return node
            
            if len(visited) < 100:
                queue.extend(node.children)
        
        return None

    def prune_tree(self, node, max_depth=None):
        """Prune the MCTS tree if it exceeds max_tree_size."""
        if max_depth is None:
            max_depth = self.max_tree_depth
        
        def _prune_recursive(current_node, depth):
            if depth >= max_depth:
                current_node.children = []
                return
            
            if current_node.children:
                current_node.children.sort(key=lambda c: c.visits, reverse=True)
                keep_count = min(len(current_node.children), 10)
                current_node.children = current_node.children[:keep_count]
                
                for child in current_node.children:
                    _prune_recursive(child, depth + 1)
        
        _prune_recursive(node, 0)

    def heavy_playout(self, state):
        """Heavy playout with heuristic-based move selection"""
        sim_state = state.copy()
        original_player = state.current_player
        depth = 0
        
        while not sim_state.is_game_over() and depth < self.heavy_playout_depth:
            moves = sim_state.get_legal_moves()
            if not moves:
                break
                
            moves_list = list(moves) if not isinstance(moves, list) else moves
            
            if np.random.random() < self.heavy_playout_threshold and len(moves_list) > 1:
                try:
                    scores = [heuristic(move, sim_state, sim_state.current_player) for move in moves_list]
                    sigmoid_scores = [sigmoid(score, self.heuristic_scale) for score in scores]
                    
                    exp_scores = np.exp(np.array(sigmoid_scores) * 2)  
                    probs = exp_scores / np.sum(exp_scores)
                    move_idx = np.random.choice(len(moves_list), p=probs)
                    move = moves_list[move_idx]
                except (TypeError, ValueError, IndexError):
                    move = moves_list[np.random.randint(len(moves_list))]
            else:
                move = moves_list[np.random.randint(len(moves_list))]
                
            sim_state.make_move(move)
            depth += 1
            
        return self.domain_knowledge(sim_state, original_player)

    def tournament_round(self, node, simulations, keep_top=None):
        while node.untried_moves:
            self.expand(node)
            
        for child in node.children:
            sims_per_child = simulations // len(node.children) if node.children else simulations
            for _ in range(sims_per_child):
                result = self.heavy_playout(child.state)
                self.backpropagate(child, result, node.state.current_player)
                
        if keep_top is not None and len(node.children) > keep_top:
            return sorted(node.children, key=lambda c: c.visits, reverse=True)[:keep_top]
        return node.children

    def backpropagate(self, node, result, root_player):
        """Backpropagation với result từ evaluate() [0,1]"""
        while node:
            node.visits += 1
            
            if node.state.current_player == root_player:
                node.value += result
            else:
                node.value += (1.0 - result)
            
            node = node.parent

    def update_game_history(self, state):
        """Cập nhật lịch sử game để track states"""
        self.previous_states.append(state.copy())
        if len(self.previous_states) > 20:
            self.previous_states.pop(0)

    def get_move(self, state):
        if not state.get_legal_moves():
            return None
        
        self.update_game_history(state)
        
        reused_root = self.find_reusable_subtree(state)
        
        if reused_root:
            print(f"🌲 Reusing tree with {reused_root.visits} visits")
            root = reused_root
            self.prune_tree(root)
        else:
            print("🌱 Creating new tree")
            root = MCTSNode(state)
        
        root_player = state.current_player
        params = self.tournament_params
        
        candidates = self.tournament_round(root, params['round1']['simulations'])
        
        if len(candidates) > params['round1']['keep_top']:
            top_candidates = sorted(candidates, key=lambda c: c.visits, reverse=True)[:params['round1']['keep_top']]
            for candidate in top_candidates:
                sims_per_candidate = params['round2']['simulations'] // len(top_candidates)
                for _ in range(sims_per_candidate):
                    result = self.heavy_playout(candidate.state)
                    self.backpropagate(candidate, result, root_player)
            candidates = top_candidates
        
        if len(candidates) > params['round2']['keep_top']:
            top_candidates = sorted(candidates, key=lambda c: c.visits, reverse=True)[:params['round2']['keep_top']]
            for candidate in top_candidates:
                sims_per_candidate = params['round3']['simulations'] // len(top_candidates)
                for _ in range(sims_per_candidate):
                    result = self.heavy_playout(candidate.state)
                    self.backpropagate(candidate, result, root_player)
            candidates = top_candidates
        
        if not candidates:
            return None
        
        best_child = max(candidates, key=lambda c: c.value / c.visits if c.visits > 0 else 0)
        
        self.root = best_child
        self.root.parent = None
        
        win_rate = best_child.value / best_child.visits if best_child.visits > 0 else 0
        print(f"🎯 Selected move with {best_child.visits} visits, win rate: {win_rate:.3f}")
        
        return best_child.move

    def reset_tree(self):
        """Reset tree manually if needed"""
        self.root = None
        self.previous_states = []
        print("🗑️ Tree reset")

    def get_tree_stats(self):
        """Get statistics about current tree"""
        if not self.root:
            return {"nodes": 0, "depth": 0, "total_visits": 0, "win_rate": 0}
        
        def count_nodes(node, depth=0):
            count = 1
            max_depth = depth
            total_visits = node.visits
            
            for child in node.children:
                child_count, child_depth, child_visits = count_nodes(child, depth + 1)
                count += child_count
                max_depth = max(max_depth, child_depth)
                total_visits += child_visits
            
            return count, max_depth, total_visits
        
        nodes, depth, visits = count_nodes(self.root)
        root_win_rate = self.root.value / self.root.visits if self.root.visits > 0 else 0
        
        return {
            "nodes": nodes,
            "depth": depth, 
            "total_visits": visits,
            "root_visits": self.root.visits,
            "root_win_rate": root_win_rate
        }