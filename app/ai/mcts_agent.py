import random
import math
import numpy as np
from heuristics import evaluate
from constants import DEFAULT_MCTS_ITERATIONS

class MCTSNode:
    def __init__(self, state, move=None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.uct_c = 1.41  
        self.untried_moves = list(state.get_legal_moves())  
        self._state_hash = self._compute_state_hash()
    
    def _compute_state_hash(self):
        """Tính hash cho state để so sánh nhanh"""
        return hash((self.state.board.tobytes(), self.state.current_player))
    
    def state_equals(self, other_state):
        """So sánh state với state khác"""
        return (self.state.current_player == other_state.current_player and 
                np.array_equal(self.state.board, other_state.board))

    def select_child(self):
        def ucb1_value(c):
            if c.visits == 0:
                return float('inf')
            exploitation = c.value / c.visits
            exploration = self.uct_c * math.sqrt(math.log(self.visits) / c.visits)
            return exploitation + exploration
        
        return max(self.children, key=ucb1_value)

    def expand(self):
        if not self.untried_moves:
            return self
        
        move = self.untried_moves.pop(0)
        new_state = self.state.copy()
        new_state.make_move(move)
        child = MCTSNode(new_state, move, self)
        self.children.append(child)
        return child

class MCTSAgent:
    def __init__(self, iterations=DEFAULT_MCTS_ITERATIONS):
        self.iterations = iterations
        self.root = None  
        self.game_history = []  
        self.max_tree_size = 10000  

    def get_move(self, state):
        """Get the best move using MCTS."""

        if not state.get_legal_moves():
            return None
        
        self.game_history.append(state.copy())
        reused_root = self._find_reusable_subtree(state)
        
        if reused_root:
            print(f"♻️ Reusing tree with {reused_root.visits} visits")
            root = reused_root
            self._prune_tree(root)

        else:
            print("🌱 Creating new tree")
            root = MCTSNode(state)

        root_player = state.current_player
        
        for iteration in range(self.iterations):
            node = root
            
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
            
            if node.untried_moves:
                node = node.expand()
            
            result = self._simulate(node.state, root_player)
            
            self._backpropagate(node, result, root_player)
            
            if iteration > 50 and iteration % 50 == 0:
                if self._should_stop_early(root):
                    print(f"⏹️ Early stop at iteration {iteration}")
                    break

        if not root.children:
            moves = state.get_legal_moves()
            return list(moves)[0] if moves else None
        
        best_child = max(root.children, key=lambda c: c.visits)
        
        self.root = best_child
        self.root.parent = None  

        print(f"🎯 Selected move: {best_child.move}, visits: {best_child.visits}, value: {best_child.value/best_child.visits:.3f}")
        
        return best_child.move
    
    def _simulate(self, state, root_player):
        """Simulate a random game from the given state to estimate the value of the state."""
        sim_state = state.copy()
        depth = 0
        max_depth = 100
        consecutive_passes = 0
        
        while not sim_state.is_game_over() and depth < max_depth:
            moves = sim_state.get_legal_moves()
            if not moves:
                sim_state.current_player = -sim_state.current_player
                consecutive_passes += 1

                if consecutive_passes >= 2:
                    break
                
            else:
                consecutive_passes = 0
                sim_state.make_move(random.choice(list(moves)))
                
            depth += 1

        eval_score = evaluate(sim_state, root_player)
        return eval_score  
    
    def _backpropagate(self, node, result, root_player):
        while node:
            node.visits += 1
            
            if node.state.current_player == root_player:
                node.value += result
            else:
                node.value += (1-result)  

            node = node.parent
    
    def _find_reusable_subtree(self, target_state):
        """Find a reusable subtree in the MCTS tree that matches the target state."""
        if not self.root:
            return None
        
        queue = [self.root]
        visited = set()
        search_limit = 200  
        
        while queue and len(visited) < search_limit:
            node = queue.pop(0)
            
            if id(node) in visited:
                continue
            visited.add(id(node))
            
            if node.state_equals(target_state):
                node.parent = None
                return node
            
            queue.extend(node.children)
        
        return None
    
    def _prune_tree(self, root):
        """Prune the MCTS tree if it exceeds max_tree_size."""
        def count_nodes(node):
            count = 1
            for child in node.children:
                count += count_nodes(child)
            return count
        
        total_nodes = count_nodes(root)
        
        if total_nodes > self.max_tree_size:
            print(f"🌲 Pruning tree: {total_nodes} -> ", end="")
            self._prune_recursive(root, max_depth=8)  
            print(f"{count_nodes(root)} nodes")
    
    def _prune_recursive(self, node, depth=0, max_depth=8):
        """Recursive pruning"""
        if depth >= max_depth:
            node.children = []
            return
        
        if node.children:
            node.children.sort(key=lambda c: c.visits, reverse=True)
            keep_count = min(len(node.children), 5)  
            node.children = node.children[:keep_count]
            
            for child in node.children:
                self._prune_recursive(child, depth + 1, max_depth)
    
    def _should_stop_early(self, root):
        """Check stop condition for early termination"""
        if len(root.children) < 2:
            return False
        
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        
        if len(sorted_children) >= 2:
            best_visits = sorted_children[0].visits
            second_visits = sorted_children[1].visits
            
            if best_visits > second_visits * 2 and best_visits > 100:
                return True
        
        return False
    
    def get_tree_stats(self):
        """Get statistics about the MCTS tree."""
        if not self.root:
            return {"nodes": 0, "depth": 0, "total_visits": 0}
        
        def analyze_tree(node, depth=0):
            nodes = 1
            max_depth = depth
            total_visits = node.visits
            
            for child in node.children:
                child_nodes, child_depth, child_visits = analyze_tree(child, depth + 1)
                nodes += child_nodes
                max_depth = max(max_depth, child_depth)
                total_visits += child_visits
            
            return nodes, max_depth, total_visits
        
        nodes, depth, visits = analyze_tree(self.root)
        return {
            "nodes": nodes,
            "max_depth": depth,
            "total_visits": visits,
            "root_visits": self.root.visits
        }
    
    def reset_tree(self):
        """Reset tree manually"""
        self.root = None
        self.game_history = []
        print("🗑️ Tree reset")