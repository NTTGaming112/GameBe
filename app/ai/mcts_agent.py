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
        self.untried_moves = list(state.get_legal_moves())  # Convert to list
        # Thêm hash để so sánh state nhanh hơn
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
        self.root = None  # Lưu trữ root node để tái sử dụng
        self.game_history = []  # Track game history
        self.max_tree_size = 10000  # Giới hạn kích thước tree
        
    def get_move(self, state):
        # Cập nhật game history
        self.game_history.append(state.copy())
        
        # Tìm kiếm node có thể tái sử dụng
        reused_root = self._find_reusable_subtree(state)
        
        if reused_root:
            print(f"♻️ Reusing tree with {reused_root.visits} visits")
            root = reused_root
            # Prune tree nếu quá lớn
            self._prune_tree(root)
        else:
            print("🌱 Creating new tree")
            root = MCTSNode(state)

        root_player = state.current_player
        
        # MCTS iterations
        for iteration in range(self.iterations):
            node = root
            
            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
            
            # Expansion
            if node.untried_moves:
                node = node.expand()
            
            # Simulation
            result = self._simulate(node.state, root_player)
            
            # Backpropagation
            self._backpropagate(node, result, root_player)
            
            # Early stopping nếu có move rõ ràng tốt nhất
            if iteration > 50 and iteration % 50 == 0:
                if self._should_stop_early(root):
                    print(f"⏹️ Early stop at iteration {iteration}")
                    break

        # Chọn move tốt nhất
        if not root.children:
            # Fallback nếu không có children
            moves = state.get_legal_moves()
            return list(moves)[0] if moves else None
        
        # Chọn child có visits cao nhất (robust choice)
        best_child = max(root.children, key=lambda c: c.visits)
        
        # Cập nhật root cho lần sau
        self.root = best_child
        self.root.parent = None  # Detach từ parent cũ
        
        print(f"🎯 Selected move: {best_child.move}, visits: {best_child.visits}, value: {best_child.value/best_child.visits:.3f}")
        
        return best_child.move
    
    def _simulate(self, state, root_player):
        """Simulation trả về kết quả theo perspective của root_player"""
        sim_state = state.copy()
        depth = 0
        max_depth = 100
        
        while not sim_state.is_game_over() and depth < max_depth:
            moves = sim_state.get_legal_moves()
            if not moves:
                break
            sim_state.make_move(random.choice(list(moves)))
            depth += 1
        
        # Đánh giá theo root_player và normalize về [-1, 1]
        if sim_state.is_game_over():
            winner = sim_state.get_winner()
            if winner == root_player:
                return 1.0
            elif winner == -root_player:
                return -1.0
            else:
                return 0.0  # Draw
        
        # Convert evaluate result từ [0,1] về [-1,1]
        eval_score = evaluate(sim_state, root_player)
        return 2 * eval_score - 1  # [0,1] -> [-1,1]
    
    def _backpropagate(self, node, result, root_player):
        """Backpropagation với perspective switching"""
        current_result = result
        
        while node:
            node.visits += 1
            
            # Đơn giản hóa: luôn cộng result theo perspective của node
            # Nếu node.state.current_player khác root_player thì flip result
            if node.state.current_player == root_player:
                node.value += current_result
            else:
                node.value -= current_result
            
            node = node.parent
    
    def _find_reusable_subtree(self, target_state):
        """Tìm node có thể tái sử dụng từ cây cũ"""
        if not self.root:
            return None
        
        # BFS search trong tree
        queue = [self.root]
        visited = set()
        search_limit = 200  # Giới hạn search để tránh chậm
        
        while queue and len(visited) < search_limit:
            node = queue.pop(0)
            
            if id(node) in visited:
                continue
            visited.add(id(node))
            
            if node.state_equals(target_state):
                # Detach node từ parent
                node.parent = None
                return node
            
            # Thêm children vào queue
            queue.extend(node.children)
        
        return None
    
    def _prune_tree(self, root):
        """Cắt tỉa tree để tiết kiệm memory"""
        def count_nodes(node):
            count = 1
            for child in node.children:
                count += count_nodes(child)
            return count
        
        total_nodes = count_nodes(root)
        
        if total_nodes > self.max_tree_size:
            print(f"🌲 Pruning tree: {total_nodes} -> ", end="")
            self._prune_recursive(root, max_depth=8)  # Limit depth
            print(f"{count_nodes(root)} nodes")
    
    def _prune_recursive(self, node, depth=0, max_depth=8):
        """Recursive pruning"""
        if depth >= max_depth:
            node.children = []
            return
        
        if node.children:
            # Giữ lại top children có visits cao
            node.children.sort(key=lambda c: c.visits, reverse=True)
            keep_count = min(len(node.children), 5)  # Keep top 5
            node.children = node.children[:keep_count]
            
            # Continue pruning children
            for child in node.children:
                self._prune_recursive(child, depth + 1, max_depth)
    
    def _should_stop_early(self, root):
        """Kiểm tra có nên dừng sớm không"""
        if len(root.children) < 2:
            return False
        
        # Sắp xếp children theo visits
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        
        # Nếu move tốt nhất có visits gấp đôi move thứ 2
        if len(sorted_children) >= 2:
            best_visits = sorted_children[0].visits
            second_visits = sorted_children[1].visits
            
            if best_visits > second_visits * 2 and best_visits > 100:
                return True
        
        return False
    
    def get_tree_stats(self):
        """Lấy thống kê về tree hiện tại"""
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