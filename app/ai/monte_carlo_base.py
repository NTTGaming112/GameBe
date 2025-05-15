#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import math
import time
from copy import deepcopy

class MonteCarloNode:
    """Lớp nút cơ bản cho cây Monte Carlo."""
    def __init__(self, state, parent=None, move=None):
        self.state = deepcopy(state)
        self.parent = parent
        self.move = move  # Nước đi dẫn đến trạng thái này
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_all_possible_moves() if not state.is_game_over() else []
        
    def uct_value(self, c=1.414):
        """Tính giá trị UCT (Upper Confidence Bound applied to Trees)."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_child(self):
        """Chọn nút con có giá trị UCT cao nhất."""
        return max(self.children, key=lambda c: c.uct_value())
    
    def expand(self):
        """Mở rộng cây tìm kiếm bằng cách tạo nút con mới."""
        if not self.untried_moves:
            return None
        
        move = self.untried_moves.pop(0)
        next_state = deepcopy(self.state)
        next_state.move_with_position(move)
        next_state.toggle_player()
        
        child = MonteCarloNode(next_state, self, move)
        self.children.append(child)
        return child
    
    def update(self, result):
        """Cập nhật số lần thắng và số lần thăm."""
        self.visits += 1
        self.wins += result

class MonteCarloBase:
    """Monte Carlo cơ bản - rollout ngẫu nhiên, không kiến thức."""
    def __init__(self, state, **kwargs):
        self.root_state = state
        self.basic_simulations = kwargs.get('basic_simulations', 300)
        self.exploration = kwargs.get('exploration', 1.414)
        self.max_time = kwargs.get('max_time', 1.0)
        self.tournament_sizes = kwargs.get('tournament_sizes', [600, 600, 300])
        self.use_simulation_formula = kwargs.get('use_simulation_formula', True)
        
    def calculate_simulations(self, state):
        """Tính số lượng mô phỏng dựa trên công thức Stotal = Sbasic * (1 + 0.1 * nfilled).
        
        Formula tăng số rollout khi bàn sắp kín.
        Với Sbasic = {300, 600, 1200} tùy theo cấu hình.
        
        Ví dụ:
        - Với Sbasic=300, bàn trống: 300 simulations
        - Với Sbasic=300, bàn đầy 1/2: 300 * (1 + 0.1*24.5) ≈ 345 simulations
        - Với Sbasic=300, bàn đầy: 300 * (1 + 0.1*49) = 447 simulations
        """
        # nfilled là tổng số quân cờ trên bàn
        total_pieces = state.balls[1] + state.balls[-1]
        
        # Nếu dùng công thức, áp dụng Stotal = Sbasic * (1 + 0.1 * nfilled)
        if self.use_simulation_formula:
            return int(self.basic_simulations * (1 + 0.1 * total_pieces))
        # Nếu không, trả về Sbasic
        else:
            return self.basic_simulations
        
    def calculate_tournament_simulations(self, state):
        """Tính số lượng mô phỏng cho từng vòng của tournament dựa trên trạng thái bàn cờ.
        
        Khi use_simulation_formula=True:
            Áp dụng công thức Stotal = Sbasic * (1 + 0.1 * nfilled) cho từng vòng tournament.
        Khi use_simulation_formula=False:
            Sử dụng giá trị cố định từ tournament_sizes.
            
        Cấu hình mặc định cho MCD: (S1, S2, S3) = (600, 600, 300)
        
        Returns:
            list: Danh sách [S1, S2, S3] chứa số lượng mô phỏng cho mỗi vòng tournament.
        """
        # Lấy các giá trị Sbasic cho từng vòng tournament
        # Mặc định: (S1, S2, S3) = (600, 600, 300)
        S1_base, S2_base, S3_base = self.tournament_sizes
        
        # Nếu dùng công thức, áp dụng Stotal = Sbasic * (1 + 0.1 * nfilled) cho từng vòng
        if self.use_simulation_formula:
            # Tính số quân cờ trên bàn (nfilled)
            total_pieces = state.balls[1] + state.balls[-1]
            
            S1 = int(S1_base * (1 + 0.1 * total_pieces))
            S2 = int(S2_base * (1 + 0.1 * total_pieces))
            S3 = int(S3_base * (1 + 0.1 * total_pieces))
        # Nếu không, sử dụng giá trị cố định từ tournament_sizes
        else:
            S1, S2, S3 = S1_base, S2_base, S3_base
        
        return [S1, S2, S3]
    
    def get_play(self):
        """Alias for get_move()"""
        return self.get_move()
        
    def get_move(self):
        """Chọn nước đi tốt nhất sử dụng Monte Carlo."""
        start_time = time.time()
        root = MonteCarloNode(self.root_state)
        simulations = self.calculate_simulations(self.root_state)
        
        simulation_count = 0
        while simulation_count < simulations and (time.time() - start_time) < self.max_time:
            # 1. Lựa chọn
            node = root
            state = deepcopy(self.root_state)
            
            # Đi xuống cây cho đến khi gặp nút có nước đi chưa thử hoặc nút lá
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
                state.move_with_position(node.move)
                state.toggle_player()
            
            # 2. Mở rộng
            if node.untried_moves != []:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                
                next_state = deepcopy(state)
                next_state.move_with_position(move)
                next_state.toggle_player()
                
                child = MonteCarloNode(next_state, parent=node, move=move)
                node.children.append(child)
                node = child
            
            # 3. Mô phỏng và 4. Lan truyền ngược
            result = self._simulate(state)
            while node:
                node.update(result)
                node = node.parent
                result = 1 - result  # Đảo ngược kết quả cho mỗi cấp
                
            simulation_count += 1
        
        # Chọn nước đi tốt nhất dựa trên số lần thăm
        if not root.children:
            return None
            
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
        
    def _evaluate_final_position(self, state, player):
        """Đánh giá vị trí cuối cùng sử dụng hàm đánh giá thống nhất.
        E(p) = Nown - Nopp
        + Win với bàn đầy: E(p) + 50
        + Win trước khi bàn đầy: E(p) + 500
        + Thua với bàn đầy: E(p) - 50
        + Thua trước khi bàn đầy: E(p) - 500
        """
        opponent = -player
        
        # Đếm số quân của mỗi bên
        num_own = state.balls[player]
        num_opp = state.balls[opponent]
        
        # Đánh giá cơ bản
        score = num_own - num_opp
        
        # Nếu trò chơi đã kết thúc, thêm điểm thưởng/phạt
        if state.is_game_over():
            if num_own > num_opp:  # Thắng
                # Kiểm tra bàn đầy hay chưa đầy
                total_pieces = num_own + num_opp
                empty_spaces = 49 - total_pieces  # Bàn cờ 7x7 có 49 ô
                if empty_spaces == 0:  # Bàn đầy
                    score += 50
                else:  # Thắng trước khi bàn đầy
                    score += 500
            elif num_own < num_opp:  # Thua
                total_pieces = num_own + num_opp
                empty_spaces = 49 - total_pieces
                if empty_spaces == 0:  # Bàn đầy
                    score -= 50
                else:  # Thua trước khi bàn đầy
                    score -= 500
                    
        # Convert to probabilities for Monte Carlo
        # Scale from large range to [0,1] range
        if score > 0:
            return min(0.9 + (score / 1000), 1.0)  # Cap at 1.0
        elif score < 0:
            return max(0.1 + (score / 1000), 0.0)  # Cap at 0.0
        else:
            return 0.5  # Draw
        
    def _simulate(self, state):
        """Mô phỏng ngẫu nhiên từ trạng thái hiện tại đến khi kết thúc."""
        state = deepcopy(state)
        player = state.current_player()
        
        while not state.is_game_over():
            moves = state.get_all_possible_moves()
            if not moves:
                break
            
            # Chọn nước đi ngẫu nhiên
            move = random.choice(moves)
            state.move_with_position(move)
            state.toggle_player()
        
        return self._evaluate_final_position(state, player)