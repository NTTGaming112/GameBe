#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
from copy import deepcopy
from .monte_carlo_base import MonteCarloBase

class MonteCarloDomain(MonteCarloBase):
    """Monte Carlo với Domain Knowledge."""
    def __init__(self, state, **kwargs):
        # Tournament parameters
        self.tournament_rounds = kwargs.get('tournament_rounds', 3)
        # Mặc định (S1, S2, S3) = (600, 600, 300)
        self.tournament_sizes = kwargs.get('tournament_sizes', [600, 600, 300])
        super().__init__(state, **kwargs)
        
    def get_move(self):
        """Sử dụng Tournament Layering để chọn nước đi tốt nhất."""
        moves = self.root_state.get_all_possible_moves()
        if not moves or len(moves) == 1:
            return moves[0] if moves else None
            
        # Cấu hình Tournament Layering
        k1 = min(5, len(moves))       # Số nước giữ lại sau vòng 1
        k2 = min(3, k1)               # Số nước giữ lại sau vòng 2
        
        # Tính số lượng mô phỏng cho mỗi vòng tournament
        # S1, S2, S3 phụ thuộc vào use_simulation_formula
        tournament_sizes = self.calculate_tournament_simulations(self.root_state)
        S1, S2, S3 = tournament_sizes  # Số mô phỏng mỗi vòng
        
        # Vòng 1: Mô phỏng S1 rollout cho mỗi nước đi
        print(f"Tournament vòng 1: Đánh giá {len(moves)} nước với {S1} mô phỏng mỗi nước")
        move_scores = []
        for move in moves:
            next_state = deepcopy(self.root_state)
            next_state.move_with_position(move)
            next_state.toggle_player()
            score = self._evaluate_move(next_state, S1)
            move_scores.append((move, score))
        
        # Sắp xếp và chọn k1 nước đi tốt nhất
        move_scores.sort(key=lambda x: x[1], reverse=True)
        candidates = move_scores[:k1]
        print(f"Kết quả vòng 1: Chọn {len(candidates)} nước đi tốt nhất")
        
        # Vòng 2: Mô phỏng thêm S2 rollout cho top k1 nước
        if len(candidates) > 1:
            print(f"Tournament vòng 2: Đánh giá {len(candidates)} nước với {S2} mô phỏng mỗi nước")
            new_scores = []
            
            for move, prev_score in candidates:
                next_state = deepcopy(self.root_state)
                next_state.move_with_position(move)
                next_state.toggle_player()
                additional_score = self._evaluate_move(next_state, S2)
                combined_score = (prev_score * S1 + additional_score * S2) / (S1 + S2)
                new_scores.append((move, combined_score))
            
            new_scores.sort(key=lambda x: x[1], reverse=True)
            candidates = new_scores[:k2]
            print(f"Kết quả vòng 2: Chọn {len(candidates)} nước đi tốt nhất")
        
        # Vòng 3: Mô phỏng thêm S3 rollout cho top k2 nước
        if len(candidates) > 1:
            print(f"Tournament vòng 3: Đánh giá {len(candidates)} nước với {S3} mô phỏng mỗi nước")
            final_scores = []
            
            for move, prev_score in candidates:
                next_state = deepcopy(self.root_state)
                next_state.move_with_position(move)
                next_state.toggle_player()
                additional_score = self._evaluate_move(next_state, S3)
                combined_score = (prev_score * (S1 + S2) + additional_score * S3) / (S1 + S2 + S3)
                final_scores.append((move, combined_score))
            
            final_scores.sort(key=lambda x: x[1], reverse=True)
            candidates = final_scores
        
        # Trả về nước đi tốt nhất
        best_move = candidates[0][0]
        print(f"Kết quả cuối cùng: Chọn nước đi với điểm số {candidates[0][1]:.4f}")
        return best_move
        
    def _evaluate_move(self, state, simulations):
        """Đánh giá nước đi bằng Monte Carlo kết hợp domain knowledge."""
        wins = 0
        for _ in range(simulations):
            result = self._simulate_with_domain_knowledge(state)
            wins += result
        return wins / simulations
        
    def _simulate_with_domain_knowledge(self, state):
        """Mô phỏng với domain knowledge và phân phối xác suất."""
        state = deepcopy(state)
        player = state.current_player()
        
        while not state.is_game_over():
            moves = state.get_all_possible_moves()
            if not moves:
                break
                
            # Áp dụng domain knowledge và phân phối xác suất để chọn nước đi
            move = self._select_move_with_probability_distribution(state, moves)
            state.move_with_position(move)
            state.toggle_player()
        
        # Đánh giá kết quả với hàm đánh giá thống nhất
        return self._evaluate_final_position(state, player)
        
    def _select_move_with_probability_distribution(self, state, moves):
        """Chọn nước đi dựa trên phân phối xác suất P(m_i) = S_i^2 / ∑_j=1^M S_j^2."""
        if not moves:
            return None
        
        # Tính điểm và bình phương cho mỗi nước đi
        scored_moves = [(move, self._score_move(state, move)) for move in moves]
        scores_squared = [score**2 for _, score in scored_moves]
        sum_scores_squared = sum(scores_squared)
        
        # Xử lý trường hợp tất cả điểm đều bằng 0
        if sum_scores_squared == 0:
            return random.choice(moves)
        
        # Chọn nước đi ngẫu nhiên dựa trên phân phối xác suất
        r = random.random() * sum_scores_squared
        current_sum = 0
        for i, score_squared in enumerate(scores_squared):
            current_sum += score_squared
            if r <= current_sum:
                return scored_moves[i][0]
        
        return scored_moves[-1][0]
    
    def _score_move(self, state, move):
        """Đánh giá một nước đi dựa trên heuristics.
        Si = s1·(số quân địch bị chiếm) + s2·(số quân ta xung quanh ô đích) 
           + s3·1{Clone} − s4·(số quân ta quanh ô nguồn nếu Jump)
        """
        player = state.current_player()
        s1, s2, s3, s4 = 1.0, 0.4, 0.7, 0.4  # Trọng số heuristic
        
        # Tạo trạng thái mới và tính số quân bị chiếm
        next_state = deepcopy(state)
        next_state.move_with_position(move)
        captures = state.balls[-player] - next_state.balls[-player]
        
        # Xác định loại nước đi và các vị trí
        if move[0] == 'c':  # Clone
            dest_x, dest_y = move[1]
            is_clone = 1
            adjacent_friendly_pieces_source = 0
        else:  # Jump
            source_x, source_y = move[1]
            dest_x, dest_y = move[2]
            is_clone = 0
            
            # Đếm quân ta xung quanh ô nguồn
            adjacent_friendly_pieces_source = sum(1 for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                                              if (dx != 0 or dy != 0) and 0 <= source_x + dx < 7 
                                              and 0 <= source_y + dy < 7 
                                              and state.board[source_x + dx][source_y + dy] == player)
        
        # Đếm quân ta xung quanh ô đích
        adjacent_friendly_pieces_dest = sum(1 for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                                        if (dx != 0 or dy != 0) and 0 <= dest_x + dx < 7 
                                        and 0 <= dest_y + dy < 7 
                                        and next_state.board[dest_x + dx][dest_y + dy] == player)
        
        # Tính điểm theo công thức và đảm bảo không âm
        score = (s1 * captures + s2 * adjacent_friendly_pieces_dest + 
                s3 * is_clone - s4 * adjacent_friendly_pieces_source)
        
        return max(0, score)