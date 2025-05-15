#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module để quản lý các vị trí bàn cờ, phục vụ cho quy tắc lặp lại 3 lần.
"""

class BoardPositionTracker:
    """
    Lớp theo dõi lịch sử vị trí bàn cờ để phát hiện trường hợp 
    một vị trí xuất hiện lần thứ 3 với cùng lượt người chơi.
    """
    
    def __init__(self):
        # Dictionary lưu trữ vị trí bàn cờ và số lần xuất hiện
        # Key: (board_tuple, player)
        # Value: số lần xuất hiện
        self.position_history = {}
        
    def board_to_tuple(self, board):
        """Chuyển đổi bảng 2D thành tuple 1D để sử dụng làm khóa trong dictionary"""
        flat_board = []
        for row in board:
            flat_board.extend(row)
        return tuple(flat_board)
    
    def update_position(self, board, player):
        """Cập nhật lịch sử vị trí bàn cờ"""
        position_key = (self.board_to_tuple(board), player)
        
        if position_key in self.position_history:
            self.position_history[position_key] += 1
        else:
            self.position_history[position_key] = 1
            
        return self.position_history[position_key]
    
    def check_three_repetitions(self, board, player):
        """Kiểm tra xem vị trí hiện tại có xuất hiện 3 lần không"""
        position_key = (self.board_to_tuple(board), player)
        return self.position_history.get(position_key, 0) >= 3
    
    def reset(self):
        """Reset lịch sử vị trí"""
        self.position_history = {}
