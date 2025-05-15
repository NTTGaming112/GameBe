#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monte Carlo module for Ataxx AI
Provides factory function for creating Monte Carlo based players
"""
from .monte_carlo_base import MonteCarloBase
from .monte_carlo_domain import MonteCarloDomain
from .alpha_beta_monte_carlo import AlphaBetaMonteCarlo

def get_monte_carlo_player(game_state, mc_type="MC", number_simulations=600, 
                      switch_threshold=31, use_simulation_formula=False,
                      s1_ratio=1.0, s2_ratio=1.0, s3_ratio=0.5):
    """Factory function để tạo player Monte Carlo theo loại thuật toán.
    
    Args:
        game_state: Trạng thái game hiện tại
        mc_type: Loại Monte Carlo ("MC", "MCD", "AB+MCD")
        number_simulations: Số mô phỏng cơ bản (Sbasic ∈ {300, 600, 1200})
        switch_threshold: Ngưỡng chuyển đổi cho AB+MCD
        use_simulation_formula: Có sử dụng công thức Stotal = Sbasic * (1 + 0.1 * nfilled) hay không
        s1_ratio: Tỷ lệ số mô phỏng S1 so với number_simulations (mặc định: 1.0)
        s2_ratio: Tỷ lệ số mô phỏng S2 so với number_simulations (mặc định: 1.0)
        s3_ratio: Tỷ lệ số mô phỏng S3 so với number_simulations (mặc định: 0.5)
        
    Returns:
        Instance của thuật toán Monte Carlo
    """

        
    kwargs = {
        'basic_simulations': number_simulations,
        'switch_threshold': switch_threshold
    }
    
    if mc_type == "MCD":
        # Tính S1, S2, S3 dựa trên Sbasic và tỷ lệ được chỉ định
        # Ví dụ: Nếu Sbasic = 600, s1_ratio=1.0, s2_ratio=1.0, s3_ratio=0.5
        # => (S1, S2, S3) = (600, 600, 300)
        S1 = int(number_simulations * s1_ratio)
        S2 = int(number_simulations * s2_ratio)
        S3 = int(number_simulations * s3_ratio)
        kwargs['tournament_sizes'] = [S1, S2, S3]
        kwargs['use_simulation_formula'] = use_simulation_formula
        return MonteCarloDomain(game_state, **kwargs)
    elif mc_type == "MC":
        kwargs['use_simulation_formula'] = use_simulation_formula
        return MonteCarloBase(game_state, **kwargs)
    elif mc_type == "AB+MCD":
        kwargs['use_simulation_formula'] = use_simulation_formula
        return AlphaBetaMonteCarlo(game_state, **kwargs)
    else:
        raise ValueError(f"Unknown Monte Carlo type: {mc_type}")
