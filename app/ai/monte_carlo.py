#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monte Carlo module for Ataxx AI.

This module provides a factory function to create different types of
Monte Carlo-based AI players for the Ataxx game.
"""
from .monte_carlo_base import MonteCarloBase
from .monte_carlo_domain import MonteCarloDomain
from .alpha_beta_monte_carlo import AlphaBetaMonteCarlo
from .minimax_player import MinimaxPlayer
from app.ai.constants import MC_TYPE_BASIC, MC_TYPE_DOMAIN, MC_TYPE_ALPHA_BETA, MC_TYPE_MINIMAX

def get_monte_carlo_player(game_state, mc_type=MC_TYPE_BASIC, number_simulations=600,
                          switch_threshold=31, use_simulation_formula=False,
                          s1_ratio=1.0, s2_ratio=1.0, s3_ratio=0.5, depth=4, time_limit=50):
    """Factory function to create Monte Carlo-based AI players.
    
    This function creates different types of Monte Carlo AI players based on the
    specified parameters.
    
    Args:
        game_state: Current game state
        mc_type: Monte Carlo algorithm type (MC_TYPE_BASIC, MC_TYPE_DOMAIN, 
                MC_TYPE_ALPHA_BETA, MC_TYPE_MINIMAX)
        number_simulations: Base simulation count (Sbasic ∈ {300, 600, 1200})
        switch_threshold: Switch threshold for hybrid algorithm (AB+MCD)
        use_simulation_formula: Whether to use Stotal = Sbasic * (1 + 0.1 * nfilled)
        s1_ratio: Ratio of S1 simulations to number_simulations (default: 1.0)
        s2_ratio: Ratio of S2 simulations to number_simulations (default: 1.0)
        s3_ratio: Ratio of S3 simulations to number_simulations (default: 0.5)
        depth: Depth for Minimax search (default: 4)
        time_limit: Time limit for AI decision making in seconds (default: 50)
        
    Returns:
        An instance of the appropriate Monte Carlo algorithm
    """
    # Common configuration parameters
    kwargs = {
        'basic_simulations': number_simulations,
        'switch_threshold': switch_threshold,
        'use_simulation_formula': use_simulation_formula
    }
    
    if mc_type == MC_TYPE_DOMAIN:
        # Monte Carlo with Domain Knowledge
        # Calculate S1, S2, S3 based on Sbasic and specified ratios
        # Example: If Sbasic = 600, s1_ratio=1.0, s2_ratio=1.0, s3_ratio=0.5
        # => (S1, S2, S3) = (600, 600, 300)
        S1 = int(number_simulations * s1_ratio)
        S2 = int(number_simulations * s2_ratio)
        S3 = int(number_simulations * s3_ratio)
        kwargs['tournament_sizes'] = [S1, S2, S3]
        return MonteCarloDomain(game_state, time_limit=time_limit, **kwargs)
    
    elif mc_type == MC_TYPE_ALPHA_BETA:
        # Hybrid: Alpha-Beta + Monte Carlo Domain
        S1 = int(number_simulations * s1_ratio)
        S2 = int(number_simulations * s2_ratio)
        S3 = int(number_simulations * s3_ratio)
        kwargs['tournament_sizes'] = [S1, S2, S3]
        return AlphaBetaMonteCarlo(game_state, time_limit=time_limit, **kwargs)
        
    elif mc_type == MC_TYPE_MINIMAX:
        # Alpha-Beta Minimax
        return MinimaxPlayer(game_state, depth=depth, time_limit=time_limit)
    
    else:
        # Default: Basic Monte Carlo Tree Search
        return MonteCarloBase(game_state, time_limit=time_limit, **kwargs)

