from app.ai.monte_carlo_base import MonteCarloBase
from app.ai.monte_carlo_domain import MonteCarloDomain
from app.ai.alpha_beta_monte_carlo import AlphaBetaMonteCarlo
from app.ai.minimax_player import MinimaxPlayer

def get_monte_carlo_player(game, algo_type, number_simulations=300, s1_ratio=1.0, s2_ratio=1.0, s3_ratio=0.5, time_limit=None, switch_threshold=31, minimax_depth=4):
    """Return a player based on the algorithm type."""
    if algo_type == "Minimax":
        return MinimaxPlayer(game, depth=minimax_depth, time_limit=time_limit)
    elif algo_type == "MC":
        return MonteCarloBase(game, basic_simulations=number_simulations, time_limit=time_limit)
    elif algo_type == "MCD":
        return MonteCarloDomain(
            game, 
            basic_simulations=number_simulations, 
            time_limit=time_limit,
            component_weights={'heuristic': s1_ratio, 'tactical': s2_ratio, 'strategic': s3_ratio}
        )
    elif algo_type == "AB+MCD":
        return AlphaBetaMonteCarlo(
            game,
            basic_simulations=number_simulations,
            time_limit=time_limit,
            switch_threshold=switch_threshold,
            minimax_depth=minimax_depth,
            s1_ratio=s1_ratio,
            s2_ratio=s2_ratio,
            s3_ratio=s3_ratio
        )
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")