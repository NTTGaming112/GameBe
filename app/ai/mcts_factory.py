from app.ai.base_mcts import BaseMCTS
from app.ai.binary_mcts import BinaryMCTS
from app.ai.binary_mcts_dk import BinaryMCTSDK
from app.ai.fractional_mcts import FractionalMCTS
from app.ai.fractional_mcts_dk import FractionalMCTSDK
from app.ai.binary_mcts_minimax2 import BinaryMCTSMinimax2
from app.ai.minimax2 import Minimax2
from app.ai.random import RandomAlgorithm


def create_mcts(algorithm: str, iterations: int = 1000, time_limit: float = None, policy_type: str = "random", **policy_args) -> BaseMCTS:
    if algorithm == "mcts-binary":
        return BinaryMCTS(iterations=iterations, time_limit=time_limit, policy_type="random", **policy_args)
    elif algorithm == "mcts-binary-dk":
        temperature = policy_args.pop("temperature", 0.7)
        return BinaryMCTSDK(iterations=iterations, time_limit=time_limit, temperature=temperature, **policy_args)
    elif algorithm == "mcts-fractional":
        return FractionalMCTS(iterations=iterations, time_limit=time_limit, policy_type="random", **policy_args)
    elif algorithm == "mcts-fractional-dk":
        temperature = policy_args.pop("temperature", 0.7)
        return FractionalMCTSDK(iterations=iterations, time_limit=time_limit, temperature=temperature, **policy_args)
    elif algorithm == "mcts-binary-minimax2":
        return BinaryMCTSMinimax2(iterations=iterations, time_limit=time_limit, policy_type="random", **policy_args)
    elif algorithm == "minimax2":
        return Minimax2(iterations=iterations, time_limit=time_limit, **policy_args)
    elif algorithm == "random":
        return RandomAlgorithm()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")