from ai.base.base_mcts import BaseMCTS
from ai.rollout.heuristic import heuristic_rollout
from ai.reward.fractional import fractional_reward
from ai.selection.uct_fractional import uct_select_fractional

class MCTSFractionalDK(BaseMCTS):
    def __init__(self, board, current_player):
        super().__init__(board, current_player, heuristic_rollout, fractional_reward, uct_select_fractional)