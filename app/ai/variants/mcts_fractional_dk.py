from app.ai.base.base_mcts import BaseMCTS
from app.ai.rollout.heuristic import heuristic_rollout
from app.ai.reward.fractional import fractional_reward
from app.ai.selection.uct_fractional import uct_select_fractional

class MCTSFractionalDK(BaseMCTS):
    def __init__(self, board, current_player, c):
        super().__init__(board, current_player, heuristic_rollout, fractional_reward, uct_select_fractional, c)