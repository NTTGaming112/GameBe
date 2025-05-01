from app.ai.base.base_mcts import BaseMCTS
from app.ai.rollout.random import random_rollout
from app.ai.reward.fractional import fractional_reward
from app.ai.selection.uct_fractional import uct_select_fractional

class MCTSFractional(BaseMCTS):
    def __init__(self, board, current_player, c):
        super().__init__(board, current_player, random_rollout, fractional_reward, uct_select_fractional, c)