from ai.base.base_mcts import BaseMCTS
from ai.rollout.random import random_rollout
from ai.reward.binary import binary_reward
from ai.selection.uct_winrate import uct_select

class MCTSBinary(BaseMCTS):
    def __init__(self, board, current_player):
        super().__init__(board, current_player, random_rollout, binary_reward, uct_select)