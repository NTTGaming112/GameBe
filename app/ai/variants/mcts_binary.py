from app.ai.base.base_mcts import BaseMCTS
from app.ai.rollout.random import random_rollout
from app.ai.reward.binary import binary_reward
from app.ai.selection.uct_winrate import uct_select_winrate

class MCTSBinary(BaseMCTS):
    def __init__(self, board, current_player, c):
        super().__init__(board, current_player, random_rollout, binary_reward, uct_select_winrate, c)