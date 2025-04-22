from ai.base.base_mcts import BaseMCTS
from ai.rollout.minimax import minimax_rollout
from ai.reward.binary import binary_reward
from ai.selection.uct_winrate import uct_select

class MCTSMinimax(BaseMCTS):
    def __init__(self, board, current_player):
        super().__init__(board, current_player, minimax_rollout, binary_reward, uct_select)