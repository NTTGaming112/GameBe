import math
from app.ai.base.base_mcts import MCTSNode
from typing import Optional

def uct_select_fractional(MCTSNode: MCTSNode, exploration_constant: float = 0.7) -> Optional[MCTSNode]:
    """
    Chọn MCTSNode con tốt nhất dựa trên công thức UCT Fractional, tối ưu hóa cho phần thưởng liên tục.
    
    Args:
        MCTSNode: MCTSNode cha hiện tại.
        exploration_constant: Hằng số khám phá (mặc định 0.7).
    
    Returns:
        MCTSNode con có giá trị UCT cao nhất, hoặc None nếu không có MCTSNode con.
    """
    if not MCTSNode.children:
        return None

    return max(
        MCTSNode.children,
        key=lambda child: (
            float('inf') if child.visits == 0 else
            (child.value / child.visits) +
            exploration_constant * math.sqrt(math.log(MCTSNode.visits + 1) / (child.visits + 1e-9))
        )
    )