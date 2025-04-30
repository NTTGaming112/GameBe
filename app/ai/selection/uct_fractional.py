import math
from app.ai.base.base_mcts import Node
from typing import Optional

def uct_select_fractional(node: Node, exploration_constant: float = 0.7) -> Optional[Node]:
    """
    Chọn node con tốt nhất dựa trên công thức UCT Fractional, tối ưu hóa cho phần thưởng liên tục.
    
    Args:
        node: Node cha hiện tại.
        exploration_constant: Hằng số khám phá (mặc định 0.7).
    
    Returns:
        Node con có giá trị UCT cao nhất, hoặc None nếu không có node con.
    """
    if not node.children:
        return None

    return max(
        node.children,
        key=lambda child: (
            float('inf') if child.visits == 0 else
            (child.value / child.visits) +
            exploration_constant * math.sqrt(math.log(node.visits + 1) / (child.visits + 1e-9))
        )
    )