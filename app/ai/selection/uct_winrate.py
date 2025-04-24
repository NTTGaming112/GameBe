import math
from app.ai.base.base_mcts import Node

def uct_select_winrate(node: Node, exploration_constant: float = 0.7) -> Node:
    """
    Chọn node con tốt nhất dựa trên công thức UCT Winrate.
    
    Args:
        node: Node cha hiện tại.
        exploration_constant: Hằng số khám phá (thường là 0.7).
    
    Returns:
        Node con có giá trị UCT cao nhất.
    """
    if not node.children:
        return None

    best_uct = float('-inf')
    best_child = None

    for child in node.children:
        if child.visits == 0:
            uct_value = float('inf')  # Ưu tiên node chưa được thăm
        else:
            # UCT Winrate: Dùng tỷ lệ thắng (wins/visits)
            x = child.wins / child.visits
            exploration_term = exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
            uct_value = x + exploration_term

        if uct_value > best_uct:
            best_uct = uct_value
            best_child = child

    return best_child