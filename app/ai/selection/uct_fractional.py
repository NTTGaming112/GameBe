import math
def uct_select_fractional(node, exploration_constant=1.4):
    return max(node.children, key=lambda c: (
        (c.wins / (c.visits or 1)) +
        exploration_constant * math.sqrt(math.log(node.visits or 1) / (c.visits or 1))
    ))