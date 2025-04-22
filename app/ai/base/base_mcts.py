class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_valid_moves())

class BaseMCTS:
    def __init__(self, board, current_player, rollout_fn, reward_fn, selection_fn):
        self.env = board.clone()
        self.bot_player = current_player
        self.rollout_fn = rollout_fn
        self.reward_fn = reward_fn
        self.selection_fn = selection_fn

    def run(self, simulations=100):
        root = MCTSNode(self.env.clone())
        for _ in range(simulations):
            node = self.select(root)
            if not node.state.is_game_over():
                self.expand(node)
                result_env = self.rollout_fn(node.state.clone(), self.bot_player)
                reward = self.reward_fn(result_env, self.bot_player)
                self.backpropagate(node, reward)
        return self.best_move(root)

    def select(self, node):
        while node.children:
            node = self.selection_fn(node)
        return node

    def expand(self, node):
        for move in node.state.get_valid_moves():
            new_env = node.state.clone()
            new_env.make_move(move["from"], move["to"])
            child = MCTSNode(new_env, parent=node)
            node.children.append(child)

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.wins += reward
            node = node.parent

    def best_move(self, root):
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.state.last_move