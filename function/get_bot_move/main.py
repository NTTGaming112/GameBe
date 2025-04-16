# backend-serverless/functions/get_bot_move/main.py
import functions_framework
import numpy as np
import random

class AtaxxEnvironment:
    def __init__(self, board, current_player):
        self.board = [row[:] for row in board]
        self.current_player = current_player
        self.size = len(board)

    def get_state(self):
        return tuple(tuple(row) for row in self.board)

    def get_valid_moves(self):
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == self.current_player:
                    for r in range(self.size):
                        for c in range(self.size):
                            if self.is_valid_move({"row": row, "col": col}, {"row": r, "col": c}):
                                moves.append({"from": {"row": row, "col": col}, "to": {"row": r, "col": c}})
        return moves

    def is_valid_move(self, from_pos, to_pos):
        from_row, from_col = from_pos["row"], from_pos["col"]
        to_row, to_col = to_pos["row"], to_pos["col"]

        if self.board[from_row][from_col] != self.current_player or self.board[to_row][to_col] is not None:
            return False

        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)

        if row_diff > 2 or col_diff > 2:
            return False
        return True

    def make_move(self, from_pos, to_pos):
        from_row, from_col = from_pos["row"], from_pos["col"]
        to_row, to_col = to_pos["row"], to_pos["col"]

        new_board = [row[:] for row in self.board]
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)

        if row_diff <= 1 and col_diff <= 1:
            new_board[to_row][to_col] = self.current_player
        else:
            new_board[to_row][to_col] = self.current_player
            new_board[from_row][from_col] = None

        for r in range(max(0, to_row - 1), min(self.size, to_row + 2)):
            for c in range(max(0, to_col - 1), min(self.size, to_col + 2)):
                if new_board[r][c] and new_board[r][c] != self.current_player and new_board[r][c] != "block":
                    new_board[r][c] = self.current_player

        self.board = new_board
        self.current_player = "red" if self.current_player == "yellow" else "yellow"

    def calculate_scores(self):
        yellow_score = 0
        red_score = 0
        for row in self.board:
            for cell in row:
                if cell == "yellow":
                    yellow_score += 1
                elif cell == "red":
                    red_score += 1
        return {"yellowScore": yellow_score, "redScore": red_score}

    def is_game_over(self):
        yellow_moves = len(self.get_valid_moves()) > 0 and self.current_player == "yellow"
        red_moves = len(self.get_valid_moves()) > 0 and self.current_player == "red"
        return not yellow_moves and not red_moves

    def get_reward(self, bot_player):
        if not self.is_game_over():
            return 0
        scores = self.calculate_scores()
        if scores["yellowScore"] > scores["redScore"]:
            return 1 if bot_player == "yellow" else -1
        elif scores["redScore"] > scores["yellowScore"]:
            return 1 if bot_player == "red" else -1
        return 0

class MCTSNode:
    def __init__(self, state, current_player, parent=None, move=None):
        self.state = state
        self.current_player = current_player
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0

class MCTS:
    def __init__(self, board, current_player, iterations=500):  # Giảm iterations để tránh timeout
        self.env = AtaxxEnvironment(board, current_player)
        self.root = MCTSNode(self.env.get_state(), current_player)
        self.iterations = iterations
        self.exploration_constant = 0.7
        self.bot_player = current_player

    def select(self):
        node = self.root
        while node.children:
            node = max(node.children, key=lambda c: (
                (c.wins / (c.visits or 1)) +
                self.exploration_constant * np.sqrt(np.log(node.visits or 1) / (c.visits or 1))
            ))
        return node

    def expand(self, node):
        env = AtaxxEnvironment([list(row) for row in node.state], node.current_player)
        valid_moves = env.get_valid_moves()
        if not valid_moves or env.is_game_over():
            return node
        for move in valid_moves:
            env_temp = AtaxxEnvironment([list(row) for row in node.state], node.current_player)
            env_temp.make_move(move["from"], move["to"])
            child = MCTSNode(env_temp.get_state(), env_temp.current_player, parent=node, move=move)
            node.children.append(child)
        return random.choice(node.children)

    def simulate(self, node):
        env = AtaxxEnvironment([list(row) for row in node.state], node.current_player)
        while not env.is_game_over():
            moves = env.get_valid_moves()
            if not moves:
                break
            move = random.choice(moves)
            env.make_move(move["from"], move["to"])
        return env.get_reward(self.bot_player)

    def backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.wins += result
            node = node.parent

    def search(self):
        for _ in range(self.iterations):
            node = self.select()
            expanded_node = self.expand(node)
            result = self.simulate(expanded_node)
            self.backpropagate(expanded_node, result)
        if not self.root.children:
            return None
        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.move

@functions_framework.http
def get_bot_move(request):
    if request.method != "POST":
        return {"statusCode": 405, "body": "Method Not Allowed"}

    try:
        data = request.get_json()
        board = data.get("board")
        current_player = data.get("current_player")
        if not board or not current_player:
            return {"statusCode": 400, "body": "Board and current_player are required"}

        mcts = MCTS(board, current_player)
        move = mcts.search()
        if not move:
            return {"statusCode": 404, "body": "No valid move found"}

        return {"statusCode": 200, "body": move}
    except Exception as e:
        return {"statusCode": 400, "body": str(e)}