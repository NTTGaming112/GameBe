import numpy as np
from collections import defaultdict
import random
from typing import List, Dict, Optional, Tuple
from app.database import get_games_collection, get_move_history_collection

BOARD_SIZE = 7
MAX_MOVE_DISTANCE = 2
NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

class AtaxxEnvironment:
    def __init__(self, board: List[List[str]], current_player: str):
        self.board = board
        self.current_player = current_player
        self.size = len(board)
        self._valid_moves_cache: Optional[List[Dict[str, Dict[str, int]]]] = None
        self._game_over_cache: Optional[bool] = None

    def get_state(self) -> Tuple[Tuple[str, ...], ...]:
        return tuple(tuple(row) for row in self.board)

    def get_valid_moves(self) -> List[Dict[str, Dict[str, int]]]:
        if self._valid_moves_cache is not None:
            return self._valid_moves_cache

        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] != self.current_player:
                    continue
                # Chỉ kiểm tra các ô trong khoảng cách MAX_MOVE_DISTANCE
                min_row = max(0, row - MAX_MOVE_DISTANCE)
                max_row = min(self.size - 1, row + MAX_MOVE_DISTANCE) + 1
                min_col = max(0, col - MAX_MOVE_DISTANCE)
                max_col = min(self.size - 1, col + MAX_MOVE_DISTANCE) + 1

                for r in range(min_row, max_row):
                    for c in range(min_col, max_col):
                        to_pos = {"row": r, "col": c}
                        if self.is_valid_move({"row": row, "col": col}, to_pos):
                            moves.append({"from": {"row": row, "col": col}, "to": to_pos})

        self._valid_moves_cache = moves
        return moves

    def is_valid_move(self, from_pos: Dict[str, int], to_pos: Dict[str, int]) -> bool:
        from_row, from_col = from_pos["row"], from_pos["col"]
        to_row, to_col = to_pos["row"], to_pos["col"]

        if (self.board[from_row][from_col] != self.current_player or 
            self.board[to_row][to_col] != "empty"):
            return False

        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        return row_diff <= MAX_MOVE_DISTANCE and col_diff <= MAX_MOVE_DISTANCE

    def count_captured_pieces(self, to_pos: Dict[str, int]) -> int:
        to_row, to_col = to_pos["row"], to_pos["col"]
        captured = 0
        for dr, dc in NEIGHBOR_OFFSETS:
            r, c = to_row + dr, to_col + dc
            if (0 <= r < self.size and 0 <= c < self.size and 
                self.board[r][c] not in ["empty", "block", self.current_player]):
                captured += 1
        return captured

    def make_move(self, from_pos: Dict[str, int], to_pos: Dict[str, int]) -> None:
        from_row, from_col = from_pos["row"], from_pos["col"]
        to_row, to_col = to_pos["row"], to_pos["col"]

        new_board = [row[:] for row in self.board]
        row_diff = abs(to_row - from_row)
        col_diff = abs(to_col - from_col)
        if row_diff <= 1 and col_diff <= 1:
            new_board[to_row][to_col] = self.current_player
        else:
            new_board[to_row][to_col] = self.current_player
            new_board[from_row][from_col] = "empty"

        for dr, dc in NEIGHBOR_OFFSETS:
            r, c = to_row + dr, to_col + dc
            if (0 <= r < self.size and 0 <= c < self.size and 
                new_board[r][c] not in ["empty", "block", self.current_player]):
                new_board[r][c] = self.current_player

        self.board = new_board
        self.current_player = "red" if self.current_player == "yellow" else "yellow"
        self._valid_moves_cache = None
        self._game_over_cache = None

    def calculate_scores(self) -> Dict[str, int]:
        yellow_score = sum(row.count("yellow") for row in self.board)
        red_score = sum(row.count("red") for row in self.board)
        return {"yellowScore": yellow_score, "redScore": red_score}

    def is_game_over(self) -> bool:
        if self._game_over_cache is not None:
            return self._game_over_cache

        moves = self.get_valid_moves()
        if not moves:
            self.current_player = "red" if self.current_player == "yellow" else "yellow"
            moves = self.get_valid_moves()
            self.current_player = "red" if self.current_player == "yellow" else "yellow"
            result = len(moves) == 0
            self._game_over_cache = result
            return result
        self._game_over_cache = False
        return False

    def get_reward(self, bot_player: str) -> float:
        if not self.is_game_over():
            return 0
        scores = self.calculate_scores()
        if scores["yellowScore"] > scores["redScore"]:
            return 1 if bot_player == "yellow" else -1
        elif scores["redScore"] > scores["yellowScore"]:
            return 1 if bot_player == "red" else -1
        return 0

class MCTSNode:
    def __init__(self, state: Tuple[Tuple[str, ...], ...], current_player: str, 
                 parent: Optional["MCTSNode"] = None, move: Optional[Dict] = None):
        self.state = state
        self.current_player = current_player
        self.parent = parent
        self.move = move
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.wins = 0
        self.captured_pieces = 0

class MCTS:
    def __init__(self, board: List[List[str]], current_player: str, 
                 iterations: int = 300, move_history: Optional[defaultdict] = None):
        self.env = AtaxxEnvironment([row[:] for row in board], current_player)
        self.root = MCTSNode(self.env.get_state(), current_player)
        self.iterations = iterations
        self.exploration_constant = 0.7
        self.move_history = move_history or defaultdict(lambda: defaultdict(float))
        self.bot_player = current_player
        self.history_weight = 0.6
        self.capture_weight = 0.4
        self.state_cache: Dict[Tuple[Tuple[str, ...], ...], float] = {}  # Cache kết quả mô phỏng

    def select(self) -> "MCTSNode":
        node = self.root
        while node.children:
            node = max(node.children, key=lambda c: (
                (c.wins / (c.visits or 1)) +
                self.exploration_constant * np.sqrt(np.log(node.visits or 1) / (c.visits or 1))
            ))
        return node

    def expand(self, node: "MCTSNode") -> "MCTSNode":
        env = AtaxxEnvironment([list(row) for row in node.state], node.current_player)
        valid_moves = env.get_valid_moves()
        if not valid_moves or env.is_game_over():
            return node

        for move in valid_moves:
            env_temp = AtaxxEnvironment([list(row) for row in node.state], node.current_player)
            captured = env_temp.count_captured_pieces(move["to"])
            env_temp.make_move(move["from"], move["to"])
            child = MCTSNode(env_temp.get_state(), env_temp.current_player, parent=node, move=move)
            child.captured_pieces = captured
            node.children.append(child)
        return random.choice(node.children)

    def simulate(self, node: "MCTSNode") -> float:
        state = node.state
        if state in self.state_cache:
            return self.state_cache[state]

        env = AtaxxEnvironment([list(row) for row in state], node.current_player)
        state_key = str(state)
        depth = 0
        max_depth = 20  # Giới hạn độ sâu mô phỏng để tránh chạy quá lâu

        while not env.is_game_over() and depth < max_depth:
            moves = env.get_valid_moves()
            if not moves:
                break

            best_move = None
            best_score = -float("inf")
            for move in moves:
                move_key = f"{move['from']['row']},{move['from']['col']}-{move['to']['row']},{move['to']['col']}"
                history_score = self.move_history[state_key].get(move_key, 0)
                capture_score = env.count_captured_pieces(move["to"])
                score = (self.history_weight * history_score) + (self.capture_weight * capture_score)
                if score > best_score:
                    best_score = score
                    best_move = move

            move = best_move if best_move else random.choice(moves)
            env.make_move(move["from"], move["to"])
            state_key = str(env.get_state())
            depth += 1

        result = env.get_reward(self.bot_player)
        self.state_cache[state] = result
        return result

    def backpropagate(self, node: "MCTSNode", result: float) -> None:
        while node:
            node.visits += 1
            node.wins += result
            node = node.parent

    def search(self) -> Optional[Dict]:
        for _ in range(self.iterations):
            node = self.select()
            expanded_node = self.expand(node)
            result = self.simulate(expanded_node)
            self.backpropagate(expanded_node, result)

        if not self.root.children:
            return None

        best_child = max(self.root.children, key=lambda c: (
            (c.visits * 0.7) + (c.captured_pieces * 0.3)
        ))
        return best_child.move

def train_mcts(games_data: Optional[List[Dict]] = None, weight_decay: float = 0.9) -> None:
    games_collection = get_games_collection()
    move_history_collection = get_move_history_collection()
    move_history = defaultdict(lambda: defaultdict(float))

    if games_data is None:
        try:
            games_data = list(games_collection.find().limit(100))  # Giới hạn số ván để tăng tốc
        except Exception as e:
            print(f"Error retrieving games data from MongoDB: {e}")
            return

    if not games_data:
        print("No games data available for training.")
        return

    try:
        history_data = move_history_collection.find_one() or {"data": {}}
        for state_key, moves in history_data.get("data", {}).items():
            for move_key, score in moves.items():
                move_history[state_key][move_key] = score
    except Exception as e:
        print(f"Error loading move history from MongoDB: {e}")
        return

    for state_key in move_history:
        for move_key in move_history[state_key]:
            move_history[state_key][move_key] *= weight_decay

    for game in games_data:
        winner = game.get("winner")
        board_states = game.get("board_states", [])
        moves = game.get("moves", [])

        if not board_states or not moves or len(board_states) <= len(moves):
            continue

        for i, board_state in enumerate(board_states):
            if i >= len(moves):
                break
            state_key = str(tuple(tuple(row) for row in board_state))
            move = moves[i]
            move_key = f"{move['from_pos']['row']},{move['from_pos']['col']}-{move['to_pos']['row']},{move['to_pos']['col']}"
            if winner == "yellow" and i % 2 == 0:
                move_history[state_key][move_key] += 1.0
            elif winner == "red" and i % 2 == 1:
                move_history[state_key][move_key] += 1.0
            elif winner == "draw":
                move_history[state_key][move_key] += 0.5
            else:
                move_history[state_key][move_key] -= 0.2

    try:
        move_history_collection.delete_many({})
        move_history_collection.insert_one({"data": {k: dict(v) for k, v in move_history.items()}})
    except Exception as e:
        print(f"Error saving move history to MongoDB: {e}")

def get_trained_move(board: List[List[str]], current_player: str) -> Optional[Dict]:
    move_history_collection = get_move_history_collection()
    move_history = defaultdict(lambda: defaultdict(float))

    try:
        history_data = move_history_collection.find_one() or {"data": {}}
        for state_key, moves in history_data.get("data", {}).items():
            for move_key, score in moves.items():
                move_history[state_key][move_key] = score
    except Exception as e:
        print(f"Error loading move history from MongoDB: {e}")
        return None

    mcts = MCTS(board, current_player, move_history=move_history)
    return mcts.search()