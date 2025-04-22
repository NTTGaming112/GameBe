class FullMinimax:
    def __init__(self, board, current_player, depth=2):
        self.env = board
        self.player = current_player
        self.depth = depth
        self.transposition_table = {}  # Cache trạng thái

    def run(self):
        moves = self.env.get_valid_moves()
        if not moves:
            return None
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        for move in moves:
            clone = self.env.clone()
            clone.make_move(move["from"], move["to"])
            score = self.minimax(clone, self.depth - 1, False, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
        return best_move

    def minimax(self, env, depth, maximizing, alpha, beta):
        # Kiểm tra cache
        state_key = str(env.board) + str(maximizing) + str(depth)
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        if depth == 0 or env.is_game_over():
            scores = env.calculate_scores()
            if self.player == "yellow":
                score = scores["yellowScore"] - scores["redScore"] + self.heuristic(env)
            else:
                score = scores["redScore"] - scores["yellowScore"] + self.heuristic(env)
            self.transposition_table[state_key] = score
            return score

        if maximizing:
            max_eval = float('-inf')
            moves = env.get_valid_moves()
            if not moves:
                score = float('-inf')
                self.transposition_table[state_key] = score
                return score
            for move in moves:
                clone = env.clone()
                clone.make_move(move["from"], move["to"])
                eval = self.minimax(clone, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.transposition_table[state_key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            moves = env.get_valid_moves()
            if not moves:
                score = float('inf')
                self.transposition_table[state_key] = score
                return score
            for move in moves:
                clone = env.clone()
                clone.make_move(move["from"], move["to"])
                eval = self.minimax(clone, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.transposition_table[state_key] = min_eval
            return min_eval

    def heuristic(self, env):
        # Ưu tiên nước đi chiếm nhiều quân và vị trí gần trung tâm
        captures = 0
        center_bonus = 0
        center = env.board_size // 2
        for row in range(env.board_size):
            for col in range(env.board_size):
                if env.board[row][col] == self.player:
                    # Đếm số quân đối phương có thể chiếm
                    for dr, dc in env.neighbor_offsets:
                        nr, nc = row + dr, col + dc
                        if (env.is_valid_position(nr, nc) and
                                env.board[nr][nc] not in [self.player, "empty", "block"]):
                            captures += 1
                    # Thưởng nếu quân gần trung tâm
                    distance_to_center = abs(row - center) + abs(col - center)
                    center_bonus += (env.board_size - distance_to_center) * 0.05
        return captures * 0.1 + center_bonus