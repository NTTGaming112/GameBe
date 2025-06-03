import time
from heuristics import evaluate

class MinimaxAgent:
    def __init__(self, max_depth=4, time_limit=None):
        self.max_depth = max_depth
        self.time_limit = time_limit

    def get_ordered_moves(self, state):
        moves = state.get_legal_moves()
        if not moves:
            return []
        move_scores = []
        for move in moves:
            r, c, nr, nc = move
            score = 0
            # Count flipped pieces
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nnr, nnc = nr + dr, nc + dc
                    if (0 <= nnr < 7 and 0 <= nnc < 7 and
                            state.board[nnr][nnc] == -state.current_player):
                        score += 1
            move_scores.append((move, score))
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]

    def minimax(self, state, depth, alpha, beta, maximizingPlayer, max_depth):
        empty_cells = state.get_empty_cells()
        if empty_cells <= 5:
            max_depth = min(max_depth, 5)
        if empty_cells <= 2:
            max_depth = min(max_depth, 6)
        if depth >= max_depth or state.is_game_over():
            return evaluate(state, 1), None
        moves = self.get_ordered_moves(state)
        if not moves:
            return evaluate(state, 1), None
        if maximizingPlayer:
            maxEval = float('-inf')
            best_move = None
            for move in moves:
                new_state = state.copy()
                new_state.make_move(move)
                eval, _ = self.minimax(new_state, depth + 1, alpha, beta, False, max_depth)
                if eval > maxEval:
                    maxEval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return maxEval, best_move
        else:
            minEval = float('inf')
            best_move = None
            for move in moves:
                new_state = state.copy()
                new_state.make_move(move)
                eval, _ = self.minimax(new_state, depth + 1, alpha, beta, True, max_depth)
                if eval < minEval:
                    minEval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return minEval, best_move

    def get_move(self, state):
        start_time = time.time()
        best_move = None
        for depth in range(1, self.max_depth + 1):
            if self.time_limit is not None:
                if time.time() - start_time > self.time_limit:
                    break
            _, move = self.minimax(state, 0, float('-inf'), float('inf'), True, depth)
            if move:
                best_move = move
        return best_move