from app.ai.ataxx_env import AtaxxEnvironment

def sort_moves(moves, env):
    """Sắp xếp nước đi ưu tiên nhảy xa và chiếm quân."""
    def move_priority(move):
        distance = abs(move['from'][0] - move['to'][0]) + abs(move['from'][1] - move['to'][1])
        captured = 0
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            ni, nj = move['to'][0] + di, move['to'][1] + dj
            if 0 <= ni < len(env.board) and 0 <= nj < len(env.board):
                if env.board[ni][nj] == ('red' if env.current_player == 'yellow' else 'yellow'):
                    captured += 1
        return (-distance, -captured)
    return sorted(moves, key=move_priority)

def evaluate_board(env: AtaxxEnvironment, player: str) -> int:
    """Đánh giá bàn cờ dựa trên nhiều yếu tố để tối ưu hóa lợi thế."""
    try:
        yellow_score = sum(row.count('yellow') for row in env.board)
        red_score = sum(row.count('red') for row in env.board)
        piece_diff = yellow_score - red_score

        env_clone = env.clone()
        env_clone.current_player = 'yellow'
        yellow_moves = len(env_clone.get_valid_moves())
        env_clone.current_player = 'red'
        red_moves = len(env_clone.get_valid_moves())
        move_diff = yellow_moves - red_moves

        center_positions = [(2,2), (2,3), (2,4), (3,2), (3,3), (3,4), (4,2), (4,3), (4,4)]
        yellow_center = sum(1 for pos in center_positions if env.board[pos[0]][pos[1]] == 'yellow')
        red_center = sum(1 for pos in center_positions if env.board[pos[0]][pos[1]] == 'red')
        center_diff = yellow_center - red_center

        def count_neighbors(board, color):
            count = 0
            for i in range(len(board)):
                for j in range(len(board)):
                    if board[i][j] == color:
                        for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < len(board) and 0 <= nj < len(board) and board[ni][nj] == color:
                                count += 1
            return count // 2
        yellow_connectivity = count_neighbors(env.board, 'yellow')
        red_connectivity = count_neighbors(env.board, 'red')
        connectivity_diff = yellow_connectivity - red_connectivity

        score = piece_diff * 1.0 + move_diff * 0.5 + center_diff * 0.3 + connectivity_diff * 0.2
        return score if player == 'yellow' else -score
    except Exception as e:
        raise RuntimeError(f"Error in evaluate_board: {type(e).__name__}: {str(e)}")

def minimax_evaluation(env: AtaxxEnvironment, player: str, depth: int, alpha: float = -float('inf'), beta: float = float('inf')) -> int:
    """Đánh giá trạng thái bàn cờ với Alpha-Beta Pruning."""
    try:
        if depth == 0 or env.is_game_over():
            return evaluate_board(env, player)

        valid_moves = sort_moves(env.get_valid_moves(), env)
        if not valid_moves and not env.is_game_over():
            raise ValueError("No valid moves but game is not over")

        if player == 'yellow':
            best_score = -float('inf')
            for move in valid_moves:
                next_env = env.clone()
                if not isinstance(next_env, AtaxxEnvironment):
                    raise ValueError("Cloned environment is invalid")
                next_env.make_move(move['from'], move['to'])
                score = minimax_evaluation(next_env, 'red', depth - 1, alpha, beta)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        else:
            best_score = float('inf')
            for move in valid_moves:
                next_env = env.clone()
                if not isinstance(next_env, AtaxxEnvironment):
                    raise ValueError("Cloned environment is invalid")
                next_env.make_move(move['from'], move['to'])
                score = minimax_evaluation(next_env, 'yellow', depth - 1, alpha, beta)
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
        
        return best_score
    except Exception as e:
        raise RuntimeError(f"Error in minimax_evaluation: {type(e).__name__}: {str(e)}")

def minimax_rollout(env: AtaxxEnvironment, player: str, depth: int = 2) -> AtaxxEnvironment:
    """Mô phỏng trò chơi sử dụng Minimax."""
    if not isinstance(env, AtaxxEnvironment):
        raise ValueError("Invalid environment: must be AtaxxEnvironment")
    if player not in ['yellow', 'red']:
        raise ValueError("Invalid player: must be 'yellow' or 'red'")
    if depth < 1:
        raise ValueError("Depth must be at least 1")

    try:
        current_env = env.clone()
        if not isinstance(current_env, AtaxxEnvironment):
            raise ValueError("Cloned environment is invalid")
        current_player = player

        while not current_env.is_game_over():
            moves = sort_moves(current_env.get_valid_moves(), current_env)
            if not moves:
                current_env.current_player = 'red' if current_env.current_player == 'yellow' else 'yellow'
                if not current_env.get_valid_moves() and not current_env.is_game_over():
                    raise ValueError("No valid moves for both players but game is not over")
                continue

            best_move = None
            best_score = -float('inf') if current_player == 'yellow' else float('inf')
            for move in moves:
                next_env = current_env.clone()
                if not isinstance(next_env, AtaxxEnvironment):
                    raise ValueError("Cloned environment is invalid")
                next_env.make_move(move['from'], move['to'])
                if next_env.board == current_env.board:
                    raise RuntimeError(f"Move {move} did not change the board state")
                score = minimax_evaluation(next_env, current_player, depth - 1)
                if (current_player == 'yellow' and score > best_score) or \
                   (current_player == 'red' and score < best_score):
                    best_score = score
                    best_move = move

            if best_move:
                current_env.make_move(best_move['from'], best_move['to'])
            else:
                current_env.current_player = 'red' if current_env.current_player == 'yellow' else 'yellow'

        return current_env

    except Exception as e:
        raise RuntimeError(f"Error in minimax_rollout: {type(e).__name__}: {str(e)}")