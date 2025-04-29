from app.ai.ataxx_env import AtaxxEnvironment

transposition_table = {}

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

def minimax_evaluation(env: AtaxxEnvironment, player: str, depth: int, alpha: float = -float('inf'), beta: float = float('inf')) -> int:
    board_hash = str(env.board) + player + str(depth)
    if board_hash in transposition_table:
        return transposition_table[board_hash]
    
    if depth == 0 or env.is_game_over():
        score = evaluate_board(env, player)
        transposition_table[board_hash] = score
        return score

    valid_moves = sort_moves(env.get_valid_moves(), env)
    if player == 'yellow':
        best_score = -float('inf')
        for move in valid_moves:
            next_env = env.clone()
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
            next_env.make_move(move['from'], move['to'])
            score = minimax_evaluation(next_env, 'yellow', depth - 1, alpha, beta)
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break
    
    transposition_table[board_hash] = best_score
    return best_score

def get_dynamic_depth(env: AtaxxEnvironment) -> int:
    total_pieces = sum(row.count('yellow') + row.count('red') for row in env.board)
    if total_pieces < 20:
        return 2
    elif total_pieces < 40:
        return 3
    else:
        return 4

def minimax_rollout(env: AtaxxEnvironment, player: str, depth: int = None) -> AtaxxEnvironment:
    if not isinstance(env, AtaxxEnvironment):
        raise ValueError("Invalid environment: must be AtaxxEnvironment")
    if player not in ['yellow', 'red']:
        raise ValueError("Invalid player: must be 'yellow' or 'red'")

    try:
        current_env = env.clone()
        while not current_env.is_game_over():
            depth = get_dynamic_depth(current_env) if depth is None else depth
            moves = sort_moves(current_env.get_valid_moves(), current_env)
            if not moves:
                current_env.current_player = 'red' if current_env.current_player == 'yellow' else 'yellow'
                continue

            best_move = None
            best_score = -float('inf') if current_env.current_player == 'yellow' else float('inf')
            for move in moves:
                next_env = current_env.clone()
                next_env.make_move(move['from'], move['to'])
                score = minimax_evaluation(next_env, 'red' if current_env.current_player == 'yellow' else 'yellow', depth - 1)
                if (current_env.current_player == 'yellow' and score > best_score) or \
                   (current_env.current_player == 'red' and score < best_score):
                    best_score = score
                    best_move = move

            if best_move:
                current_env.make_move(best_move['from'], best_move['to'])
            else:
                current_env.current_player = 'red' if current_env.current_player == 'yellow' else 'yellow'

        return current_env

    except Exception as e:
        raise RuntimeError(f"Error in minimax_rollout: {str(e)}")