import numpy as np
from constants import BOARD_SIZE, S1, S2, S3, S4, ALPHA, BETA

def heuristic(move, state, player):
    r, c, nr, nc = move
    is_clone = abs(r - nr) <= 1 and abs(c - nc) <= 1

    temp_state = state.copy()
    temp_state.make_move(move)
    captured = np.sum(temp_state.board == player) - np.sum(state.board == player) - (1 if is_clone else 0)

    allies = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue

            nnr, nnc = nr + dr, nc + dc
            if 0 <= nnr < BOARD_SIZE and 0 <= nnc < BOARD_SIZE and state.board[nnr][nnc] == player:
                allies += 1

    bonus_clone = 1 if is_clone else 0

    penalty_jump = 0
    if not is_clone:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                nnr, nnc = r + dr, c + dc
                if 0 <= nnr < BOARD_SIZE and 0 <= nnc < BOARD_SIZE and state.board[nnr][nnc] == player:
                    penalty_jump += 1

    return S1 * captured + S2 * allies + S3 * bonus_clone - S4 * penalty_jump

def evaluate(state, player):
    own = np.sum(state.board == player)
    opp = np.sum(state.board == -player)
    total_pieces = own + opp
    piece_diff = own - opp
    max_diff = 49  

    if total_pieces > 0:
        piece_ratio = own / total_pieces  
    else:
        piece_ratio = 0.5  
    normalized_diff = piece_diff / max_diff  
    f_i = sigmoid(normalized_diff, scale=1.0)  

    if state.is_game_over():
        winner = state.get_winner()
        if winner == player:
            b_i = 1.0  
        elif winner == -player:
            b_i = 0.0  
        else:
            b_i = 0.5  
    else:
        b_i = piece_ratio

    w_i = b_i * (ALPHA + BETA * f_i)

    w_i = max(0.0, min(1.0, w_i))

    return w_i

def sigmoid(x, scale=1.0):
    """Hàm sigmoid để chuẩn hóa giá trị về [0, 1]"""
    return 1.0 / (1.0 + np.exp(-x / scale))