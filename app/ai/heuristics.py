import numpy as np
from ataxx_state import AtaxxState, BOARD_SIZE

# Heuristic coefficients
S1, S2, S3, S4 = 10, 5, 2, 3

def heuristic(move, state, player):
    r, c, nr, nc = move
    is_clone = abs(r - nr) <= 1 and abs(c - nc) <= 1
    # C(m, s): Number of opponent pieces captured
    captured = 0
    temp_state = state.copy()
    temp_state.make_move(move)
    captured = np.sum(temp_state.board == player) - np.sum(state.board == player) - (1 if is_clone else 0)
    # A(m, s): Number of allied pieces around destination
    allies = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nnr, nnc = nr + dr, nc + dc
            if 0 <= nnr < BOARD_SIZE and 0 <= nnc < BOARD_SIZE and state.board[nnr][nnc] == player:
                allies += 1
    # B(m): Bonus for Clone
    bonus_clone = 1 if is_clone else 0
    # P(m, s): Penalty for Jump
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
    score = own - opp
    if state.is_game_over():
        winner = state.get_winner()
        if winner == player:
            score += 50 if np.sum(state.board == 0) == 0 else 500
        elif winner == -player:
            score -= 50 if np.sum(state.board == 0) == 0 else 500
    # Normalize to [0, 1]
    return (score + 549) / 1098