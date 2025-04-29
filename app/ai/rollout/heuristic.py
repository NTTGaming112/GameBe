from app.ai.ataxx_env import AtaxxEnvironment

def heuristic_rollout(env: AtaxxEnvironment, player: str) -> AtaxxEnvironment:
    """
    Mô phỏng game với heuristic nâng cao từ domain knowledge trong paper Ataxx.
    """
    if not isinstance(env, AtaxxEnvironment):
        raise ValueError("Invalid environment: must be AtaxxEnvironment")
    if player not in {"yellow", "red"}:
        raise ValueError("Invalid player: must be 'yellow' or 'red'")

    def neighbors(r, c):
        return [(r + dr, c + dc)
                for dr in [-1, 0, 1]
                for dc in [-1, 0, 1]
                if not (dr == 0 and dc == 0) and 0 <= r + dr < 7 and 0 <= c + dc < 7]

    # hệ số từ domain knowledge
    s1, s2, s3, s4 = 1.0, 0.4, 0.7, 0.4

    try:
        current_env = env.clone()

        while not current_env.is_game_over():
            moves = current_env.get_valid_moves()
            if not moves:
                current_env.current_player = "red" if current_env.current_player == "yellow" else "yellow"
                continue

            best_move = None
            best_score = float('-inf')
            scores_before = current_env.calculate_scores()
            cur_player = current_env.current_player

            for move in moves:
                from_row, from_col = move["from"]["row"], move["from"]["col"]
                to_row, to_col = move["to"]["row"], move["to"]["col"]
                is_copy = abs(to_row - from_row) <= 1 and abs(to_col - from_col) <= 1

                # domain-knowledge scoring
                score = 0.0

                # s1: số quân địch bị chiếm tại vị trí đến
                for nr, nc in neighbors(to_row, to_col):
                    cell = current_env.get_cell(nr, nc)
                    if cell and cell != cur_player:
                        score += s1

                # s2: số quân ta xung quanh điểm đến
                score += s2 * sum(1 for nr, nc in neighbors(to_row, to_col)
                                  if current_env.get_cell(nr, nc) == cur_player)

                # s3: thưởng nếu là nước copy
                if is_copy:
                    score += s3
                else:
                    # s4: phạt nếu là jump và điểm đi bị "rỗng"
                    score -= s4 * sum(1 for nr, nc in neighbors(from_row, from_col)
                                      if current_env.get_cell(nr, nc) == cur_player)

                # thêm heuristic phụ: chiếm trung tâm, ưu tiên trung tâm
                distance_to_center = abs(to_row - 3) + abs(to_col - 3)
                score -= distance_to_center * 0.1

                if score > best_score:
                    best_score = score
                    best_move = move

            if best_move:
                current_env.make_move(best_move["from"], best_move["to"])
            else:
                current_env.current_player = "red" if current_env.current_player == "yellow" else "yellow"

        return current_env

    except Exception as e:
        raise RuntimeError(f"Error in heuristic_rollout: {str(e)}")
