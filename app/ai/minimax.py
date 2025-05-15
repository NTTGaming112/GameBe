
def minimax(board, state, depth_minimax=4):
    """Alpha-Beta Minimax với move ordering nâng cao.
    - Mặc định: 4 ply
    - Khi còn 5 ô trống: 5 ply
    - Khi còn 2 ô trống: 6 ply
    
    Tối ưu hóa:
    - Move ordering: sắp xếp theo công thức Si = s1·(số quân địch bị chiếm) + s2·(số quân ta xung quanh ô đích)
      + s3·1{Clone} − s4·(số quân ta quanh ô nguồn nếu Jump), với (s1,s2,s3,s4)=(1.0,0.4,0.7,0.4)
    - Iterative deepening: tăng dần độ sâu tìm kiếm
    - Heuristic thống nhất: E(p) = Nown - Nopp + (50 hoặc 500 khi thắng/thua)
      (dùng chung với Monte Carlo, Alpha-Beta đánh giá sau n nước, Monte Carlo đánh giá khi game kết thúc)
    """
    
    def evaluate_position(state):
        """Hàm đánh giá đơn giản E(p) = Nown - Nopp + giá trị thắng/thua."""
        player = state.player
        opponent = -player
        
        # Trường hợp đặc biệt để tránh chia cho 0
        if state.balls[player] + state.balls[opponent] == 0:
            return 0
            
        # Đếm số quân của mỗi bên
        num_own = state.balls[player]
        num_opp = state.balls[opponent]
        
        # Nếu trò chơi đã kết thúc
        if board.is_gameover(state):
            if num_own > num_opp:  # Thắng
                # Kiểm tra nếu bàn đầy hoặc chưa đầy
                total_pieces = num_own + num_opp
                empty_spaces = 49 - total_pieces  # Bàn cờ 7x7 có 49 ô
                if empty_spaces == 0:  # Bàn đầy
                    return num_own - num_opp + 50
                else:  # Thắng trước khi bàn đầy
                    return num_own - num_opp + 500
            elif num_own < num_opp:  # Thua
                total_pieces = num_own + num_opp
                empty_spaces = 49 - total_pieces
                if empty_spaces == 0:  # Bàn đầy
                    return num_own - num_opp - 50
                else:  # Thua trước khi bàn đầy
                    return num_own - num_opp - 500
            else:  # Hòa
                return 0
        
        # Đánh giá đơn giản: chỉ dựa trên số quân hiện tại
        score = num_own - num_opp
        
        return score
            
    def max_value(state, depth, alpha, beta):
        """Chọn nước đi tốt nhất cho người chơi hiện tại (max)."""
        # Điều kiện dừng
        if depth == 0 or board.is_gameover(state):
            return evaluate_position(state), None
            
        moves = board.legal_plays(state)
        if not moves:
            return evaluate_position(state), None
            
        # Đánh giá và sắp xếp nước đi theo công thức heuristic mới
        # Si = + s1 · (số quân địch bị chiếm) + s2 · (số quân ta xung quanh ô đích)
        #     + s3 · 1{Clone} − s4 · (số quân ta quanh ô nguồn nếu Jump)
        s1, s2, s3, s4 = 1.0, 0.4, 0.7, 0.4
        ordered_moves = []
        for move in moves:
            # Mô phỏng nước đi để đếm quân bị lật
            next_state = board.next_state(state, move)
            stones_taken = next_state.balls[state.player] - state.balls[state.player]
            
            # Xác định loại nước đi (Clone hay Jump)
            is_clone = move[0] == 'c'
            
            # Đếm số quân ta xung quanh ô đích
            target_pos = move[1]
            own_stones_around_target = 0
            pos_around = [(-1,1),(0,1),(1,1),(-1,0),(1,0),(-1,-1),(0,-1),(1,-1)]
            for dx, dy in pos_around:
                x, y = target_pos[0] + dx, target_pos[1] + dy
                if 0 <= x < 7 and 0 <= y < 7 and state.board[x][y] == state.player:
                    own_stones_around_target += 1
            
            # Đếm số quân ta xung quanh ô nguồn (nếu là Jump)
            own_stones_around_source = 0
            if not is_clone:  # Nếu là Jump
                source_pos = move[2]
                for dx, dy in pos_around:
                    x, y = source_pos[0] + dx, source_pos[1] + dy
                    if 0 <= x < 7 and 0 <= y < 7 and state.board[x][y] == state.player:
                        own_stones_around_source += 1
            
            # Tính điểm cho nước đi theo công thức
            move_score = (s1 * stones_taken) + (s2 * own_stones_around_target)
            if is_clone:
                move_score += s3
            else:
                move_score -= s4 * own_stones_around_source
            
            # Đảm bảo điểm không âm
            move_score = max(0, move_score)
            
            ordered_moves.append((move, move_score))
            
        # Sắp xếp giảm dần theo điểm đánh giá nước đi
        ordered_moves.sort(key=lambda x: x[1], reverse=True)
        
        best_score = float('-inf')
        best_move = None
        
        for move, _ in ordered_moves:
            next_state = board.next_state(state, move)
            score, _ = min_value(next_state, depth - 1, alpha, beta)
            
            if score > best_score:
                best_score = score
                best_move = move
                
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # Beta cutoff
                
        return best_score, best_move

    def min_value(state, depth, alpha, beta):
        """Chọn nước đi tốt nhất cho đối thủ (min)."""
        # Điều kiện dừng: đạt độ sâu tối đa hoặc trò chơi kết thúc
        if depth == 0 or board.is_gameover(state):
            return evaluate_position(state), None
            
        moves = board.legal_plays(state)
        if not moves:
            return evaluate_position(state), None
        
        # Đánh giá và sắp xếp nước đi theo công thức heuristic mới
        # Si = + s1 · (số quân địch bị chiếm) + s2 · (số quân ta xung quanh ô đích)
        #     + s3 · 1{Clone} − s4 · (số quân ta quanh ô nguồn nếu Jump)
        s1, s2, s3, s4 = 1.0, 0.4, 0.7, 0.4
        ordered_moves = []
        for move in moves:
            # Mô phỏng nước đi để đếm quân bị lật
            next_state = board.next_state(state, move)
            stones_taken = next_state.balls[state.player] - state.balls[state.player]
            
            # Xác định loại nước đi (Clone hay Jump)
            is_clone = move[0] == 'c'
            
            # Đếm số quân ta xung quanh ô đích
            target_pos = move[1]
            own_stones_around_target = 0
            pos_around = [(-1,1),(0,1),(1,1),(-1,0),(1,0),(-1,-1),(0,-1),(1,-1)]
            for dx, dy in pos_around:
                x, y = target_pos[0] + dx, target_pos[1] + dy
                if 0 <= x < 7 and 0 <= y < 7 and state.board[x][y] == state.player:
                    own_stones_around_target += 1
            
            # Đếm số quân ta xung quanh ô nguồn (nếu là Jump)
            own_stones_around_source = 0
            if not is_clone:  # Nếu là Jump
                source_pos = move[2]
                for dx, dy in pos_around:
                    x, y = source_pos[0] + dx, source_pos[1] + dy
                    if 0 <= x < 7 and 0 <= y < 7 and state.board[x][y] == state.player:
                        own_stones_around_source += 1
            
            # Tính điểm cho nước đi theo công thức
            move_score = (s1 * stones_taken) + (s2 * own_stones_around_target)
            if is_clone:
                move_score += s3
            else:
                move_score -= s4 * own_stones_around_source
            
            # Đảm bảo điểm không âm
            move_score = max(0, move_score)
            
            ordered_moves.append((move, move_score))
        
        # Sắp xếp giảm dần theo điểm đánh giá nước đi
        ordered_moves.sort(key=lambda x: x[1], reverse=True)
        
        best_score = float('inf')
        best_move = None
        
        for move, _ in ordered_moves:
            next_state = board.next_state(state, move)
            score, _ = max_value(next_state, depth - 1, alpha, beta)
            
            if score < best_score:
                best_score = score
                best_move = move
                
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # Cắt tỉa Alpha
                
        return best_score, best_move

    # Tính số ô trống trên bàn cờ
    total_pieces = state.balls[1] + state.balls[-1]
    empty_spaces = 49 - total_pieces  # Bàn cờ 7x7 có 49 ô
    
    # Điều chỉnh độ sâu dựa trên số ô trống
    if empty_spaces <= 2:
        max_depth = 6  # 6 ply khi còn 2 ô trống hoặc ít hơn
    elif empty_spaces <= 5:
        max_depth = 5  # 5 ply khi còn 3-5 ô trống
    else:
        max_depth = min(depth_minimax, 4)  # Mặc định là 4 ply
    
    # Áp dụng iterative deepening để có kết quả nhanh hơn
    best_move = None
    for current_depth in range(1, max_depth + 1):
        _, move = max_value(state, current_depth, float('-inf'), float('inf'))
        if move:
            best_move = move
    
    return best_move