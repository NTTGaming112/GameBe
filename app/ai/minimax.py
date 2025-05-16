from app.ai.constants import BOARD_SIZE, BOARD_TOTAL_CELLS, CLONE_MOVE, WIN_BONUS_FULL_BOARD, WIN_BONUS_EARLY, MOVE_WEIGHTS, ADJACENT_POSITIONS

def minimax(board, state, depth_minimax=4):
    """Alpha-Beta Minimax with advanced move ordering.
    
    Arguments:
        board: Game board instance
        state: Current game state
        depth_minimax: Maximum search depth (default: 4 ply)
    
    Optimizations:
        - Move ordering: prioritizes moves based on capturing, surroundings and move type
        - Iterative deepening: gradually increases search depth
        - Unified evaluation function across different AI implementations
        
    Returns:
        Best move found by the algorithm
    """
    
    def evaluate_position(state):
        """Evaluates board position using formula: E(p) = Nown - Nopp + win/loss bonus."""
        player = state.player
        opponent = -player
        
        # Special case to avoid division by zero
        if state.balls[player] + state.balls[opponent] == 0:
            return 0
            
        # Count pieces for each player
        num_own = state.balls[player]
        num_opp = state.balls[opponent]
        score_diff = num_own - num_opp
        
        # Game over evaluation with win/loss bonuses
        if board.is_gameover(state):
            if num_own > num_opp:  # Win
                total_pieces = num_own + num_opp
                is_full_board = (total_pieces == BOARD_TOTAL_CELLS)
                return score_diff + (WIN_BONUS_FULL_BOARD if is_full_board else WIN_BONUS_EARLY)
            elif num_own < num_opp:  # Loss
                total_pieces = num_own + num_opp
                is_full_board = (total_pieces == BOARD_TOTAL_CELLS)
                return score_diff - (WIN_BONUS_FULL_BOARD if is_full_board else WIN_BONUS_EARLY)
            else:  # Draw
                return 0
        
        # Basic evaluation for non-terminal positions
        return score_diff
    
    def count_adjacent_pieces(position, state, player):
        """Counts player's pieces adjacent to the given position."""
        count = 0
        for dx, dy in ADJACENT_POSITIONS:
            x, y = position[0] + dx, position[1] + dy
            if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and state.board[x][y] == player:
                count += 1
        return count
    
    def order_moves(moves, state):
        """Orders moves using heuristic scoring for better pruning efficiency."""
        ordered_moves = []
        
        for move in moves:
            # Simulate move to count captured pieces
            next_state = board.next_state(state, move)
            stones_taken = next_state.balls[state.player] - state.balls[state.player]
            
            # Determine move type (Clone or Jump)
            is_clone = move[0] == CLONE_MOVE
            target_pos = move[1]
            
            # Count player pieces around target position
            own_stones_around_target = count_adjacent_pieces(target_pos, state, state.player)
            
            # Calculate move score
            move_score = (MOVE_WEIGHTS["capture"] * stones_taken + 
                          MOVE_WEIGHTS["target_surroundings"] * own_stones_around_target)
            
            if is_clone:
                move_score += MOVE_WEIGHTS["clone_bonus"]
            else:  # Jump move
                source_pos = move[2]
                own_stones_around_source = count_adjacent_pieces(source_pos, state, state.player)
                move_score -= MOVE_WEIGHTS["jump_penalty"] * own_stones_around_source
            
            # Ensure non-negative score
            move_score = max(0, move_score)
            ordered_moves.append((move, move_score))
        
        # Sort moves by score (highest first)
        return sorted(ordered_moves, key=lambda x: x[1], reverse=True)
            
    def max_value(state, depth, alpha, beta):
        """Maximizing player function for minimax algorithm."""
        # Terminal conditions
        if depth == 0 or board.is_gameover(state):
            return evaluate_position(state), None
            
        moves = board.legal_plays(state)
        if not moves:
            return evaluate_position(state), None
        
        # Order moves for better pruning efficiency
        ordered_moves = order_moves(moves, state)
        
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
        """Minimizing player function for minimax algorithm."""
        # Terminal conditions
        if depth == 0 or board.is_gameover(state):
            return evaluate_position(state), None
            
        moves = board.legal_plays(state)
        if not moves:
            return evaluate_position(state), None
        
        # Order moves for better pruning efficiency
        ordered_moves = order_moves(moves, state)
        
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
                break  # Alpha cutoff
                
        return best_score, best_move

    # Dynamic depth adjustment based on game state
    total_pieces = state.balls[1] + state.balls[-1]
    empty_spaces = BOARD_TOTAL_CELLS - total_pieces
    
    # Choose search depth based on game progression
    if empty_spaces <= 2:
        max_depth = 6  # 6-ply when 2 or fewer empty spaces
    elif empty_spaces <= 5:
        max_depth = 5  # 5-ply when 3-5 empty spaces
    else:
        max_depth = min(depth_minimax, 4)  # Default: 4-ply
    
    # Apply iterative deepening for faster results on shallow depths
    best_move = None
    for current_depth in range(1, max_depth + 1):
        _, move = max_value(state, current_depth, float('-inf'), float('inf'))
        if move:
            best_move = move
    
    return best_move