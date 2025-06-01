from app.ai.constants import (BOARD_SIZE, PLAYER_ONE, PLAYER_TWO, EMPTY_CELL, REPEAT_THRESHOLD)

class Ataxx:
    def __init__(self):
        self.n_fields = BOARD_SIZE
        self.balls = {PLAYER_ONE: 2, PLAYER_TWO: 2}
        self.moves = {PLAYER_ONE: 0, PLAYER_TWO: 0}
        self.turn_player = PLAYER_ONE
        self.player1_board = (1 << 0) | (1 << 48)  # (0,0) and (6,6) for Player 1
        self.player2_board = (1 << 6) | (1 << 42)  # (0,6) and (6,0) for Player 2
        self.stop_game = False
        self.position_history = {}
        self.move_cache = None
        self.move_cache_player = None
        self.update_position_history()

    def pos_to_bit(self, x, y):
        return x * self.n_fields + y

    def get_all_possible_moves(self):
        if self.move_cache is not None and self.move_cache_player == self.turn_player:
            return self.move_cache
        
        possible_moves = []
        clone_destinations = set()  # Track positions reachable by clone moves
        my_board = self.player1_board if self.turn_player == PLAYER_ONE else self.player2_board
        empty_board = ~(self.player1_board | self.player2_board) & ((1 << 49) - 1)

        # First pass: Find all clone moves (adjacent moves)
        # These have priority over jump moves to the same destination
        for i in range(49):
            if my_board & (1 << i):
                x, y = i // self.n_fields, i % self.n_fields
                
                # Check adjacent positions for clone moves (distance 1)
                for dx, dy in [(-1,1),(0,1),(1,1),(-1,0),(1,0),(-1,-1),(0,-1),(1,-1)]:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < self.n_fields and 0 <= new_y < self.n_fields:
                        new_bit = self.pos_to_bit(new_x, new_y)
                        if empty_board & (1 << new_bit) and (new_x, new_y) not in clone_destinations:
                            # Clone move: from_pos=None indicates duplication
                            possible_moves.append((None, (new_x, new_y)))
                            clone_destinations.add((new_x, new_y))

        # Second pass: Find jump moves (distance 2) 
        # Only add if destination is not already reachable by clone
        for i in range(49):
            if my_board & (1 << i):
                x, y = i // self.n_fields, i % self.n_fields
                
                # Check jump positions (distance 2)
                for dx, dy in [(-2,2),(-2,1),(-2,0),(-2,-1),(-2,-2),
                              (-1,2),(-1,-2),(0,2),(0,-2),(1,2),
                              (1,-2),(2,2),(2,1),(2,0),(2,-1),(2,-2)]:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < self.n_fields and 0 <= new_y < self.n_fields:
                        new_bit = self.pos_to_bit(new_x, new_y)
                        # Only add jump move if position is empty AND not reachable by clone
                        if (empty_board & (1 << new_bit)) and ((new_x, new_y) not in clone_destinations):
                            # Jump move: from_pos indicates source position
                            possible_moves.append(((x, y), (new_x, new_y)))

        self.move_cache = possible_moves
        self.move_cache_player = self.turn_player
        return possible_moves

    def apply_move_with_undo(self, move):
        self.move_cache = None
        undo_info = {
            'player1_board': self.player1_board,
            'player2_board': self.player2_board,
            'balls': self.balls.copy(),
            'turn_player': self.turn_player,
            'moves': self.moves.copy(),
            'position_history': self.position_history.copy(),
            'converted': []
        }
        from_pos, to_pos = move
        to_bit = self.pos_to_bit(*to_pos)
        my_board = self.player1_board if self.turn_player == PLAYER_ONE else self.player2_board
        opp_board = self.player2_board if self.turn_player == PLAYER_ONE else self.player1_board
        if self.turn_player == PLAYER_ONE:
            self.player1_board |= (1 << to_bit)
            if from_pos:
                self.player1_board &= ~(1 << self.pos_to_bit(*from_pos))
                self.balls[PLAYER_ONE] -= 1
            for dx, dy in [
                (-1,1),(0,1),(1,1),(-1,0),(1,0),(-1,-1),(0,-1),(1,-1)
            ]:
                adj_x, adj_y = to_pos[0] + dx, to_pos[1] + dy
                if 0 <= adj_x < self.n_fields and 0 <= adj_y < self.n_fields:
                    adj_bit = self.pos_to_bit(adj_x, adj_y)
                    if self.player2_board & (1 << adj_bit):
                        self.player2_board &= ~(1 << adj_bit)
                        self.player1_board |= (1 << adj_bit)
                        undo_info['converted'].append((adj_x, adj_y))
                        self.balls[PLAYER_ONE] += 1
                        self.balls[PLAYER_TWO] -= 1
            self.balls[PLAYER_ONE] += 1
        else:
            self.player2_board |= (1 << to_bit)
            if from_pos:
                self.player2_board &= ~(1 << self.pos_to_bit(*from_pos))
                self.balls[PLAYER_TWO] -= 1
            for dx, dy in [
                (-1,1),(0,1),(1,1),(-1,0),(1,0),(-1,-1),(0,-1),(1,-1)
            ]:
                adj_x, adj_y = to_pos[0] + dx, to_pos[1] + dy
                if 0 <= adj_x < self.n_fields and 0 <= adj_y < self.n_fields:
                    adj_bit = self.pos_to_bit(adj_x, adj_y)
                    if self.player1_board & (1 << adj_bit):
                        self.player1_board &= ~(1 << adj_bit)
                        self.player2_board |= (1 << adj_bit)
                        undo_info['converted'].append((adj_x, adj_y))
                        self.balls[PLAYER_TWO] += 1
                        self.balls[PLAYER_ONE] -= 1
            self.balls[PLAYER_TWO] += 1
        self.moves[self.turn_player] += 1
        self.turn_player = -self.turn_player
        self.update_position_history()
        return undo_info

    def undo_move(self, undo_info):
        self.player1_board = undo_info['player1_board']
        self.player2_board = undo_info['player2_board']
        self.balls = undo_info['balls'].copy()
        self.turn_player = undo_info['turn_player']
        self.moves = undo_info['moves'].copy()
        self.position_history = undo_info['position_history'].copy()
        self.move_cache = None

    def move_with_position(self, move):
        self.apply_move_with_undo(move)

    def toggle_player(self):
        self.turn_player = -self.turn_player

    def current_player(self):
        return self.turn_player

    def is_game_over(self):
        if self.stop_game:
            return True
        if self.balls[PLAYER_ONE] == 0 or self.balls[PLAYER_TWO] == 0:
            return True
        if self.full_squares():
            return True
        if self.position_repeated_three_times():
            return True
        
        # Check if both players have no valid moves
        current_moves = self.get_all_possible_moves()
        if not current_moves:
            # Current player has no moves, check if opponent also has no moves
            self.toggle_player()
            opponent_moves = self.get_all_possible_moves()
            self.toggle_player()  # Restore original player
            
            if not opponent_moves:
                # Both players have no moves - game over
                return True
        
        return False

    def full_squares(self):
        total_pieces = self.balls[PLAYER_ONE] + self.balls[PLAYER_TWO]
        return total_pieces >= self.n_fields * self.n_fields

    def board_to_tuple(self):
        player1_board = bin(self.player1_board)[2:].zfill(self.n_fields * self.n_fields)
        player2_board = bin(self.player2_board)[2:].zfill(self.n_fields * self.n_fields)
        return (player1_board, player2_board)

    def get_board_array(self):
        """Convert bitboards to a 2D array representation.
        
        Returns:
            list: 2D array where each cell contains PLAYER_ONE, PLAYER_TWO, or EMPTY_CELL
        """
        board = [[EMPTY_CELL for _ in range(self.n_fields)] for _ in range(self.n_fields)]
        for i in range(self.n_fields * self.n_fields):
            x, y = i // self.n_fields, i % self.n_fields
            if self.player1_board & (1 << i):
                board[x][y] = PLAYER_ONE
            elif self.player2_board & (1 << i):
                board[x][y] = PLAYER_TWO
        return board

    def update_position_history(self):
        position_key = (self.board_to_tuple(), self.turn_player)
        self.position_history[position_key] = self.position_history.get(position_key, 0) + 1

    def position_repeated_three_times(self):
        position_key = (self.board_to_tuple(), self.turn_player)
        return self.position_history.get(position_key, 0) >= REPEAT_THRESHOLD

    def get_actual_piece_counts(self):
        """Calculate actual piece counts from bitboards"""
        player1_count = bin(self.player1_board).count('1')
        player2_count = bin(self.player2_board).count('1')
        return {PLAYER_ONE: player1_count, PLAYER_TWO: player2_count}
    
    def sync_balls_with_bitboards(self):
        """Synchronize balls dictionary with actual bitboard counts"""
        actual_counts = self.get_actual_piece_counts()
        self.balls[PLAYER_ONE] = actual_counts[PLAYER_ONE]
        self.balls[PLAYER_TWO] = actual_counts[PLAYER_TWO]
    
    def get_winner(self):
        if self.balls[PLAYER_TWO] > self.balls[PLAYER_ONE]:
            return PLAYER_TWO
        elif self.balls[PLAYER_ONE] > self.balls[PLAYER_TWO]:
            return PLAYER_ONE
        return 100  # Draw

    def get_score(self, player):
        total_pieces = self.balls[PLAYER_ONE] + self.balls[PLAYER_TWO]
        return float(self.balls[player]) / total_pieces if total_pieces > 0 else 0.0

    def get_score_pieces(self, player):
        """Get the board coverage ratio for a player.
        
        Args:
            player: Player to get score for (PLAYER_ONE or PLAYER_TWO)
            
        Returns:
            float: Ratio of player's pieces to total board cells (0 to 1)
        """
        total_cells = self.n_fields * self.n_fields
        return float(self.balls[player]) / total_cells if total_cells > 0 else 0.0

    def update_board(self, state):
        """Update the game state from a StateMinimax object.
        
        Args:
            state: StateMinimax object containing board state, player, and balls
        """
        # Update current player
        self.turn_player = state.player
        
        # Update piece counts
        self.balls[PLAYER_ONE] = state.balls[PLAYER_ONE]
        self.balls[PLAYER_TWO] = state.balls[PLAYER_TWO]
        
        # Convert the 2D board array back to bitboards
        self.player1_board = 0
        self.player2_board = 0
        
        for i in range(self.n_fields):
            for j in range(self.n_fields):
                bit_pos = self.pos_to_bit(i, j)
                if state.board[i][j] == PLAYER_ONE:
                    self.player1_board |= (1 << bit_pos)
                elif state.board[i][j] == PLAYER_TWO:
                    self.player2_board |= (1 << bit_pos)
        
        # Clear move cache since board state changed
        self.move_cache = None
        self.move_cache_player = None

    def print_board(self):
        board = [[EMPTY_CELL for _ in range(self.n_fields)] for _ in range(self.n_fields)]
        for i in range(self.n_fields * self.n_fields):
            x, y = i // self.n_fields, i % self.n_fields
            if self.player1_board & (1 << i):
                board[x][y] = PLAYER_ONE
            elif self.player2_board & (1 << i):
                board[x][y] = PLAYER_TWO
        print(f"Player: {self.turn_player}")
        s = [[str(e) for e in row] for row in board]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
        print(f"White pieces: {self.balls[PLAYER_ONE]} | Black pieces: {self.balls[PLAYER_TWO]}")