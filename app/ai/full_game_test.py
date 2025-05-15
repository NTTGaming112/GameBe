#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to run a full game between different AI algorithms.
This script allows for:
1. One-on-one comparisons of any two algorithms
2. Running multiple games to get statistical results
3. Starting from a half-filled board or standard initial positions
"""
import argparse
import time
import numpy as np
import sys
from app.ai.ataxx_state import Ataxx
from app.ai.board import StateMinimax, Board
from app.ai.minimax import minimax
from app.ai.monte_carlo import get_monte_carlo_player

def setup_default_board():
    """Create a standard Ataxx game with initial positions."""
    return Ataxx()

def setup_half_filled_board():
    """Create an Ataxx game with a half-filled board configuration."""
    game = Ataxx()
    # Clear the default setup
    game.board = [[0 for x in range(game.n_fields)] for y in range(game.n_fields)]
    game.balls = {1: 0, -1: 0}
    
    # Create a specific half-filled board configuration
    # For a 7x7 board, let's define a precise layout
    # 0 = empty, 1 = white, -1 = black
    board_layout = [
        [ 1,  1,  1,  0, -1,  0, -1],
        [ 1,  1,  1,  1,  0, -1, -1],
        [ 1,  0,  1,  0, -1, -1,  0],
        [ 1,  1,  1,  1,  0, -1, -1],
        [ 0, -1, -1, -1,  0,  0,  1],
        [-1, -1,  0, -1,  0,  1,  1],
        [-1,  0,  0,  0,  1,  1,  1]
    ]
    
    # Apply the layout to the game
    white_count = 0
    black_count = 0
    
    for x in range(game.n_fields):
        for y in range(game.n_fields):
            piece = board_layout[x][y]
            game.board[x][y] = piece
            if piece == 1:
                white_count += 1
            elif piece == -1:
                black_count += 1
    
    # Update the piece counts
    game.balls[1] = white_count
    game.balls[-1] = black_count
    
    # Set first player to white (1)
    game.turn_player = 1
    
    return game

def get_ai_move(game, algo_type, depth_minimax=4, num_simulations=300, switch_threshold=31):
    """Get a move from the specified AI algorithm."""
    if algo_type == "Minimax":
        board = Board()
        state = StateMinimax(game.board, game.current_player(), game.balls)
        start_time = time.time()
        move = minimax(board, state, depth_minimax)
        elapsed = time.time() - start_time
        return move, elapsed
    else:
        # Monte Carlo algorithms
        start_time = time.time()
        mc = get_monte_carlo_player(
            game, 
            algo_type, 
            number_simulations=num_simulations,
            switch_threshold=switch_threshold,
            use_simulation_formula=False
        )
        move = mc.get_play()
        elapsed = time.time() - start_time
        return move, elapsed

def run_full_game(algo1="Minimax", algo2="MC", half_filled=True, max_moves=None, 
                  depth_minimax=4, num_simulations=300, switch_threshold=31, verbose=True):
    """Run a full game between two AI algorithms."""
    # Set up the initial board
    if half_filled:
        game = setup_half_filled_board()
    else:
        game = setup_default_board()
    
    # Initialize game stats
    move_count = 0
    algo1_times = []
    algo2_times = []
    game_log = []
    
    if verbose:
        print(f"\n=== Starting Game: {algo1} vs {algo2} ===")
        print(f"Initial board:")
        print(game.show_board())
    
    # Main game loop
    while not game.is_game_over() and (max_moves is None or move_count < max_moves):
        current_player = game.current_player()
        
        # Determine which algorithm to use
        if (current_player == 1):  # White's turn
            algo = algo1
            algo_name = f"White ({algo1})"
        else:  # Black's turn
            algo = algo2
            algo_name = f"Black ({algo2})"
        
        if verbose:
            print(f"\nMove {move_count + 1}: {algo_name}'s turn")
        
        # Get move from AI
        move, elapsed = get_ai_move(
            game, 
            algo, 
            depth_minimax=depth_minimax,
            num_simulations=num_simulations,
            switch_threshold=switch_threshold
        )
        
        # Record timing
        if algo == algo1:
            algo1_times.append(elapsed)
        else:
            algo2_times.append(elapsed)
        
        if verbose:
            print(f"Move: {move}")
            print(f"Time taken: {elapsed:.4f} seconds")
        
        # Apply the move
        if move:
            game.move_with_position(move)
            game_log.append((current_player, move))
            
            if verbose:
                print("\nBoard after move:")
                print(game.show_board())
        else:
            if verbose:
                print("No valid move found, skipping turn.")
        
        # Toggle player
        game.toggle_player()
        move_count += 1
    
    # Get the game result
    winner = game.get_winner()
    winner_name = ""
    
    if winner == 1:
        winner_name = f"White ({algo1})"
    elif winner == -1:
        winner_name = f"Black ({algo2})"
    else:
        winner_name = "Draw"
    
    if verbose:
        print("\n=== Game Over ===")
        print(f"Total moves: {move_count}")
        print(f"Winner: {winner_name}")
        print(f"Final score: White {game.balls[1]} - Black {game.balls[-1]}")
        print(f"Average time per move for {algo1}: {np.mean(algo1_times) if algo1_times else 0:.4f} seconds")
        print(f"Average time per move for {algo2}: {np.mean(algo2_times) if algo2_times else 0:.4f} seconds")
    
    # Return game statistics
    return {
        "winner": winner,
        "winner_name": winner_name,
        "total_moves": move_count,
        "white_score": game.balls[1],
        "black_score": game.balls[-1],
        f"{algo1}_avg_time": np.mean(algo1_times) if algo1_times else 0,
        f"{algo2}_avg_time": np.mean(algo2_times) if algo2_times else 0,
        "game_log": game_log
    }

def run_multiple_games(num_games=5, algo1="Minimax", algo2="MC", half_filled=True, max_moves=None,
                       depth_minimax=4, num_simulations=300, switch_threshold=31, verbose=True):
    """Run multiple games between two AI algorithms and collect statistics."""
    print(f"\n=== Running {num_games} games: {algo1} vs {algo2} ===")
    print(f"Settings:")
    print(f"- Board: {'Half-filled' if half_filled else 'Default'}")
    print(f"- Max moves per game: {'Unlimited' if max_moves is None else max_moves}")
    print(f"- Minimax depth: {depth_minimax}")
    print(f"- Simulations per move: {num_simulations}")
    print(f"- Switch threshold: {switch_threshold}\n")
    
    # Collect statistics
    results = []
    algo1_wins = 0
    algo2_wins = 0
    draws = 0
    total_moves = 0
    algo1_total_time = 0
    algo2_total_time = 0
    
    for i in range(num_games):
        print(f"Starting game {i+1}/{num_games}...")
        
        # Run a game, alternating who plays white
        swap_sides = i % 2 == 1
        
        if swap_sides:
            game_algo1, game_algo2 = algo2, algo1
            print(f"Swapping sides: {game_algo1} as White, {game_algo2} as Black")
        else:
            game_algo1, game_algo2 = algo1, algo2
            print(f"{game_algo1} as White, {game_algo2} as Black")
        
        result = run_full_game(
            algo1=game_algo1,
            algo2=game_algo2,
            half_filled=half_filled,
            max_moves=max_moves,
            depth_minimax=depth_minimax,
            num_simulations=num_simulations,
            switch_threshold=switch_threshold,
            verbose=verbose
        )
        
        # Update statistics, accounting for side swapping
        if result["winner"] == 1:  # White won
            if swap_sides:
                algo2_wins += 1
            else:
                algo1_wins += 1
        elif result["winner"] == -1:  # Black won
            if swap_sides:
                algo1_wins += 1
            else:
                algo2_wins += 1
        else:
            draws += 1
        
        # Accumulate stats
        total_moves += result["total_moves"]
        results.append(result)
        
        # Accumulate times (account for swapping)
        if swap_sides:
            algo1_total_time += result[f"{algo2}_avg_time"] * result["total_moves"]
            algo2_total_time += result[f"{algo1}_avg_time"] * result["total_moves"]
        else:
            algo1_total_time += result[f"{algo1}_avg_time"] * result["total_moves"]
            algo2_total_time += result[f"{algo2}_avg_time"] * result["total_moves"]
        
        print(f"Game {i+1} result: {result['winner_name']} wins.")
        print(f"Current standings: {algo1}: {algo1_wins}, {algo2}: {algo2_wins}, Draws: {draws}\n")
    
    # Calculate final statistics
    avg_moves_per_game = total_moves / num_games
    algo1_avg_time = algo1_total_time / total_moves if total_moves > 0 else 0
    algo2_avg_time = algo2_total_time / total_moves if total_moves > 0 else 0
    
    # Print summary
    print("\n=== Final Results ===")
    print(f"Games played: {num_games}")
    print(f"{algo1} wins: {algo1_wins} ({algo1_wins/num_games*100:.1f}%)")
    print(f"{algo2} wins: {algo2_wins} ({algo2_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print(f"Average moves per game: {avg_moves_per_game:.1f}")
    print(f"Average time per move for {algo1}: {algo1_avg_time:.4f} seconds")
    print(f"Average time per move for {algo2}: {algo2_avg_time:.4f} seconds")
    
    return {
        "games_played": num_games,
        f"{algo1}_wins": algo1_wins,
        f"{algo2}_wins": algo2_wins,
        "draws": draws,
        "avg_moves_per_game": avg_moves_per_game,
        f"{algo1}_avg_time": algo1_avg_time,
        f"{algo2}_avg_time": algo2_avg_time,
        "detailed_results": results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run full games between AI algorithms')
    parser.add_argument('--algo1', type=str, default="Minimax", 
                    choices=["Minimax", "MC", "MCD", "MCTS", "AB+MCD"],
                    help='First algorithm to use')
    parser.add_argument('--algo2', type=str, default="MC", 
                    choices=["Minimax", "MC", "MCD", "MCTS", "AB+MCD"],
                    help='Second algorithm to use')
    parser.add_argument('--games', type=int, default=2,
                    help='Number of games to play')
    parser.add_argument('--half-filled', action='store_true', default=True,
                    help='Start with a half-filled board')
    parser.add_argument('--max-moves', type=int, default=None,
                    help='Maximum moves per game (None for unlimited)')
    parser.add_argument('--depth', type=int, default=4,
                    help='Depth for minimax search')
    parser.add_argument('--simulations', type=int, default=300,
                    help='Number of simulations per move for Monte Carlo')
    parser.add_argument('--threshold', type=int, default=31,
                    help='Threshold for AB+MCD algorithm')
    parser.add_argument('--verbose', action='store_true', default=True,
                    help='Print detailed game information')
    
    args = parser.parse_args()
    
    run_multiple_games(
        num_games=args.games,
        algo1=args.algo1,
        algo2=args.algo2,
        half_filled=args.half_filled,
        max_moves=args.max_moves,
        depth_minimax=args.depth,
        num_simulations=args.simulations,
        switch_threshold=args.threshold,
        verbose=args.verbose
    )
