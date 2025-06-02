#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to run a full game between different AI algorithms.
"""
import argparse
import time
import numpy as np
from app.ai.ataxx_state import Ataxx
from app.ai.monte_carlo import get_monte_carlo_player
from app.ai.board_layouts import get_layout

def setup_default_board():
    return Ataxx()

def setup_custom_board(layout_name="map1"):
    game = Ataxx()
    game.player1_board = 0
    game.player2_board = 0
    game.balls = {1: 0, -1: 0}
    
    board_layout = get_layout(layout_name)
    
    white_count = 0
    black_count = 0
    
    for x in range(game.n_fields):
        for y in range(game.n_fields):
            piece = board_layout[x][y]
            bit = x * game.n_fields + y
            if piece == 1:
                game.player1_board |= (1 << bit)
                white_count += 1
            elif piece == -1:
                game.player2_board |= (1 << bit)
                black_count += 1
    
    game.balls[1] = white_count
    game.balls[-1] = black_count
    game.turn_player = 1
    game.update_position_history()
    
    return game

def get_ai_move(game, algo_type, depth_minimax=4, num_simulations=300, switch_threshold=13,
               s1_ratio=1.0, s2_ratio=1.0, s3_ratio=0.5, time_limit=None):
    start_time = time.time()
    mc = get_monte_carlo_player(
        game, 
        algo_type, 
        number_simulations=num_simulations,
        s1_ratio=s1_ratio,
        s2_ratio=s2_ratio,
        s3_ratio=s3_ratio,
        time_limit=time_limit,
        switch_threshold=switch_threshold,
        minimax_depth=depth_minimax
    )
    if algo_type == "Minimax":
        move = mc.get_move(time_limit)
    elif algo_type in ["MCD", "AB+MCD"]:
        move = mc.get_mcts_move(time_limit)
    else:
        move = mc.get_move()
    elapsed = time.time() - start_time

    return move, elapsed

def run_full_game(algo1="Minimax", algo2="MCD", max_moves=None, 
                  depth_minimax=4, num_simulations=300, switch_threshold=13, 
                  s1_ratio=1.0, s2_ratio=1.0, s3_ratio=0.5, time_limit=None, verbose=True, layout=None):
    if layout:
        game = setup_custom_board(layout)
    else:
        game = setup_default_board()
    
    move_count = 0
    algo1_times = []
    algo2_times = []
    game_log = []
    consecutive_none_moves = 0  # Track consecutive None moves
    
    if verbose:
        print(f"\n=== Starting Game: {algo1} vs {algo2} ===")
        print(f"Initial board:")
        game.print_board()
    
    while not game.is_game_over() and (max_moves is None or move_count < max_moves) and consecutive_none_moves < 2:
        current_player = game.current_player()
        
        algo = algo1 if current_player == 1 else algo2
        algo_name = f"White ({algo1})" if current_player == 1 else f"Black ({algo2})"
        
        if verbose:
            print(f"\nMove {move_count + 1}: {algo_name}'s turn")
        
        move, elapsed = get_ai_move(
            game, 
            algo, 
            depth_minimax=depth_minimax,
            num_simulations=num_simulations,
            switch_threshold=switch_threshold,
            s1_ratio=s1_ratio,
            s2_ratio=s2_ratio,
            s3_ratio=s3_ratio,
            time_limit=time_limit
        )
        
        if current_player == 1:
            algo1_times.append(elapsed)
        else:
            algo2_times.append(elapsed)
        
        if verbose:
            print(f"Move: {move}")
            print(f"Time taken: {elapsed:.4f} seconds")
        
        if move:
            game.move_with_position(move)
            game_log.append((current_player, move))
            consecutive_none_moves = 0  # Reset counter on valid move
            
            if verbose:
                print("\nBoard after move:")
                game.print_board()
        else:
            consecutive_none_moves += 1  # Increment counter for None move
            if verbose:
                print("No valid move found, skipping turn.")
                if consecutive_none_moves >= 2:
                    print("Two consecutive players with no moves - game should end.")
            # Only toggle player when no move is available, as move_with_position() already handles it for valid moves
            game.toggle_player()
        
        move_count += 1
    
    winner = game.get_winner()
    winner_name = "White" if winner == 1 else "Black" if winner == -1 else "Draw"
    
    if verbose:
        print("\n=== Game Over ===")
        print(f"Total moves: {move_count}")
        print(f"Winner: {winner_name}")
        print(f"Final score: White {game.balls[1]} - Black {game.balls[-1]}")
        print(f"Average time per move for {algo1}: {np.mean(algo1_times) if algo1_times else 0:.4f} seconds")
        print(f"Average time per move for {algo2}: {np.mean(algo2_times) if algo2_times else 0:.4f} seconds")
    
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

def run_multiple_games(num_games=5, algo1="Minimax", algo2="MCD", max_moves=None,
                       depth_minimax=4, num_simulations=300, switch_threshold=13, 
                       s1_ratio=1.0, s2_ratio=1.0, s3_ratio=0.5, time_limit=None, verbose=True,
                       layout=None):
    print(f"\n=== Running {num_games} games: {algo1} vs {algo2} ===")
    print(f"Settings:")
    board_type = f"Custom layout: {layout}" if layout else "Standard"
    print(f"- Board: {board_type}")
    print(f"- Max moves per game: {'Unlimited' if max_moves is None else max_moves}")
    print(f"- Minimax depth: {depth_minimax}")
    print(f"- Simulations per move: {num_simulations}")
    print(f"- Tournament sizes ratio (S1:S2:S3): {s1_ratio}:{s2_ratio}:{s3_ratio}")
    print(f"- Switch threshold: {switch_threshold}")
    print(f"- Time limit per move: {time_limit if time_limit is not None else 'None'}\n")
    
    results = []
    algo1_wins = 0
    algo2_wins = 0
    draws = 0
    total_moves = 0
    algo1_total_time = 0
    algo2_total_time = 0
    
    for i in range(num_games):
        print(f"Starting game {i+1}/{num_games}...")
        swap_sides = i % 2 == 0
        
        game_algo1, game_algo2 = (algo1, algo2) if swap_sides else (algo2, algo1)
        print(f"{game_algo1} as White, {game_algo2} as Black")
        
        result = run_full_game(
            algo1=game_algo1,
            algo2=game_algo2,
            max_moves=max_moves,
            depth_minimax=depth_minimax,
            num_simulations=num_simulations,
            switch_threshold=switch_threshold,
            s1_ratio=s1_ratio,
            s2_ratio=s2_ratio,
            s3_ratio=s3_ratio,
            time_limit=time_limit,
            verbose=verbose,
            layout=layout,
        )
        
        if result["winner"] == 1:
            if swap_sides:
                algo1_wins += 1
            else:
                algo2_wins += 1
        elif result["winner"] == -1:
            if swap_sides:
                algo2_wins += 1
            else:
                algo1_wins += 1
        else:
            draws += 1
        
        total_moves += result["total_moves"]
        if swap_sides:
            algo1_total_time += result[f"{game_algo1}_avg_time"] * result["total_moves"]
            algo2_total_time += result[f"{game_algo2}_avg_time"] * result["total_moves"]
        else:
            algo1_total_time += result[f"{game_algo2}_avg_time"] * result["total_moves"]
            algo2_total_time += result[f"{game_algo1}_avg_time"] * result["total_moves"]
        
        print(f"Game {i+1} result: {result['winner_name']} wins.")
        print(f"Current standings: {algo1}: {algo1_wins}, {algo2}: {algo2_wins}, Draws: {draws}\n")
    
    avg_moves_per_game = total_moves / num_games
    algo1_avg_time = algo1_total_time / total_moves if total_moves > 0 else 0
    algo2_avg_time = algo2_total_time / total_moves if total_moves > 0 else 0
    
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
                        choices=["Minimax", "MC", "MCD", "AB+MCD"],
                        help='First algorithm to use')
    parser.add_argument('--algo2', type=str, default="MCD", 
                        choices=["Minimax", "MC", "MCD", "AB+MCD"],
                        help='Second algorithm to use')
    parser.add_argument('--games', type=int, default=2,
                        help='Number of games to play')
    parser.add_argument('--max-moves', type=int, default=None,
                        help='Maximum moves per game (None for unlimited)')
    parser.add_argument('--depth', type=int, default=4,
                        help='Depth for minimax search')
    parser.add_argument('--simulations', type=int, default=300,
                        help='Number of simulations per move for Monte Carlo')
    parser.add_argument('--threshold', type=int, default=13,
                        help='Switch threshold for AB+MCD')
    parser.add_argument('--s1-ratio', type=float, default=1.0,
                        help='Ratio for heuristic component weight')
    parser.add_argument('--s2-ratio', type=float, default=1.0,
                        help='Ratio for tactical component weight')
    parser.add_argument('--s3-ratio', type=float, default=0.5,
                        help='Ratio for strategic component weight')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed game information')
    parser.add_argument('--time-limit', type=float, default=None,
                        help='Time limit per move in seconds (None for unlimited)')
    parser.add_argument('--layout', type=str, default=None,
                        help='Board layout to use (map1, etc.)')
    
    args = parser.parse_args()
    
    run_multiple_games(
        num_games=args.games,
        algo1=args.algo1,
        algo2=args.algo2,
        max_moves=args.max_moves,
        depth_minimax=args.depth,
        num_simulations=args.simulations,
        switch_threshold=args.threshold,
        s1_ratio=args.s1_ratio,
        s2_ratio=args.s2_ratio,
        s3_ratio=args.s3_ratio,
        time_limit=args.time_limit,
        layout=args.layout,
        verbose=args.verbose
    )