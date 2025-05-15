#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main test script to run AI algorithm comparisons
"""
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.ataxx_state import Ataxx
from app.ai.board import StateMinimax, Board
from app.ai.minimax import minimax
from app.ai.monte_carlo import get_monte_carlo_player
from app.ai.binary_mcts import MonteCarloTreeSearch
from app.ai.game_simulation import main as run_simulation

def setup_half_filled_board():
    """
    Create an Ataxx game with a half-filled board configuration.
    This matches the configuration from the provided image.
    """
    game = Ataxx()
    # Clear the default setup
    game.board = [[0 for x in range(game.n_fields)] for y in range(game.n_fields)]
    game.balls = {1: 0, -1: 0}
    
    # Create a specific half-filled board configuration based on the image
    # For a 7x7 board, let's define a precise layout
    # 0 = empty, 1 = white, -1 = black
    board_layout = [
        [ 1,  1,  1,  0,  0,  0, -1],
        [ 1,  1,  0,  0,  0, -1, -1],
        [ 1,  0,  0,  0, -1, -1,  0],
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 0, -1, -1,  0,  0,  0,  1],
        [-1, -1,  0,  0,  0,  1,  1],
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

def test_move(game, algo_type, move_data, number_simulations=200, depth_minimax=4, switch_threshold=31):
    """Test a single move with the given algorithm and record timing/results."""
    game_copy = deepcopy(game)
    
    start_time = time.time()
    
    if algo_type == "Minimax":
        state = StateMinimax(game_copy.board, game_copy.current_player(), game_copy.balls)
        board = Board()
        move = minimax(board, state, depth_minimax)
    else:
        # Monte Carlo algorithms
        if algo_type == "MCTS":
            algo = MonteCarloTreeSearch(game_copy, number_simulations=number_simulations)
        else:
            algo = get_monte_carlo_player(
                game_copy, 
                algo_type, 
                number_simulations=number_simulations,
                switch_threshold=switch_threshold
            )
        move = algo.get_play()
    
    end_time = time.time()
    
    # Record the move and timing data
    move_data[algo_type]["moves"].append(move)
    move_data[algo_type]["times"].append(end_time - start_time)
    
    # Apply the move to the game to return the resulting state
    if move:
        game_copy.move_with_position(move)
        game_copy.toggle_player()
    
    return game_copy, move_data

def compare_algorithms_on_half_filled():
    """
    Compare all algorithms on a half-filled board configuration.
    Tests each algorithm for a single move and records the results.
    """
    # Setup
    game = setup_half_filled_board()
    
    # Parameters
    number_simulations = 300
    depth_minimax = 4
    switch_threshold = 31
    
    # Initialize recording data
    algorithms = ["Minimax", "MC", "MCD", "MCTS", "AB+MCD"]
    move_data = {algo: {"moves": [], "times": []} for algo in algorithms}
    
    print("\n=== Half-Filled Board Test ===")
    print("Initial board state:")
    print(game.show_board())
    
    # Test each algorithm on the same initial state
    for algo in algorithms:
        print(f"\nTesting {algo}...")
        result_game, move_data = test_move(
            game, algo, move_data, 
            number_simulations=number_simulations,
            depth_minimax=depth_minimax,
            switch_threshold=switch_threshold
        )
        
        move = move_data[algo]["moves"][-1]
        time_taken = move_data[algo]["times"][-1]
        
        print(f"  Move chosen: {move}")
        print(f"  Time taken: {time_taken:.4f} seconds")
        
        # Show resulting board after this move
        print("\nResulting board:")
        print(result_game.show_board())
    
    # Create visualization of timing results
    create_timing_visualization(move_data)
    
    return move_data

def create_timing_visualization(move_data):
    """Create a bar chart visualization of algorithm timing results."""
    algorithms = list(move_data.keys())
    times = [move_data[algo]["times"][0] for algo in algorithms]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, times, color='skyblue')
    
    # Add timing labels above bars
    for bar, time_val in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 0.01,
            f"{time_val:.4f}s",
            ha='center', 
            fontsize=9
        )
    
    plt.title('Performance Comparison on Half-Filled Board')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Algorithm')
    plt.tight_layout()
    plt.savefig('half_filled_board_comparison.png')
    print("\nPerformance visualization saved as 'half_filled_board_comparison.png'")

def run_extended_test(num_moves=3):
    """
    Run an extended test where each algorithm plays multiple moves 
    from the half-filled board state.
    """
    # Setup
    initial_game = setup_half_filled_board()
    
    # Parameters
    number_simulations = 300
    depth_minimax = 4
    switch_threshold = 31
    
    # Initialize recording data
    algorithms = ["Minimax", "MC", "MCD", "MCTS", "AB+MCD"]
    results = {algo: {
        "total_time": 0,
        "avg_time": 0,
        "game_state": None,
        "moves": []
    } for algo in algorithms}
    
    print("\n=== Extended Test: Multiple Moves ===")
    print(f"Running {num_moves} consecutive moves for each algorithm")
    
    # Test each algorithm starting from the same state
    for algo in algorithms:
        print(f"\nTesting {algo} for {num_moves} moves...")
        
        # Start with a fresh copy of the initial game
        game = deepcopy(initial_game)
        
        for move_num in range(num_moves):
            print(f"  Move {move_num + 1}:")
            
            start_time = time.time()
            
            if algo == "Minimax":
                state = StateMinimax(game.board, game.current_player(), game.balls)
                board = Board()
                move = minimax(board, state, depth_minimax)
            else:
                # Monte Carlo algorithms
                if algo == "MCTS":
                    algo_instance = MonteCarloTreeSearch(game, number_simulations=number_simulations)
                else:
                    algo_instance = get_monte_carlo_player(
                        game, 
                        algo, 
                        number_simulations=number_simulations,
                        switch_threshold=switch_threshold
                    )
                move = algo_instance.get_play()
            
            end_time = time.time()
            time_taken = end_time - start_time
            
            results[algo]["total_time"] += time_taken
            results[algo]["moves"].append(move)
            
            print(f"    Move chosen: {move}")
            print(f"    Time taken: {time_taken:.4f} seconds")
            
            # Apply the move to continue the game
            if move:
                game.move_with_position(move)
                game.toggle_player()
            else:
                print("    No valid move found, ending sequence.")
                break
                
        # Save the final game state and calculate average time
        results[algo]["game_state"] = game
        results[algo]["avg_time"] = results[algo]["total_time"] / len(results[algo]["moves"]) if results[algo]["moves"] else 0
        print(f"  Average move time: {results[algo]['avg_time']:.4f} seconds")
        
        # Show final board state
        print(f"\n  Final board after {len(results[algo]['moves'])} moves:")
        print(game.show_board())
    
    # Create visualization of timing results
    create_extended_visualization(results, num_moves)
    
    return results

def create_extended_visualization(results, num_moves):
    """Create visualizations for the extended test results."""
    algorithms = list(results.keys())
    avg_times = [results[algo]["avg_time"] for algo in algorithms]
    total_times = [results[algo]["total_time"] for algo in algorithms]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot average times
    bars1 = ax1.bar(algorithms, avg_times, color='skyblue')
    ax1.set_title('Average Move Time')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xlabel('Algorithm')
    
    # Add timing labels
    for bar, time_val in zip(bars1, avg_times):
        ax1.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 0.01,
            f"{time_val:.4f}s",
            ha='center', 
            fontsize=9
        )
    
    # Plot total times
    bars2 = ax2.bar(algorithms, total_times, color='lightgreen')
    ax2.set_title('Total Time for All Moves')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_xlabel('Algorithm')
    
    # Add timing labels
    for bar, time_val in zip(bars2, total_times):
        ax2.text(
            bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 0.01,
            f"{time_val:.4f}s",
            ha='center', 
            fontsize=9
        )
    
    # Add overall title
    fig.suptitle(f'Algorithm Performance Comparison ({num_moves} consecutive moves)')
    plt.tight_layout()
    plt.savefig('extended_half_filled_comparison.png')
    print("\nExtended test visualization saved as 'extended_half_filled_comparison.png'")

def run_full_game_test(monte_carlo_type="MC", num_games=2):
    """Run full game simulations comparing one Monte Carlo variant against Minimax."""
    # Parameters
    num_simulations = 300
    max_moves = 50
    minimax_depth = 4
    switch_threshold = 31
    
    print(f"\n=== Full Game Test: {monte_carlo_type} vs Minimax ===")
    print(f"Running {num_games} full games")
    
    # Time the entire simulation
    start_time = time.time()
    
    results = run_simulation(
        number_simulations=num_simulations,
        number_moves=max_moves,
        number_games=num_games,
        depth_minimax=minimax_depth,
        monte_carlo_type=monte_carlo_type,
        switch_threshold=switch_threshold,
        return_stats=True
    )
    
    total_time = time.time() - start_time
    
    # Print results
    print("\n=== Results ===")
    print(f"Algorithm: {monte_carlo_type}")
    print(f"Games played: {results['total_games']}")
    print(f"Monte Carlo wins: {results['monte_carlo_wins']}")
    print(f"Minimax wins: {results['minimax_wins']}")
    print(f"Draws: {results.get('draws', 0)}")
    print(f"Win rate: {results['monte_carlo_wins'] / results['total_games']:.2f}")
    print(f"Average moves per game: {results['average_moves']:.1f}")
    print(f"Total time: {total_time:.2f}s")
    
    return results

if __name__ == "__main__":
    print("=== AI Algorithm Comparison Tests ===")
    print("1. Single Move Test (Half-filled Board)")
    print("2. Extended Test (3 consecutive moves)")
    print("3. Full Game Test (MC vs Minimax)")
    print("4. Full Game Test (MCD vs Minimax)")
    print("5. Full Game Test (AB+MCD vs Minimax)")
    print("6. Full Game Test (MCTS vs Minimax)")
    
    choice = input("\nEnter test number (1-6) or 'all' to run all tests: ")
    
    if choice == '1' or choice.lower() == 'all':
        compare_algorithms_on_half_filled()
    
    if choice == '2' or choice.lower() == 'all':
        run_extended_test(num_moves=3)
    
    if choice == '3' or choice.lower() == 'all':
        run_full_game_test(monte_carlo_type="MC", num_games=2)
    
    if choice == '4' or choice.lower() == 'all':
        run_full_game_test(monte_carlo_type="MCD", num_games=2)
    
    if choice == '5' or choice.lower() == 'all':
        run_full_game_test(monte_carlo_type="AB+MCD", num_games=2)
    
    if choice == '6' or choice.lower() == 'all':
        run_full_game_test(monte_carlo_type="MCTS", num_games=2)
