#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import time
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.game_simulation import main

def run_tests():
    # Test parameters
    test_cases = [
        # (mcts_simulations, minimax_depth, number_games)
        (200, 2, 1)  # Base case
    ]
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for mcts_sim, min_depth, num_games in test_cases:
        print(f"\nRunning test case:")
        print(f"MCTS simulations: {mcts_sim}")
        print(f"Minimax depth: {min_depth}")
        print(f"Number of games: {num_games}")
        
        start_time = time.time()
        main(mcts_sim, 50, num_games, min_depth)
        end_time = time.time()
        
        duration = end_time - start_time
        avg_move_time = duration / (num_games * 25)  # Assuming ~25 moves per game
        results.append({
            'mcts_simulations': mcts_sim,
            'minimax_depth': min_depth,
            'games': num_games,
            'duration': duration,
            'avg_move_time': avg_move_time
        })
        
        print(f"\nTest completed in {duration:.2f} seconds")
        print(f"Average move time: {avg_move_time:.2f} seconds")
    
    # Save results
    with open(f"test_results_{timestamp}.txt", "w") as f:
        f.write("Test Results\n")
        f.write("============\n\n")
        for r in results:
            f.write(f"MCTS simulations: {r['mcts_simulations']}\n")
            f.write(f"Minimax depth: {r['minimax_depth']}\n")
            f.write(f"Games played: {r['games']}\n")
            f.write(f"Total duration: {r['duration']:.2f} seconds\n")
            f.write(f"Average move time: {r['avg_move_time']:.2f} seconds\n")
            f.write("-" * 50 + "\n")

if __name__ == "__main__":
    run_tests() 