#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from ucb_monte_carlo import MonteCarlo
from .minimax import minimax
from .board import Board, StateMinimax
from .binary_mcts import MonteCarloTreeSearch
from .ataxx_state import Ataxx
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import sleep
import pickle
import argparse, logging
import time

def main(number_simulations=200, number_moves=50, number_games=10, depth_minimax=2):
    """Run a simulation between Minimax and MCTS players.
    
    Args:
        number_simulations: Number of MCTS simulations per move
        number_moves: Maximum number of moves per game
        number_games: Number of games to play
        depth_minimax: Depth for minimax search
    """
    logging.basicConfig(
        filename=f"data-ucb1-number_simulations-{number_simulations}--"
                f"number_moves-{number_moves}--"
                f"number_games-{number_games}--"
                f"depth_minimax-{depth_minimax}.log",
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s %(message)s'
    )

    print("\n=== Ataxx Game Simulation ===")
    print(f"Parameters:")
    print(f"- MCTS simulations per move: {number_simulations}")
    print(f"- Maximum moves per game: {number_moves}")
    print(f"- Number of games: {number_games}")
    print(f"- Minimax depth: {depth_minimax}\n")

    board = Board()
    dados = {}
    jogadas = []
    per_preto = 0
    data_best_moves = {}
    moves = {}
    cont_moves = {}

    # Track which algorithm is playing as White/Black
    minimax_is_white = True

    for i in range(number_games):
        moves[-1] = []
        moves[1] = []
        cont_moves[-1] = 0
        cont_moves[1] = 0
        game = Ataxx()
        
        c = 1
        print(f"\nGame {i+1}/{number_games}")
        print("=" * 30)
        print(f"Minimax is playing as {'White' if minimax_is_white else 'Black'}")
        print(f"MCTS is playing as {'Black' if minimax_is_white else 'White'}")

        while not game.is_game_over():
            # Determine which algorithm to use based on current player
            if (game.current_player() == 1 and minimax_is_white) or (game.current_player() == -1 and not minimax_is_white):
                # Minimax player's turn
                print(f"\nMove {c}: {'White' if minimax_is_white else 'Black'} (Minimax)")
                state = StateMinimax(game.board, game.current_player(), game.balls)
                begin = time.time()
                move = minimax(board, state, depth_minimax)
                move_time = time.time() - begin
                print(f"Move time: {move_time:.2f}s")
            else:
                # MCTS player's turn
                print(f"\nMove {c}: {'Black' if minimax_is_white else 'White'} (MCTS)")
                mc = MonteCarloTreeSearch(game, number_simulations=number_simulations)
                
                begin = time.time()
                move = mc.get_play()
                move_time = time.time() - begin
                print(f"Move time: {move_time:.2f}s")

            player = game.current_player()
            if move:
                if move in moves[player]:
                    cont_moves[player] += 1
                else:
                    cont_moves[player] = 0
                    moves[player].append(move)

                game.move_with_position(move)

            if cont_moves[game.current_player()] > 6:
                print("Move repeated too many times")
                break

            if len(moves[player]) > 6:
                moves[player] = []
                cont_moves[player] = 0

            print("\nCurrent board:")
            print(game.show_board())

            game.toggle_player()

            if game.is_game_over():
                break

            c += 1
        
        ga = game.get_winner()
        print(f"\nGame {i+1} finished in {c} moves")
        
        # Adjust winner based on who was playing which color
        if minimax_is_white:
            actual_winner = ga
        else:
            actual_winner = -ga  # Invert the result since colors were swapped
            
        if actual_winner not in dados:
            dados[actual_winner] = 0
        dados[actual_winner] += 1
        jogadas.append({'moves': c, 'winner': actual_winner})
        
        # Print game result
        if actual_winner == 1:
            print("Winner: Minimax")
        elif actual_winner == -1:
            print("Winner: MCTS")
        else:
            print("Game ended in a draw")
        
        print("\nCurrent standings:")
        print(f"Minimax wins: {dados.get(1, 0)}")
        print(f"MCTS wins: {dados.get(-1, 0)}")
        print(f"Draws: {dados.get(0, 0)}")
        print("=" * 30)

        # Switch colors for next game
        minimax_is_white = not minimax_is_white

    print("\n=== Final Results ===")
    print(f"Total games played: {number_games}")
    print(f"Minimax wins: {dados.get(1, 0)}")
    print(f"MCTS wins: {dados.get(-1, 0)}")
    print(f"Draws: {dados.get(0, 0)}")
    
    # Save results
    with open(f"data-ucb1-number_simulations-{number_simulations}--"
              f"number_moves-{number_moves}--"
              f"number_games-{number_games}--"
              f"depth_minimax-{depth_minimax}.pickle", 'wb') as handle:
        pickle.dump(data_best_moves, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Ataxx game simulation')
    parser.add_argument('--number-simulations', type=int, default=200,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--number-moves', type=int, default=50,
                        help='Maximum number of moves per game')
    parser.add_argument('--number-games', type=int, default=10,
                        help='Number of games to play')
    parser.add_argument('--depth-minimax', type=int, default=2,
                        help='Depth for minimax search')
    
    args = parser.parse_args()
    main(args.number_simulations, args.number_moves, 
         args.number_games, args.depth_minimax)