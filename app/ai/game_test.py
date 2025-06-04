import asyncio
import platform
from constants import MAP
import numpy as np
from ataxx_state import AtaxxState
from minimax_agent import MinimaxAgent
from mcts_agent import MCTSAgent
from mcts_domain_agent import MCTSDomainAgent
from ab_mcts_domain_agent import ABMCTSDomainAgent
from initial_maps import get_initial_map, INITIAL_MAPS


async def run_game_test():
    agents = {
        "Minimax+AB": MinimaxAgent(max_depth=4),
        "MCTS_300": MCTSAgent(iterations=300),
        "MCTS_Domain_300": MCTSDomainAgent(iterations=300),
        "MCTS_Domain_600": MCTSDomainAgent(iterations=600),
        "AB+MCTS_Domain_600": ABMCTSDomainAgent(iterations=600, ab_depth=4)
    }
    results = {name: {"wins": 0, "losses": 0, "draws": 0, "avg_pieces": 0, "games_played": 0} for name in agents}
    matches = [(name, "Minimax+AB") for name in agents if name != "Minimax+AB"]
    
    for agent1_name, agent2_name in matches:
        print(f"\nMatch: {agent1_name} (X) vs {agent2_name} (O)")

        for game_num in range(5):
            print(f"\nGame {game_num + 1} (Forward)")
            initial_board = get_initial_map(MAP)
            map_index = np.where(np.array([np.array_equal(initial_board, m) for m in INITIAL_MAPS]))[0][0]
            state = AtaxxState(initial_board=initial_board)
            print(f"Initial map #{map_index + 1} (MAP{map_index + 1})")
            state.display_board()
            move_count = 0
            x_pieces, o_pieces = 0, 0

            while not state.is_game_over():
                if not state.get_legal_moves():
                    break

                agent = agents[agent1_name] if state.current_player == 1 else agents[agent2_name]
                move = agent.get_move(state)

                if move:
                    r, c, nr, nc = move
                    is_clone = abs(r - nr) <= 1 and abs(c - nc) <= 1
                    move_type = "Clone" if is_clone else "Jump"
                    print(f"\nMove {move_count + 1}: {agent1_name if state.current_player == 1 else agent2_name} "
                          f"moves from ({r},{c}) to ({nr},{nc}) ({move_type})")
                    state.make_move(move)
                    move_count += 1
                    state.display_board()
                    x_pieces = np.sum(state.board == 1)
                    o_pieces = np.sum(state.board == -1)
                    print(f"Pieces - X: {x_pieces}, O: {o_pieces}")

            winner = state.get_winner()
            results[agent1_name]["avg_pieces"] += x_pieces
            results[agent2_name]["avg_pieces"] += o_pieces
            results[agent1_name]["games_played"] += 1
            results[agent2_name]["games_played"] += 1

            if winner == 1:
                results[agent1_name]["wins"] += 1
                results[agent2_name]["losses"] += 1
                print(f"Winner: {agent1_name} (X)")

            elif winner == -1:
                results[agent1_name]["losses"] += 1
                results[agent2_name]["wins"] += 1
                print(f"Winner: {agent2_name} (O)")

            else:
                results[agent1_name]["draws"] += 1
                results[agent2_name]["draws"] += 1
                print("Draw")
        
        print(f"\nMatch: {agent2_name} (X) vs {agent1_name} (O)")

        for game_num in range(5):
            print(f"\nGame {game_num + 1} (Reverse)")
            initial_board = get_initial_map(MAP)
            map_index = np.where(np.array([np.array_equal(initial_board, m) for m in INITIAL_MAPS]))[0][0]
            state = AtaxxState(initial_board=initial_board)
            print(f"Initial map #{map_index + 1} (MAP{map_index + 1})")
            state.display_board()
            move_count = 0
            x_pieces, o_pieces = 0, 0
            consecutive_passes = 0

            while not state.is_game_over():
                if not state.get_legal_moves():
                    print(f"Player {state.current_player} has no legal moves")
                    consecutive_passes += 1

                    if consecutive_passes >= 2:
                        print("Both players have no legal moves, ending game")
                        break

                    state.current_player = -state.current_player
                    continue

                consecutive_passes = 0

                agent = agents[agent2_name] if state.current_player == 1 else agents[agent1_name]
                move = agent.get_move(state)

                if move:
                    r, c, nr, nc = move
                    is_clone = abs(r - nr) <= 1 and abs(c - nc) <= 1
                    move_type = "Clone" if is_clone else "Jump"
                    print(f"\nMove {move_count + 1}: {agent2_name if state.current_player == 1 else agent1_name} "
                          f"moves from ({r},{c}) to ({nr},{nc}) ({move_type})")
                    state.make_move(move)
                    move_count += 1
                    state.display_board()
                    x_pieces = np.sum(state.board == 1)
                    o_pieces = np.sum(state.board == -1)
                    print(f"Pieces - X: {x_pieces}, O: {o_pieces}")

            winner = state.get_winner()
            results[agent2_name]["avg_pieces"] += x_pieces
            results[agent1_name]["avg_pieces"] += o_pieces
            results[agent2_name]["games_played"] += 1
            results[agent1_name]["games_played"] += 1

            if winner == 1:
                results[agent2_name]["wins"] += 1
                results[agent1_name]["losses"] += 1
                print(f"Winner: {agent2_name} (X)")

            elif winner == -1:
                results[agent2_name]["losses"] += 1
                results[agent1_name]["wins"] += 1
                print(f"Winner: {agent1_name} (O)")

            else:
                results[agent2_name]["draws"] += 1
                results[agent1_name]["draws"] += 1
                print("Draw")
    
    print("\nTournament Results (Minimax+AB vs Others):")

    for name in results:
        if results[name]["games_played"] > 0:
            results[name]["avg_pieces"] /= results[name]["games_played"]
            print(f"{name}: Wins={results[name]['wins']}, Losses={results[name]['losses']}, Draws={results[name]['draws']}, Avg Pieces={results[name]['avg_pieces']:.2f}")

async def main():
    await run_game_test()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
    
else:
    if __name__ == "__main__":
        asyncio.run(main())