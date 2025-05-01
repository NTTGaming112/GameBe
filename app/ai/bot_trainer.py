from typing import List, Dict, Any
from app.ai.ataxx_env import AtaxxEnvironment
from app.ai.variants.random_bot import RandomBot
from app.ai.variants.full_minimax import FullMinimax
from app.ai.variants.mcts_binary import MCTSBinary
from app.ai.variants.mcts_binary_dk import MCTSBinaryDK
from app.ai.variants.mcts_fractional import MCTSFractional
from app.ai.variants.mcts_fractional_dk import MCTSFractionalDK
from app.ai.variants.mcts_minimax import MCTSMinimax
from app.ai.base.base_mcts import BaseMCTS, MCTSNode

MCTS_INSTANCES: Dict[str, BaseMCTS] = {}

def get_trained_move(
    board: List[List[str]],
    current_player: str,
    iterations: int = 50,
    algorithm: str = "mcts-binary",
    temperature: float = 0.0,
    game_id: str = "default"
) -> Dict[str, Any]:
    try:
        env = AtaxxEnvironment(board, current_player)
        print("Board in get_trained_move:", board)
        
        if game_id not in MCTS_INSTANCES:
            print(f"Initializing MCTS instance for game_id: {game_id}, algorithm: {algorithm}")
            if algorithm == "mcts-binary":
                MCTS_INSTANCES[game_id] = MCTSBinary(board, current_player, c=2)
            elif algorithm == "mcts-binary-dk":
                MCTS_INSTANCES[game_id] = MCTSBinaryDK(board, current_player, c=2)
            elif algorithm == "mcts-fractional":
                MCTS_INSTANCES[game_id] = MCTSFractional(board, current_player, c=2)
            elif algorithm == "mcts-fractional-dk":
                MCTS_INSTANCES[game_id] = MCTSFractionalDK(board, current_player, c=2)
            elif algorithm == "mcts-minimax":
                MCTS_INSTANCES[game_id] = MCTSMinimax(board, current_player, c=2)
            else:
                MCTS_INSTANCES[game_id] = MCTSBinary(board, current_player, c=2)
        
        mcts = MCTS_INSTANCES[game_id]
        
        if not hasattr(mcts, 'tree') or not mcts.tree:
            print("MCTS tree is empty or not initialized, reinitializing")
            mcts.tree = {env.get_state_key(): MCTSNode(env.get_state_key(), env.get_valid_moves())}
            mcts.root_key = env.get_state_key()
        
        if mcts.env.get_state_key() != env.get_state_key():
            print("Updating MCTS state due to state mismatch")
            valid_moves = mcts.env.get_valid_moves()
            if not valid_moves:
                print("No valid moves available for current state")
                return {"error": "No valid moves available"}
            for move in valid_moves:
                temp_env = mcts.env.clone()
                temp_env.make_move(move["from"], move["to"])
                if temp_env.get_state_key() == env.get_state_key():
                    print(f"Updating root with move: {move}")
                    mcts.update_root(move)
                    break
            else:
                print("No matching move found, resetting MCTS state")
                mcts.env = env
                mcts.player = current_player
                new_root_key = env.get_state_key()
                if new_root_key not in mcts.tree or mcts.tree[new_root_key] is None:
                    print(f"Creating new node for root_key: {new_root_key}")
                    mcts.tree[new_root_key] = MCTSNode(new_root_key, env.get_valid_moves())
                mcts.root_key = new_root_key

        if mcts.root_key not in mcts.tree or mcts.tree[mcts.root_key] is None:
            print("Root node is None or missing, reinitializing")
            mcts.tree[mcts.root_key] = MCTSNode(mcts.root_key, env.get_valid_moves())

        move = mcts.get_move(board, current_player, simulations=iterations, temperature=temperature)
        if not move:
            print("No move returned by MCTS")
            return {"error": "No valid move found"}
        return move

    except Exception as e:
        print("Error in get_trained_move:", str(e))
        raise Exception(f"Error in get_trained_move with algorithm {algorithm}: {str(e)}")

def end_game(game_id: str) -> None:
    """Xóa MCTS instance khi trò chơi kết thúc."""
    MCTS_INSTANCES.pop(game_id, None)

def get_minimax_move(board: List[List[str]], current_player: str, depth: int = 2) -> Dict[str, Any]:
    try:
        env = AtaxxEnvironment(board, current_player)
        minimax = FullMinimax(env, current_player, depth=depth)
        return minimax.run()
    except Exception as e:
        raise Exception(f"Error in get_minimax_move: {str(e)}")
    
def get_bot_move(
    initial_board: List[List[str]], 
    current_player: str, 
    algorithm: str = "mcts-binary", 
    iterations: int = 50
) -> Dict[str, Any]:
    try:
        env = AtaxxEnvironment(initial_board, current_player)
        if algorithm.startswith("mcts"):
            move = get_trained_move(initial_board, current_player, iterations, algorithm)
        elif algorithm == "minimax":
            move = get_minimax_move(initial_board, current_player, depth=iterations)
        else:
            bot = RandomBot(env)
            move = bot.run()
        
        result = {"current_player": current_player, "move": move}
        return result
    
    except Exception as e:
        raise Exception(f"Error in get_bot_move: {str(e)}")

# def play_bot_vs_bot(
#     initial_board: List[List[str]] = None,
#     yellow_algorithm: str = "mcts-binary",
#     yellow_iterations: int = 50, 
#     red_algorithm: str = "mcts-binary",
#     red_iterations: int = 50  
# ) -> Dict[str, Any]:
#     try:
#         env = AtaxxEnvironment(initial_board, "yellow")
#     except ValueError as e:
#         return {"error": f"Invalid initial board: {str(e)}"}

#     current_player = "yellow"
#     moves_history = []
#     max_moves = 20

#     try:
#         for move_count in range(max_moves):
#             scores = env.calculate_scores()
#             if env.is_game_over():
#                 break

#             if not env.has_valid_moves(current_player):
#                 print(f"No valid moves for {current_player}, switching turn")
#                 current_player = "red" if current_player == "yellow" else "yellow"
#                 env.current_player = current_player
#                 continue

#             if current_player == "yellow":
#                 algorithm = yellow_algorithm
#                 iterations = yellow_iterations
#             else:
#                 algorithm = red_algorithm
#                 iterations = red_iterations

#             move = None
#             if algorithm.startswith("mcts"):
#                 move = get_trained_move(env.board, current_player, iterations, algorithm)
#             elif algorithm == "minimax":
#                 move = get_minimax_move(env.board, current_player, depth=iterations)
#             else:
#                 bot = RandomBot(env)
#                 move = bot.run()

#             if move:
#                 env.make_move(move["from"], move["to"])
#                 moves_history.append({"player": current_player, "move": move})
#                 current_player = "red" if current_player == "yellow" else "yellow"
#                 env.current_player = current_player
#             else:
#                 print(f"No move returned for {current_player}, switching turn")
#                 current_player = "red" if current_player == "yellow" else "yellow"
#                 env.current_player = current_player

#         scores = env.calculate_scores()
#         winner = (
#             "yellow" if scores["yellowScore"] > scores["redScore"]
#             else "red" if scores["redScore"] > scores["yellowScore"]
#             else "draw"
#         )

#         result = {
#             "winner": winner,
#             "scores": scores,
#             "moves": moves_history,
#             "board": env.board
#         }
#         return result
#     except Exception as e:
#         return {"error": f"Internal Server Error: {str(e)}"}