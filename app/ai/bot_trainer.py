from typing import List, Dict, Any
from app.ai.ataxx_env import AtaxxEnvironment
from app.ai.variants.random_bot import RandomBot
from app.ai.variants.full_minimax import FullMinimax
from app.ai.variants.mcts_binary import MCTSBinary
from app.ai.variants.mcts_binary_dk import MCTSBinaryDK
from app.ai.variants.mcts_fractional import MCTSFractional
from app.ai.variants.mcts_fractional_dk import MCTSFractionalDK
from app.ai.variants.mcts_minimax import MCTSMinimax

def get_trained_move(board: List[List[str]], current_player: str, iterations: int = 50, algorithm: str = "mcts-binary", temperature: float = 0.0) -> Dict[str, Any]:
    try:
        if algorithm == "mcts-binary":
            mcts = MCTSBinary(board, current_player)
        elif algorithm == "mcts-binary-dk":
            mcts = MCTSBinaryDK(board, current_player)
        elif algorithm == "mcts-fractional":
            mcts = MCTSFractional(board, current_player)
        elif algorithm == "mcts-fractional-dk":
            mcts = MCTSFractionalDK(board, current_player)
        elif algorithm == "mcts-minimax":
            mcts = MCTSMinimax(board, current_player)
        else:
            mcts = MCTSBinary(board, current_player)
        move = mcts.get_move(board, current_player, simulations=iterations, temperature=temperature)
        return move
    except Exception as e:
        raise Exception(f"Error in get_trained_move with algorithm {algorithm}: {str(e)}")

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
            move = get_trained_move(env.board, current_player, iterations, algorithm)
        elif algorithm == "minimax":
            move = get_minimax_move(env.board, current_player, depth=iterations)
        else:
            bot = RandomBot(env)
            move = bot.run()
        
        result = {"current_player": current_player, "move": move}
        return result
    
    except Exception as e:
        raise Exception(f"Error in get_bot_move: {str(e)}")

def play_bot_vs_bot(
    initial_board: List[List[str]] = None,
    yellow_algorithm: str = "mcts-binary",
    yellow_iterations: int = 50, 
    red_algorithm: str = "mcts-binary",
    red_iterations: int = 50  
) -> Dict[str, Any]:
    try:
        env = AtaxxEnvironment(initial_board, "yellow")
    except ValueError as e:
        return {"error": f"Invalid initial board: {str(e)}"}

    current_player = "yellow"
    moves_history = []
    max_moves = 20

    try:
        for move_count in range(max_moves):
            scores = env.calculate_scores()
            if env.is_game_over():
                break

            if not env.has_valid_moves(current_player):
                print(f"No valid moves for {current_player}, switching turn")
                current_player = "red" if current_player == "yellow" else "yellow"
                env.current_player = current_player
                continue

            if current_player == "yellow":
                algorithm = yellow_algorithm
                iterations = yellow_iterations
            else:
                algorithm = red_algorithm
                iterations = red_iterations

            move = None
            if algorithm.startswith("mcts"):
                move = get_trained_move(env.board, current_player, iterations, algorithm)
            elif algorithm == "minimax":
                move = get_minimax_move(env.board, current_player, depth=iterations)
            else:
                bot = RandomBot(env)
                move = bot.run()

            if move:
                env.make_move(move["from"], move["to"])
                moves_history.append({"player": current_player, "move": move})
                current_player = "red" if current_player == "yellow" else "yellow"
                env.current_player = current_player
            else:
                print(f"No move returned for {current_player}, switching turn")
                current_player = "red" if current_player == "yellow" else "yellow"
                env.current_player = current_player

        scores = env.calculate_scores()
        winner = (
            "yellow" if scores["yellowScore"] > scores["redScore"]
            else "red" if scores["redScore"] > scores["yellowScore"]
            else "draw"
        )

        result = {
            "winner": winner,
            "scores": scores,
            "moves": moves_history,
            "board": env.board
        }
        return result
    except Exception as e:
        return {"error": f"Internal Server Error: {str(e)}"}