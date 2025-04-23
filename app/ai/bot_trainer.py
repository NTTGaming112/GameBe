from typing import List, Dict, Any
from collections import defaultdict
from app.ai.ataxx_env import AtaxxEnvironment
from app.ai.variants.random_bot import RandomBot
from app.ai.variants.full_minimax import FullMinimax

from app.database import get_move_history_collection
from app.ai.variants.mcts_binary import MCTSBinary
from app.ai.variants.mcts_binary_dk import MCTSBinaryDK
from app.ai.variants.mcts_fractional import MCTSFractional
from app.ai.variants.mcts_fractional_dk import MCTSFractionalDK
from app.ai.variants.mcts_minimax import MCTSMinimax

def get_trained_move(board: List[List[str]], current_player: str, iterations: int = 300, algorithm: str = "mcts") -> Dict[str, Any]:
    env = AtaxxEnvironment(board, current_player)
    
    # Chọn thuật toán MCTS dựa trên algorithm
    if algorithm == "mcts-binary":
        mcts = MCTSBinary(env, current_player)
    elif algorithm == "mcts-binary-dk":
        mcts = MCTSBinaryDK(env, current_player)
    elif algorithm == "mcts-fractional":
        mcts = MCTSFractional(env, current_player)
    elif algorithm == "mcts-fractional-dk":
        mcts = MCTSFractionalDK(env, current_player)
    elif algorithm == "mcts-minimax":
        mcts = MCTSMinimax(env, current_player)
    else:
        mcts = MCTSBinary(env, current_player)  # Mặc định

    return mcts.run(simulations=iterations)

def get_minimax_move(board: List[List[str]], current_player: str, depth: int = 2) -> Dict[str, Any]:
    env = AtaxxEnvironment(board, current_player)
    minimax = FullMinimax(env, current_player, depth=depth)
    return minimax.run()

def play_bot_vs_bot(
    initial_board: List[List[str]] = None,
    yellow_algorithm: str = "mcts",
    yellow_iterations: int = 300,
    red_algorithm: str = "mcts",
    red_iterations: int = 300
) -> Dict[str, Any]:

    # Sao chép bàn cờ để tránh thay đổi initial_board
    board = [row[:] for row in initial_board]
    current_player = "yellow"

    # Tải move history từ MongoDB (nếu có)
    move_history_collection = get_move_history_collection()
    move_history = defaultdict(lambda: defaultdict(float))
    try:
        history_data = move_history_collection.find_one() or {"data": {}}
        for state_key, moves in history_data.get("data", {}).items():
            for move_key, score in moves.items():
                move_history[state_key][move_key] = score
    except Exception as e:
        return {"error": f"Error loading move history from MongoDB: {e}"}

    moves_history = []
    max_moves = 500  # Giới hạn số nước đi tối đa để tránh vòng lặp vô hạn

    # Vòng lặp chính của game
    for move_count in range(max_moves):
        env = AtaxxEnvironment(board, current_player)

        # Kiểm tra điều kiện kết thúc game
        scores = env.calculate_scores()
        if (scores["yellowScore"] == 0 or scores["redScore"] == 0 or
            env.is_board_full() or
            (not env.has_valid_moves("yellow") and not env.has_valid_moves("red"))):
            break

        # Nếu người chơi hiện tại không có nước đi hợp lệ, chuyển lượt
        if not env.has_valid_moves(current_player):
            current_player = "red" if current_player == "yellow" else "yellow"
            continue

        # Chọn thuật toán dựa trên người chơi hiện tại
        if current_player == "yellow":
            algorithm = yellow_algorithm
            iterations = yellow_iterations
        else:
            algorithm = red_algorithm
            iterations = red_iterations

        # Lấy nước đi từ thuật toán tương ứng
        if algorithm.startswith("mcts"):
            move = get_trained_move(board, current_player, iterations, algorithm)
        elif algorithm == "minimax":
            move = get_minimax_move(board, current_player, depth=iterations)
        else:  # "random"
            bot = RandomBot(env)
            move = bot.run()

        # Thực hiện nước đi nếu có
        if move:
            env.make_move(move["from"], move["to"])
            moves_history.append({"player": current_player, "move": move})
            board = [row[:] for row in env.board]
            current_player = "red" if current_player == "yellow" else "yellow"
        else:
            current_player = "red" if current_player == "yellow" else "yellow"

    # Tính điểm và xác định người thắng
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
        "board": board
    }
    return result