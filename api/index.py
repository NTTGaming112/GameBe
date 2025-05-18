from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import time
from app.ai.ataxx_state import Ataxx
from app.ai.monte_carlo import get_monte_carlo_player

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Ataxx MCTS API (FastAPI)"}

@app.post("/get_move")
async def get_move(request: Request):
    """
    Get the best move for the current game state using MCTS.
    
    Request JSON format:
    {
        "board": 7x7 array representing the Ataxx board,
        "current_player": "red" or "yellow",
        "algorithm": "MC" (Basic Monte Carlo), "MCD" (Monte Carlo with Domain Knowledge), 
                    "AB+MCD" (Alpha-Beta + Monte Carlo), or "MINIMAX" (Alpha-Beta Minimax),
        "iterations": number of MCTS iterations (default: 300),
        "policy_args": {
            "switch_threshold": threshold for switching algorithms in AB+MCD (default: 31),
            "use_simulation_formula": whether to use simulation formula (default: false),
            "s1_ratio": ratio of S1 simulations to iterations (default: 1.0),
            "s2_ratio": ratio of S2 simulations to iterations (default: 1.0),
            "s3_ratio": ratio of S3 simulations to iterations (default: 0.5),
            "depth": search depth for Minimax algorithm (default: 4)
        }
    }
    
    Response JSON format:
    {
        "move": {
            "from": {"row": row, "col": col},
            "to": {"row": row, "col": col}
        },
        "execution_time": execution time in seconds,
        "current_player": current player after move
    }
    """
    data = await request.json()
    required_fields = ["board", "current_player", "algorithm"]
    for field in required_fields:
        if field not in data:
            return {"error": f"Missing required field: {field}"}
    state = Ataxx()
    def cell_to_num(cell):
        if cell == 'red' or cell == 1:
            return 1
        elif cell == 'yellow' or cell == -1:
            return -1
        else:
            return 0
    state.board = [[cell_to_num(cell) for cell in row] for row in data["board"]]
    state.balls[1] = sum(cell == 1 for row in state.board for cell in row)
    state.balls[-1] = sum(cell == -1 for row in state.board for cell in row)
    if data["current_player"] == "red":
        state.turn_player = 1
    elif data["current_player"] == "yellow":
        state.turn_player = -1
    else:
        state.turn_player = data["current_player"]
    algorithm = data.get("algorithm", "MC")
    iterations = int(data.get("iterations", 300))
    policy_args = data.get("policy_args", {})
    switch_threshold = policy_args.get("switch_threshold", 31)
    use_simulation_formula = policy_args.get("use_simulation_formula", False)
    s1_ratio = policy_args.get("s1_ratio", 1.0)
    s2_ratio = policy_args.get("s2_ratio", 1.0)
    s3_ratio = policy_args.get("s3_ratio", 0.5)
    depth = policy_args.get("depth", 4)
    time_limit = float(data.get("time_limit", 50))
    start_time = time.time()
    mcts = get_monte_carlo_player(
        state,
        mc_type=algorithm,
        number_simulations=iterations,
        switch_threshold=switch_threshold,
        use_simulation_formula=use_simulation_formula,
        s1_ratio=s1_ratio,
        s2_ratio=s2_ratio,
        s3_ratio=s3_ratio,
        depth=depth,
        time_limit=time_limit
    )
    if hasattr(mcts, 'get_move'):
        best_move = mcts.get_move(time_limit=time_limit)
    elif hasattr(mcts, 'get_play'):
        best_move = mcts.get_play()
    else:
        return {"error": f"AI player object {type(mcts)} has no get_move or get_play method"}
    execution_time = time.time() - start_time
    if best_move is None:
        return {"error": "No legal moves available", "execution_time": execution_time}
    if best_move[0] == 'c':
        _, to_pos = best_move
        from_pos = None
    else:
        _, to_pos, from_pos = best_move
    player_str = 'red' if state.turn_player == 1 else 'yellow'
    return {
        "move": {
            "from": {"row": from_pos[0], "col": from_pos[1]} if from_pos else None,
            "to": {"row": to_pos[0], "col": to_pos[1]}
        },
        "execution_time": float(execution_time),
        "current_player": player_str,
    }

@app.post("/get_legal_moves")
async def get_legal_moves(request: Request):
    data = await request.json()
    state = Ataxx()
    if "board" in data:
        state.board = data["board"]
    if "current_player" in data:
        if data["current_player"] == "red":
            state.turn_player = 1
        elif data["current_player"] == "yellow":
            state.turn_player = -1
        else:
            state.turn_player = data["current_player"]
    legal_moves = state.get_legal_moves()
    return {
        "legal_moves": [
            {"from_pos": list(from_pos), "to_pos": list(to_pos)}
            for from_pos, to_pos in legal_moves
        ]
    }

@app.post("/evaluate_state")
async def evaluate_state(request: Request):
    data = await request.json()
    state = Ataxx()
    if "board" in data:
        state.board = data["board"]
    if "current_player" in data:
        if data["current_player"] == "red":
            state.turn_player = 1
        elif data["current_player"] == "yellow":
            state.turn_player = -1
        else:
            state.turn_player = data["current_player"]
    winner = state.get_winner()
    is_terminal = state.is_terminal()
    piece_counts = state.get_pieces_count()
    return {
        "winner": winner,
        "is_terminal": is_terminal,
        "piece_counts": piece_counts
    }