# Ataxx Game Constants
BOARD_SIZE = 7

# Heuristic coefficients for move evaluation
S1, S2, S3, S4 = 10, 5, 2, 3

# Default agent parameters
DEFAULT_MINIMAX_DEPTH = 4
DEFAULT_MCTS_ITERATIONS = 300
DEFAULT_MCTS_DOMAIN_ITERATIONS = 600

# Game state values
EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = -1
BLOCKED = -2  # For future use if blocked cells are added