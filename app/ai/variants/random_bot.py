import random

class RandomBot:
    def __init__(self, board):
        self.env = board

    def run(self):
        moves = self.env.get_valid_moves()
        return random.choice(moves) if moves else None