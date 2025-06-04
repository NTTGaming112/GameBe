import asyncio
import platform
import argparse
import pygame
import numpy as np
import pandas as pd
import os
from ataxx_state import AtaxxState
from minimax_agent import MinimaxAgent
from mcts_agent import MCTSAgent
from mcts_domain_agent import MCTSDomainAgent
from ab_mcts_domain_agent import ABMCTSDomainAgent

CELL_SIZE = 80
BOARD_SIZE = 7
WINDOW_WIDTH = BOARD_SIZE * CELL_SIZE + 300
WINDOW_HEIGHT = BOARD_SIZE * CELL_SIZE + 100
COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'gray': (128, 128, 128),
    'bg': (200, 200, 200)
}

MAP_FILES = {
    "position_3_3_empty_w.txt": """WWWWWWW\nBWWWWWW\nWWWBBBW\nWBBBBBW\nWBBBBBW\n#BBBBBB\n##WBBBW"""
}

def read_map_file(filename):
    if platform.system() == "Emscripten":
        if filename not in MAP_FILES:
            raise ValueError(f"Map file {filename} not available in Pyodide")
        lines = MAP_FILES[filename].split('\n')
    else:
        try:
            with open(f"map/{filename}", 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise ValueError(f"Map file 'map/{filename}' not found")
        except Exception as e:
            raise ValueError(f"Error reading map file: {e}")
    
    board = np.zeros((7, 7), dtype=int)
    for r, line in enumerate(lines[:7]):
        for c, char in enumerate(line.strip()[:7]):
            if char == 'B':
                board[r][c] = 1
            elif char == 'W':
                board[r][c] = -1
            elif char == '#':
                board[r][c] = 0
    return board

class AtaxxGame:
    def __init__(self, games_per_match=5, iterations=300, algo1="MCTS_Domain_600", 
                 algo2="Minimax+AB", display="pygame", map_file=None, delay=0.5, first_player='W'):
        self.map_file = map_file
        self.games_per_match = games_per_match
        self.iterations = iterations
        self.delay = delay
        self.display = display.lower()
        self.first_player = 1 if first_player == 'W' else -1
        if self.display not in ['pygame', 'terminal']:
            raise ValueError("Display must be 'pygame' or 'terminal'")
        self.agents = {
            "Minimax+AB": MinimaxAgent(max_depth=4),
            "MCTS_300": MCTSAgent(iterations=self.iterations),
            "MCTS_Domain_300": MCTSDomainAgent(iterations=self.iterations),
            "MCTS_Domain_600": MCTSDomainAgent(iterations=max(self.iterations, 600)),
            "AB+MCTS_Domain_600": ABMCTSDomainAgent(iterations=max(self.iterations, 600), ab_depth=4)
        }
        if algo1 not in self.agents or algo2 not in self.agents:
            raise ValueError(f"Invalid agent(s). Choose from: {list(self.agents.keys())}")
        self.algo1 = algo1
        self.algo2 = algo2
        self.results = {name: {"wins": 0, "losses": 0, "draws": 0, "avg_pieces": 0, "games_played": 0} 
                        for name in [algo1, algo2]}
        self.screen = None
        self.font = None
        self.state = None
        self.running = True
        self.menu_active = self.display == 'pygame'
        self.selected_games = self.games_per_match
        self.selected_algo1 = self.algo1
        self.selected_algo2 = self.algo2
        self.agent_names = list(self.agents.keys())
        self.initial_board = read_map_file(self.map_file)

    def init_pygame(self):
        if self.display != 'pygame':
            return
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Ataxx Tournament")
        try:
            self.font = pygame.font.SysFont('Arial', 30)
        except:
            self.font = pygame.font.Font(None, 36)

    def draw_board(self):
        if not self.state:
            return
        if self.display == 'terminal':
            self.state.display_board()
            return
        
        self.screen.fill((220, 220, 220))  
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x, y = c * CELL_SIZE, r * CELL_SIZE
                pygame.draw.rect(self.screen, (255, 255, 255), (x, y, CELL_SIZE, CELL_SIZE), 0)
                pygame.draw.rect(self.screen, (150, 150, 150), (x, y, CELL_SIZE, CELL_SIZE), 1)
                if self.state.board[r][c] == 1:
                    pygame.draw.circle(self.screen, (200, 50, 50), (x + CELL_SIZE//2, y + CELL_SIZE//2), CELL_SIZE//3)
                    pygame.draw.circle(self.screen, (255, 100, 100), (x + CELL_SIZE//2, y + CELL_SIZE//2), CELL_SIZE//3 - 3)
                elif self.state.board[r][c] == -1:
                    pygame.draw.circle(self.screen, (50, 50, 200), (x + CELL_SIZE//2, y + CELL_SIZE//2), CELL_SIZE//3)
                    pygame.draw.circle(self.screen, (100, 100, 255), (x + CELL_SIZE//2, y + CELL_SIZE//2), CELL_SIZE//3 - 3)
                elif self.state.board[r][c] == -2:
                    pygame.draw.rect(self.screen, (100, 100, 100), (x + 5, y + 5, CELL_SIZE - 10, CELL_SIZE - 10))
        
        info_box_x = BOARD_SIZE * CELL_SIZE + 10
        info_box_y = 10
        info_box_width = 280
        info_box_height = 380
        pygame.draw.rect(self.screen, (240, 240, 240), (info_box_x, info_box_y, info_box_width, info_box_height), 0)
        pygame.draw.rect(self.screen, (150, 150, 150), (info_box_x, info_box_y, info_box_width, info_box_height), 2)
        
        x_pieces = np.sum(self.state.board == 1)
        o_pieces = np.sum(self.state.board == -1)
        player = "X" if self.state.current_player == 1 else "O"
        info_texts = [
            f"{self.algo1} (X): {x_pieces}",
            f"{self.algo2} (O): {o_pieces}",
            f"Turn: {player}",
            f"Map: {self.map_file or 'Default'}"
        ]
        for i, text in enumerate(info_texts):
            surface = self.font.render(text, True, (50, 50, 50))
            self.screen.blit(surface, (info_box_x + 20, info_box_y + 20 + i * 40))
        
        pygame.display.flip()

    def draw_menu(self):
        self.screen.fill((220, 220, 220))
        
        title = self.font.render("Ataxx Tournament", True, (50, 50, 50))
        self.screen.blit(title, (WINDOW_WIDTH//2 - title.get_width()//2, 30))
        
        menu_box_x = 50
        menu_box_y = 100
        menu_box_width = WINDOW_WIDTH - 100
        menu_box_height = 400
        pygame.draw.rect(self.screen, (240, 240, 240), (menu_box_x, menu_box_y, menu_box_width, menu_box_height), 0)
        pygame.draw.rect(self.screen, (150, 150, 150), (menu_box_x, menu_box_y, menu_box_width, menu_box_height), 2)
        
        options = [
            (f"Map: {self.map_file}", None, None),
            (f"Games per Match: {self.selected_games}", "games_prev", "games_next"),
            (f"Algo1 (X): {self.selected_algo1}", "algo1_prev", "algo1_next"),
            (f"Algo2 (O): {self.selected_algo2}", "algo2_prev", "algo2_next"),
            (f"First Player: {'X (White)' if self.first_player == 1 else 'O (Black)'}", "first_player_prev", "first_player_next")
        ]
        
        buttons = {}
        for i, (text, prev_key, next_key) in enumerate(options):
            y_pos = menu_box_y + 40 + i * 70
            label = self.font.render(text, True, (50, 50, 50))
            self.screen.blit(label, (menu_box_x + 30, y_pos))
            
            if prev_key and next_key:
                prev_rect = pygame.Rect(menu_box_x + menu_box_width - 120, y_pos - 5, 40, 40)
                next_rect = pygame.Rect(menu_box_x + menu_box_width - 60, y_pos - 5, 40, 40)
                
                mouse_pos = pygame.mouse.get_pos()
                prev_color = (180, 180, 180) if prev_rect.collidepoint(mouse_pos) else (200, 200, 200)
                next_color = (180, 180, 180) if next_rect.collidepoint(mouse_pos) else (200, 200, 200)
                
                pygame.draw.rect(self.screen, prev_color, prev_rect)
                pygame.draw.rect(self.screen, next_color, next_rect)
                pygame.draw.rect(self.screen, (150, 150, 150), prev_rect, 1)
                pygame.draw.rect(self.screen, (150, 150, 150), next_rect, 1)
                
                prev_text = self.font.render("<", True, (50, 50, 50))
                next_text = self.font.render(">", True, (50, 50, 50))
                self.screen.blit(prev_text, (menu_box_x + menu_box_width - 110, y_pos))
                self.screen.blit(next_text, (menu_box_x + menu_box_width - 50, y_pos))
                
                buttons[prev_key] = prev_rect
                buttons[next_key] = next_rect
        
        start_button = pygame.Rect(WINDOW_WIDTH//2 - 100, menu_box_y + menu_box_height - 60, 200, 50)
        mouse_pos = pygame.mouse.get_pos()
        start_color = (180, 180, 180) if start_button.collidepoint(mouse_pos) else (200, 200, 200)
        pygame.draw.rect(self.screen, start_color, start_button)
        pygame.draw.rect(self.screen, (150, 150, 150), start_button, 1)
        start_text = self.font.render("Start", True, (50, 50, 50))
        self.screen.blit(start_text, (WINDOW_WIDTH//2 - start_text.get_width()//2, menu_box_y + menu_box_height - 50))
        buttons['start_button'] = start_button
        
        pygame.display.flip()
        return buttons.get('map_prev'), buttons.get('map_next'), buttons.get('games_prev'), buttons.get('games_next'), \
            buttons.get('algo1_prev'), buttons.get('algo1_next'), buttons.get('algo2_prev'), buttons.get('algo2_next'), \
            buttons.get('first_player_prev'), buttons.get('first_player_next'), buttons.get('start_button')

    async def run_menu(self):
        if self.display != 'pygame':
            return
        while self.menu_active and self.running:
            buttons = self.draw_menu()
            map_prev, map_next, games_prev, games_next, algo1_prev, algo1_next, algo2_prev, algo2_next, \
            first_player_prev, first_player_next, start_button = buttons
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    if games_prev.collidepoint(pos):
                        self.selected_games = max(1, self.selected_games - 1)
                    elif games_next.collidepoint(pos):
                        self.selected_games = min(10, self.selected_games + 1)
                    elif algo1_prev.collidepoint(pos):
                        idx = self.agent_names.index(self.selected_algo1)
                        self.selected_algo1 = self.agent_names[(idx - 1) % len(self.agent_names)]
                    elif algo1_next.collidepoint(pos):
                        idx = self.agent_names.index(self.selected_algo1)
                        self.selected_algo1 = self.agent_names[(idx + 1) % len(self.agent_names)]
                    elif algo2_prev.collidepoint(pos):
                        idx = self.agent_names.index(self.selected_algo2)
                        self.selected_algo2 = self.agent_names[(idx - 1) % len(self.agent_names)]
                    elif algo2_next.collidepoint(pos):
                        idx = self.agent_names.index(self.selected_algo2)
                        self.selected_algo2 = self.agent_names[(idx + 1) % len(self.agent_names)]
                    elif first_player_prev.collidepoint(pos):
                        self.first_player = 1 if self.first_player == -1 else -1
                    elif first_player_next.collidepoint(pos):
                        self.first_player = -1 if self.first_player == 1 else 1
                    elif start_button.collidepoint(pos):
                        self.menu_active = False
                        self.games_per_match = self.selected_games
                        self.algo1 = self.selected_algo1
                        self.algo2 = self.selected_algo2
                        self.results = {name: {"wins": 0, "losses": 0, "draws": 0, "avg_pieces": 0, "games_played": 0} 
                                        for name in [self.algo1, self.algo2]}

    async def play_game(self, agent1_name, agent2_name, forward=True):
        self.state = AtaxxState(initial_board=self.initial_board, current_player=self.first_player)
        map_name = self.map_file or f"Map #{self.map_idx + 1}"
        print(f"\nGame ({'Forward' if forward else 'Reverse'}) on {map_name}")
        legal_moves = self.state.get_legal_moves()
        if not legal_moves:
            print(f"Warning: No initial legal moves for player {self.state.current_player}")
        self.draw_board()
        
        move_count = 0
        x_pieces, o_pieces = 0, 0

        while not self.state.is_game_over() and self.running:
            legal_moves = self.state.get_legal_moves()
            
            if not legal_moves:
                print(f"Player {self.state.current_player} has no legal moves - PASS")
                self.state.current_player = -self.state.current_player
                opponent_moves = self.state.get_legal_moves()
                if not opponent_moves:
                    print("Both players have no legal moves - game ends")
                    break
                continue
            
            agent_name = agent1_name if (forward and self.state.current_player == 1) or (not forward and self.state.current_player == -1) else agent2_name
            agent = self.agents[agent_name]
            move = agent.get_move(self.state)

            if move:
                r, c, nr, nc = move
                is_clone = abs(r - nr) <= 1 and abs(c - nc) <= 1
                move_type = "Clone" if is_clone else "Jump"
                print(f"\nMove {move_count + 1}: {agent_name} moves from ({r},{c}) to ({nr},{nc}) ({move_type})")
                self.state.make_move(move)
                move_count += 1
                x_pieces = np.sum(self.state.board == 1)
                o_pieces = np.sum(self.state.board == -1)
                print(f"Pieces - X: {x_pieces}, O: {o_pieces}")
                self.draw_board()
                
                if self.display == 'pygame':
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                    await asyncio.sleep(self.delay)
            else:
                print(f"\n{agent_name} has no legal moves - PASS")
                self.state.current_player = -self.state.current_player
        
        winner = self.state.get_winner()
        p1_name = agent1_name if forward else agent2_name
        p2_name = agent2_name if forward else agent1_name
        self.results[p1_name]["avg_pieces"] += x_pieces
        self.results[p2_name]["avg_pieces"] += o_pieces
        self.results[p1_name]["games_played"] += 1
        self.results[p2_name]["games_played"] += 1
        
        if winner == 1:
            self.results[p1_name]["wins"] += 1
            self.results[p2_name]["losses"] += 1
            print(f"Winner: {p1_name} (X)")
        elif winner == -1:
            self.results[p1_name]["losses"] += 1
            self.results[p2_name]["wins"] += 1
            print(f"Winner: {p2_name} (O)")
        else:
            self.results[p1_name]["draws"] += 1
            self.results[p2_name]["draws"] += 1
            print("Draw")
        
        if self.display == 'pygame':
            result_box_x = 10
            result_box_y = BOARD_SIZE * CELL_SIZE + 10
            result_box_width = BOARD_SIZE * CELL_SIZE + 280
            result_box_height = 80
            pygame.draw.rect(self.screen, (240, 240, 240), (result_box_x, result_box_y, result_box_width, result_box_height), 0)
            pygame.draw.rect(self.screen, (150, 150, 150), (result_box_x, result_box_y, result_box_width, result_box_height), 2)
            
            result_text = self.font.render(f"Winner: {'Draw' if winner == 0 else p1_name if winner == 1 else p2_name}", True, (50, 50, 50))
            self.screen.blit(result_text, (result_box_x + 20, result_box_y + 20))
            pygame.display.flip()
            await asyncio.sleep(2)

    def save_results(self):
        data = []
        map_name = self.map_file or f"Map_{self.map_idx + 1}"
        for name in self.results:
            if self.results[name]["games_played"] > 0:
                data.append({
                    'Agent': name,
                    'Wins': self.results[name]['wins'],
                    'Losses': self.results[name]['losses'],
                    'Draws': self.results[name]['draws'],
                    'AvgPieces': self.results[name]['avg_pieces'],
                    'TotalGames': self.results[name]['games_played'],
                    'Map': map_name
                })
        df = pd.DataFrame(data)
        output_path = '/kaggle/working/results.csv' if os.path.exists('/kaggle/working') else 'results.csv'
        if platform.system() == "Emscripten":
            print("Pyodide: Cannot save CSV. Results:")
            print(df.to_string(index=False))
        else:
            df.to_csv(output_path, index=False)
            print(f"Saved results to {output_path}")
        return df

    async def run_tournament(self):
        await self.run_menu()
        if not self.running and self.display == 'pygame':
            return
        
        print(f"\nMatch: {self.algo1} (X) vs {self.algo2} (O)")
        for game_num in range(self.games_per_match):
            await self.play_game(self.algo1, self.algo2, forward=True)
            if not self.running and self.display == 'pygame':
                break
        
        print(f"\nMatch: {self.algo2} (X) vs {self.algo1} (O)")
        for game_num in range(self.games_per_match):
            await self.play_game(self.algo1, self.algo2, forward=False)
            if not self.running and self.display == 'pygame':
                break
        
        print(f"\nTournament Results ({self.algo1} vs {self.algo2}):")
        if self.display == 'pygame':
            self.screen.fill((220, 220, 220))
            
            result_box_x = 50
            result_box_y = 50
            result_box_width = WINDOW_WIDTH - 100
            result_box_height = WINDOW_HEIGHT - 100
            pygame.draw.rect(self.screen, (240, 240, 240), (result_box_x, result_box_y, result_box_width, result_box_height), 0)
            pygame.draw.rect(self.screen, (150, 150, 150), (result_box_x, result_box_y, result_box_width, result_box_height), 2)
            
            # Tiêu đề
            title = self.font.render("Tournament Results", True, (50, 50, 50))
            self.screen.blit(title, (WINDOW_WIDTH//2 - title.get_width()//2, result_box_y + 20))
            
            y_offset = result_box_y + 80
            for name in self.results:
                if self.results[name]["games_played"] > 0:
                    self.results[name]["avg_pieces"] /= self.results[name]["games_played"]
                    result_text = (f"{name}: Wins={self.results[name]['wins']}, Losses={self.results[name]['losses']}, "
                                f"Draws={self.results[name]['draws']}, Avg Pieces={self.results[name]['avg_pieces']:.2f}")
                    print(result_text)
                    surface = self.font.render(result_text, True, (50, 50, 50))
                    self.screen.blit(surface, (result_box_x + 30, y_offset))
                    y_offset += 50
        
        else:
            for name in self.results:
                if self.results[name]["games_played"] > 0:
                    self.results[name]["avg_pieces"] /= self.results[name]["games_played"]
                    result_text = (f"{name}: Wins={self.results[name]['wins']}, Losses={self.results[name]['losses']}, "
                                f"Draws={self.results[name]['draws']}, Avg Pieces={self.results[name]['avg_pieces']:.2f}")
                    print(result_text)
        
        if self.display == 'pygame':
            pygame.display.flip()
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
        
        self.save_results()

def parse_args():
    parser = argparse.ArgumentParser(description="Ataxx Tournament with Pygame or Terminal")
    parser.add_argument("--map_file", type=str, default=None, help="Map file in 'map/' directory")
    parser.add_argument("--games", type=int, default=5, help="Games per match")
    parser.add_argument("--iterations", type=int, default=300, help="MCTS iterations")
    parser.add_argument("--algo1", type=str, default="MCTS_Domain_600", help="First agent")
    parser.add_argument("--algo2", type=str, default="Minimax+AB", help="Second agent")
    parser.add_argument("--display", type=str, default="terminal", help="Display mode: pygame or terminal")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay per move (seconds)")
    parser.add_argument("--first_player", type=str, default="W", help="First player (White or Black)")
    return parser.parse_args()

async def main():
    args = parse_args()
    try:
        game = AtaxxGame(
            map_file=args.map_file,
            games_per_match=args.games,
            iterations=args.iterations,
            algo1=args.algo1,
            algo2=args.algo2,
            display=args.display,
            delay=args.delay,
            first_player=args.first_player
        )
        game.init_pygame()
        await game.run_tournament()
        if game.display == 'pygame':
            pygame.quit()
    except ValueError as e:
        print(e)
        exit(1)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())