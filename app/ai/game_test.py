import asyncio
import platform
import argparse
import pygame
import numpy as np
import pandas as pd
import os
import glob
from ataxx_state import AtaxxState
from minimax_agent import MinimaxAgent
from mcts_agent import MCTSAgent
from mcts_domain_agent import MCTSDomainAgent
from ab_mcts_domain_agent import ABMCTSDomainAgent

COLORS = {
    'bg': (45, 52, 64),           
    'board': (236, 240, 241),    
    'grid': (189, 195, 199),      
    'player_x': (231, 76, 60),    
    'player_x_light': (192, 57, 43), 
    'player_o': (52, 152, 219),   
    'player_o_light': (41, 128, 185), 
    'blocked': (149, 165, 166),   
    'panel': (52, 73, 94),        
    'panel_light': (69, 90, 120), 
    'text': (220, 220, 220),     
    'text_dark': (44, 62, 80),  
    'accent': (155, 89, 182),     
    'success': (46, 204, 113),  
    'warning': (241, 196, 15),    
    'shadow': (44, 62, 80, 100),   
}

CELL_SIZE = 70
BOARD_SIZE = 7
PANEL_WIDTH = 320
WINDOW_WIDTH = BOARD_SIZE * CELL_SIZE + PANEL_WIDTH + 40
WINDOW_HEIGHT = max(BOARD_SIZE * CELL_SIZE + 100, 600)

MAP_FILES = {
    "position_3_3_empty_w.txt": """WWWWWWW\nBWWWWWW\nWWWBBBW\nWBBBBBW\nWBBBBBW\n#BBBBBB\n##WBBBW"""
}

def get_available_maps():
    """Get list of map files from map directory"""
    map_files = []
    
    if os.path.exists('map'):
        txt_files = glob.glob('map/*.txt')
        map_files = [os.path.basename(f) for f in txt_files]
    
    if not map_files:
        map_files = ["default_map.txt"]
    
    return sorted(map_files)

def read_map_file(filename):
    if not filename:
        lines = ["WWWWWWW", "BWWWWWW", "WWWBBBW", "WBBBBBW", "WBBBBBW", "#BBBBBB", "##WBBBW"]
    elif platform.system() == "Emscripten":
        if filename == "position_3_3_empty_w.txt":
            lines = ["WWWWWWW", "BWWWWWW", "WWWBBBW", "WBBBBBW", "WBBBBBW", "#BBBBBB", "##WBBBW"]
        else:
            raise ValueError(f"Map file {filename} not available in Pyodide")
    else:
        try:
            map_path = os.path.join("map", filename)
            with open(map_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Warning: Map file 'map/{filename}' not found, using default map")
            lines = ["WWWWWWW", "BWWWWWW", "WWWBBBW", "WBBBBBW", "WBBBBBW", "#BBBBBB", "##WBBBW"]
        except Exception as e:
            print(f"Warning: Error reading map file: {e}, using default map")
            lines = ["WWWWWWW", "BWWWWWW", "WWWBBBW", "WBBBBBW", "WBBBBBW", "#BBBBBB", "##WBBBW"]
    
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
        self.available_maps = get_available_maps()
        self.map_file = map_file if map_file in self.available_maps else self.available_maps[0] if self.available_maps else None
        self.games_per_match = games_per_match
        self.iterations = iterations
        self.delay = delay
        self.display = display.lower()
        self.first_player = 1 if first_player == 'W' else -1
        if self.display not in ['pygame', 'terminal']:
            raise ValueError("Display must be 'pygame' or 'terminal'")
        self.agents = {
            "Minimax+AB": MinimaxAgent(max_depth=4),
            "MCTS_300": MCTSAgent(iterations=iterations),
            "MCTS_Domain_300": MCTSDomainAgent(iterations=iterations),
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
        self.font_small = None
        self.font_large = None
        self.state = None
        self.running = True
        self.menu_active = self.display == 'pygame'
        self.selected_games = self.games_per_match
        self.selected_algo1 = self.algo1
        self.selected_algo2 = self.algo2
        self.agent_names = list(self.agents.keys())
        self.initial_board = read_map_file(self.map_file)
        self.fullscreen = False
        self.paused = False

    def init_pygame(self):
        if self.display != 'pygame':
            return
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Ataxx Tournament")
        
        try:
            self.font = pygame.font.SysFont('Segoe UI', 18, bold=False)
            self.font_small = pygame.font.SysFont('Segoe UI', 14, bold=False)
            self.font_large = pygame.font.SysFont('Segoe UI', 24, bold=True)
        except:
            self.font = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 18)
            self.font_large = pygame.font.Font(None, 32)

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        if self.fullscreen:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.fullscreen = False
            print("Switched to windowed mode")
        else:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.fullscreen = True
            print("Switched to fullscreen mode (Press F11 or ESC to exit)")

    def handle_keyboard_events(self, event):
        """Handle keyboard events"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                self.toggle_fullscreen()
            elif event.key == pygame.K_ESCAPE:
                if self.fullscreen:
                    self.toggle_fullscreen()
                else:
                    self.return_to_menu()
            elif event.key == pygame.K_SPACE:
                self.toggle_pause()

    def draw_pause_overlay(self):
        """Draw overlay when game is paused"""
        if not self.paused:
            return
            
        screen_width, screen_height = self.screen.get_size()
        
        if self.fullscreen:
            available_width = screen_width - PANEL_WIDTH - 100
            available_height = screen_height - 100
            
            max_cell_size_by_width = available_width // BOARD_SIZE
            max_cell_size_by_height = available_height // BOARD_SIZE
            cell_size = min(max_cell_size_by_width, max_cell_size_by_height, 100)
            
            board_width = BOARD_SIZE * cell_size
            board_height = BOARD_SIZE * cell_size
            board_x = (available_width - board_width) // 2 + 20
            board_y = (screen_height - board_height) // 2
        else:
            cell_size = CELL_SIZE
            board_x = 20
            board_y = 20
            board_width = BOARD_SIZE * cell_size
            board_height = BOARD_SIZE * cell_size
        
        board_overlay_rect = pygame.Rect(board_x, board_y, board_width + 20, board_height + 20)
        
        overlay = pygame.Surface((board_overlay_rect.width, board_overlay_rect.height))
        overlay.set_alpha(120)
        overlay.fill(COLORS['text_dark'])
        self.screen.blit(overlay, (board_overlay_rect.x, board_overlay_rect.y))
        
        pause_rect = pygame.Rect(
            board_overlay_rect.centerx - 150, 
            board_overlay_rect.centery - 75, 
            300, 
            150
        )
        
        self.draw_shadow(self.screen, pause_rect, 5)
        self.draw_gradient_rect(self.screen, pause_rect, COLORS['panel'], COLORS['panel_light'])
        pygame.draw.rect(self.screen, COLORS['panel'], pause_rect, border_radius=20)
        pygame.draw.rect(self.screen, COLORS['warning'], pause_rect, 4, border_radius=20)
        
        pause_text = self.font_large.render("PAUSED", True, COLORS['warning'])
        text_rect = pause_text.get_rect(center=(pause_rect.centerx, pause_rect.centery - 20))
        self.screen.blit(pause_text, text_rect)
        
        instruction_text = self.font.render("Press SPACE to resume", True, COLORS['text'])
        instruction_rect = instruction_text.get_rect(center=(pause_rect.centerx, pause_rect.centery + 20))
        self.screen.blit(instruction_text, instruction_rect)


    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            print("Game PAUSED (Press SPACE to resume)")
        else:
            print("Game RESUMED")

    def draw_control_panel(self):
        if self.display != 'pygame' or not self.state:
            return
            
        screen_width, screen_height = self.screen.get_size()
        
        panel_width = 280
        panel_height = 60
        panel_x = screen_width - panel_width - 20
        panel_y = screen_height - panel_height - 20
        
        control_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        
        self.draw_shadow(self.screen, control_rect, 3)
        pygame.draw.rect(self.screen, COLORS['panel'], control_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLORS['accent'], control_rect, 2, border_radius=10)
        
        mouse_pos = pygame.mouse.get_pos()
        buttons = {}
        
        button_width = 80
        button_height = 30
        button_x = panel_x + 10
        button_y = panel_y + 15
        
        pause_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        pause_text = "RESUME" if self.paused else "PAUSE"
        pause_color = COLORS['success'] if self.paused else COLORS['warning']
        
        is_hovered = pause_rect.collidepoint(mouse_pos)
        button_color = pause_color if not is_hovered else COLORS['accent']
        
        pygame.draw.rect(self.screen, button_color, pause_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLORS['text'], pause_rect, 1, border_radius=6)
        
        text_surface = self.font_small.render(pause_text, True, COLORS['text'])
        text_rect = text_surface.get_rect(center=pause_rect.center)
        self.screen.blit(text_surface, text_rect)
        
        buttons['pause'] = pause_rect
        
        menu_button_width = 90
        menu_button_x = button_x + button_width + 10
        menu_rect = pygame.Rect(menu_button_x, button_y, menu_button_width, button_height)
        
        is_menu_hovered = menu_rect.collidepoint(mouse_pos)
        menu_button_color = COLORS['player_x'] if not is_menu_hovered else COLORS['accent']
        
        pygame.draw.rect(self.screen, menu_button_color, menu_rect, border_radius=6)
        pygame.draw.rect(self.screen, COLORS['text'], menu_rect, 1, border_radius=6)
        
        menu_text_surface = self.font_small.render("MENU", True, COLORS['text'])
        menu_text_rect = menu_text_surface.get_rect(center=menu_rect.center)
        self.screen.blit(menu_text_surface, menu_text_rect)
        
        buttons['menu'] = menu_rect
        
        speed_text = f"Speed: {2.0 - self.delay:.1f}x"
        speed_surface = self.font_small.render(speed_text, True, COLORS['text'])
        self.screen.blit(speed_surface, (panel_x + 10, panel_y + panel_height - 15))
        
        return buttons

    def draw_shadow(self, surface, rect, offset=3):
        shadow_rect = rect.copy()
        shadow_rect.x += offset
        shadow_rect.y += offset
        shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height))
        shadow_surface.set_alpha(50)
        shadow_surface.fill(COLORS['text_dark'])
        surface.blit(shadow_surface, shadow_rect)

    def draw_gradient_rect(self, surface, rect, color1, color2, vertical=True):
        if vertical:
            for y in range(rect.height):
                ratio = y / rect.height
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                pygame.draw.line(surface, (r, g, b), 
                               (rect.x, rect.y + y), (rect.x + rect.width, rect.y + y))

    def draw_board(self):
        if not self.state:
            return
        if self.display == 'terminal':
            self.state.display_board()
            return
        
        screen_width, screen_height = self.screen.get_size()
        
        self.draw_gradient_rect(self.screen, pygame.Rect(0, 0, screen_width, screen_height),
                               COLORS['bg'], (COLORS['bg'][0]+10, COLORS['bg'][1]+10, COLORS['bg'][2]+10))
        
        if self.fullscreen:
            available_width = screen_width - PANEL_WIDTH - 100  
            available_height = screen_height - 100 
            
            max_cell_size_by_width = available_width // BOARD_SIZE
            max_cell_size_by_height = available_height // BOARD_SIZE
            cell_size = min(max_cell_size_by_width, max_cell_size_by_height, 100) 
            
            board_width = BOARD_SIZE * cell_size
            board_height = BOARD_SIZE * cell_size
            board_x = (available_width - board_width) // 2 + 20
            board_y = (screen_height - board_height) // 2
        else:
            cell_size = CELL_SIZE
            board_x = 20
            board_y = 20
            board_width = BOARD_SIZE * cell_size
            board_height = BOARD_SIZE * cell_size
        
        board_rect = pygame.Rect(board_x, board_y, board_width + 20, board_height + 20)
        self.draw_shadow(self.screen, board_rect)
        pygame.draw.rect(self.screen, COLORS['board'], board_rect, border_radius=15)
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x = board_x + 10 + c * cell_size
                y = board_y + 10 + r * cell_size
                cell_rect = pygame.Rect(x, y, cell_size, cell_size)
                
                pygame.draw.rect(self.screen, COLORS['board'], cell_rect)
                pygame.draw.rect(self.screen, COLORS['grid'], cell_rect, 1)
                
                if self.state.board[r][c] == 1:  
                    center = (x + cell_size//2, y + cell_size//2)
                    radius_outer = cell_size//3 + 2
                    radius_inner = cell_size//3 - 2
                    radius_highlight = cell_size//6
                    
                    pygame.draw.circle(self.screen, COLORS['player_x_light'], center, radius_outer)
                    pygame.draw.circle(self.screen, COLORS['player_x'], center, radius_inner)
                    highlight_offset = min(8, cell_size//8)
                    pygame.draw.circle(self.screen, (255, 255, 255, 100), 
                                     (center[0] - highlight_offset, center[1] - highlight_offset), radius_highlight)
                    
                elif self.state.board[r][c] == -1:  
                    center = (x + cell_size//2, y + cell_size//2)
                    radius_outer = cell_size//3 + 2
                    radius_inner = cell_size//3 - 2
                    radius_highlight = cell_size//6
                    
                    pygame.draw.circle(self.screen, COLORS['player_o_light'], center, radius_outer)
                    pygame.draw.circle(self.screen, COLORS['player_o'], center, radius_inner)
                    highlight_offset = min(8, cell_size//8)
                    pygame.draw.circle(self.screen, (255, 255, 255, 100), 
                                     (center[0] - highlight_offset, center[1] - highlight_offset), radius_highlight)
                    
                elif self.state.board[r][c] == -2:  
                    margin = max(5, cell_size//14)
                    pygame.draw.rect(self.screen, COLORS['blocked'], 
                                   (x + margin, y + margin, cell_size - 2*margin, cell_size - 2*margin), 
                                   border_radius=8)
        
        self.draw_info_panel()
        
        pause_button = self.draw_control_panel()
        
        self.draw_pause_overlay()
        
        pygame.display.flip()
        return pause_button

    def return_to_menu(self):
        self.menu_active = True
        self.paused = False
        self.state = None
        print("Returning to main menu...")

    def draw_info_panel(self):
        screen_width, screen_height = self.screen.get_size()
        
        if self.fullscreen:
            available_width = screen_width - PANEL_WIDTH - 100
            panel_x = available_width + 50
            panel_width = min(PANEL_WIDTH, screen_width - panel_x - 30)
            panel_height = screen_height - 140 
        else:
            panel_x = BOARD_SIZE * CELL_SIZE + 50
            panel_width = PANEL_WIDTH - 30
            panel_height = WINDOW_HEIGHT - 140 
            
        panel_y = 20
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        
        self.draw_shadow(self.screen, panel_rect)
        self.draw_gradient_rect(self.screen, panel_rect, COLORS['panel'], COLORS['panel_light'])
        pygame.draw.rect(self.screen, COLORS['panel'], panel_rect, border_radius=15)
        pygame.draw.rect(self.screen, COLORS['accent'], panel_rect, 3, border_radius=15)
        
        y_offset = panel_y + 20
        
        title_text = self.font_large.render("Game Info", True, COLORS['text'])
        self.screen.blit(title_text, (panel_x + 20, y_offset))
        y_offset += 50
        
        status_text = "PAUSED" if self.paused else "RUNNING"
        status_color = COLORS['warning'] if self.paused else COLORS['success']
        status_surface = self.font.render(f"Status: {status_text}", True, status_color)
        self.screen.blit(status_surface, (panel_x + 20, y_offset))
        y_offset += 35
        
        if self.state:
            x_pieces = np.sum(self.state.board == 1)
            o_pieces = np.sum(self.state.board == -1)
            current_player = "X" if self.state.current_player == 1 else "O"
            
            if self.map_file and self.map_file in self.available_maps:
                map_idx = self.available_maps.index(self.map_file)
                map_display = f"Map {map_idx}: {self.map_file.replace('.txt', '')}"
            else:
                map_display = "Map 1: Default"
            
            stats_data = [
                (f"Red {self.algo1}: {x_pieces} pieces", COLORS['player_x']),
                (f"Blue {self.algo2}: {o_pieces} pieces", COLORS['player_o']),
                (f"Current Turn: Player {current_player}", COLORS['accent']),
                (map_display, COLORS['text_dark']),
            ]
            
            for text, color in stats_data:
                stat_rect = pygame.Rect(panel_x + 10, y_offset - 5, panel_width - 30, 35)
                pygame.draw.rect(self.screen, (255, 255, 255, 20), stat_rect, border_radius=8)
                
                text_surface = self.font.render(text, True, color)
                self.screen.blit(text_surface, (panel_x + 20, y_offset))
                y_offset += 45
        
        y_offset += 20
        controls_title = self.font.render("Controls:", True, COLORS['text'])
        self.screen.blit(controls_title, (panel_x + 20, y_offset))
        y_offset += 25
        
        controls = [
            "SPACE - Pause/Resume",
            "ESC - Return to Menu",
            "F11 - Fullscreen",
        ]
        
        for control in controls:
            control_surface = self.font_small.render(control, True, COLORS['text'])
            self.screen.blit(control_surface, (panel_x + 20, y_offset))
            y_offset += 20
        
        y_offset += 15
        progress_title = self.font.render("Match Progress", True, COLORS['text'])
        self.screen.blit(progress_title, (panel_x + 20, y_offset))
        y_offset += 35
        
        for name in self.results:
            if self.results[name]["games_played"] > 0:
                wins = self.results[name]["wins"]
                losses = self.results[name]["losses"]
                draws = self.results[name]["draws"]
                
                result_rect = pygame.Rect(panel_x + 10, y_offset - 5, panel_width - 30, 30)
                pygame.draw.rect(self.screen, (255, 255, 255, 15), result_rect, border_radius=6)
                
                result_text = f"{name}: {wins}W-{losses}L-{draws}D"
                color = COLORS['success'] if wins > losses else COLORS['warning'] if wins == losses else COLORS['player_x']
                
                text_surface = self.font_small.render(result_text, True, color)
                self.screen.blit(text_surface, (panel_x + 15, y_offset))
                y_offset += 35

    def draw_modern_button(self, surface, rect, text, font, base_color, hover_color, text_color, mouse_pos):
        """Vẽ button hiện đại với hover effect"""
        is_hovered = rect.collidepoint(mouse_pos)
        color = hover_color if is_hovered else base_color
        
        self.draw_shadow(surface, rect, 2)
        
        pygame.draw.rect(surface, color, rect, border_radius=8)
        pygame.draw.rect(surface, COLORS['accent'], rect, 2, border_radius=8)
        
        text_surface = font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)
        
        return rect
    
    def draw_menu(self):
        """Menu với thiết kế hiện đại"""
        screen_width, screen_height = self.screen.get_size()
        
        self.draw_gradient_rect(self.screen, pygame.Rect(0, 0, screen_width, screen_height),
                               COLORS['bg'], (COLORS['bg'][0]+15, COLORS['bg'][1]+15, COLORS['bg'][2]+15))
        
        title = self.font_large.render("Ataxx Tournament", True, COLORS['text'])
        title_rect = title.get_rect(center=(screen_width//2, 60))
        title_shadow = self.font_large.render("Ataxx Tournament", True, COLORS['text_dark'])
        self.screen.blit(title_shadow, (title_rect.x + 3, title_rect.y + 3))
        self.screen.blit(title, title_rect)
        
        hint_text = "Press F11 for fullscreen, ESC to exit fullscreen"
        hint_surface = self.font_small.render(hint_text, True, COLORS['text'])
        hint_rect = hint_surface.get_rect(center=(screen_width//2, 90))
        self.screen.blit(hint_surface, hint_rect)
        
        margin = 50
        menu_rect = pygame.Rect(margin, 120, screen_width - 2*margin, screen_height - 200)
        self.draw_shadow(self.screen, menu_rect)
        self.draw_gradient_rect(self.screen, menu_rect, COLORS['panel'], COLORS['panel_light'])
        pygame.draw.rect(self.screen, COLORS['panel'], menu_rect, border_radius=20)
        pygame.draw.rect(self.screen, COLORS['accent'], menu_rect, 3, border_radius=20)
        
        mouse_pos = pygame.mouse.get_pos()
        buttons = {}
        
        if self.map_file and self.map_file in self.available_maps:
            map_idx = self.available_maps.index(self.map_file)
            map_display = f"Map {map_idx}: {self.map_file.replace('.txt', '')}"
        else:
            map_display = "Map 1: Default"
        
        options = [
            (map_display, "maps_prev", "maps_next"),
            (f"Games per Match: {self.selected_games}", "games_prev", "games_next"),
            (f"Player X: {self.selected_algo1}", "algo1_prev", "algo1_next"),
            (f"Player O: {self.selected_algo2}", "algo2_prev", "algo2_next"),
            (f"First Player: {'X (Red)' if self.first_player == 1 else 'O (Blue)'}", "first_player_prev", "first_player_next")
        ]
        
        y_start = menu_rect.y + 40
        for i, (text, prev_key, next_key) in enumerate(options):
            y_pos = y_start + i * 70
            
            option_rect = pygame.Rect(menu_rect.x + 20, y_pos - 10, menu_rect.width - 40, 50)
            pygame.draw.rect(self.screen, (255, 255, 255, 10), option_rect, border_radius=10)
            
            text_surface = self.font.render(text, True, COLORS['text_dark'])
            self.screen.blit(text_surface, (menu_rect.x + 40, y_pos))
            
            if prev_key and next_key:
                button_y = y_pos - 5
                prev_rect = pygame.Rect(menu_rect.right - 120, button_y, 45, 35)
                next_rect = pygame.Rect(menu_rect.right - 65, button_y, 45, 35)
                
                buttons[prev_key] = self.draw_modern_button(
                    self.screen, prev_rect, "<", self.font, 
                    COLORS['panel_light'], COLORS['accent'], COLORS['text'], mouse_pos
                )
                buttons[next_key] = self.draw_modern_button(
                    self.screen, next_rect, ">", self.font,
                    COLORS['panel_light'], COLORS['accent'], COLORS['text'], mouse_pos
                )
        
        start_rect = pygame.Rect(screen_width//2 - 100, menu_rect.bottom - 80, 200, 50)
        buttons['start_button'] = self.draw_modern_button(
            self.screen, start_rect, "START TOURNAMENT", self.font,
            COLORS['success'], COLORS['accent'], COLORS['text'], mouse_pos
        )
        
        pygame.display.flip()
        return buttons

    async def run_menu(self):
        if self.display != 'pygame':
            return
        while self.menu_active and self.running:
            buttons = self.draw_menu()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.menu_active = False
                    return  
                elif event.type == pygame.KEYDOWN:
                    self.handle_keyboard_events(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos

                    if 'maps_prev' in buttons and buttons['maps_prev'].collidepoint(pos):
                        if self.available_maps:
                            current_idx = self.available_maps.index(self.map_file) if self.map_file in self.available_maps else 0
                            self.map_file = self.available_maps[(current_idx - 1) % len(self.available_maps)]
                            self.initial_board = read_map_file(self.map_file)
                    elif 'maps_next' in buttons and buttons['maps_next'].collidepoint(pos):
                        if self.available_maps:
                            current_idx = self.available_maps.index(self.map_file) if self.map_file in self.available_maps else 0
                            self.map_file = self.available_maps[(current_idx + 1) % len(self.available_maps)]
                            self.initial_board = read_map_file(self.map_file)
                    elif 'games_prev' in buttons and buttons['games_prev'].collidepoint(pos):
                        self.selected_games = max(1, self.selected_games - 1)
                    elif 'games_next' in buttons and buttons['games_next'].collidepoint(pos):
                        self.selected_games = min(10, self.selected_games + 1)
                    elif 'algo1_prev' in buttons and buttons['algo1_prev'].collidepoint(pos):
                        idx = self.agent_names.index(self.selected_algo1)
                        self.selected_algo1 = self.agent_names[(idx - 1) % len(self.agent_names)]
                    elif 'algo1_next' in buttons and buttons['algo1_next'].collidepoint(pos):
                        idx = self.agent_names.index(self.selected_algo1)
                        self.selected_algo1 = self.agent_names[(idx + 1) % len(self.agent_names)]
                    elif 'algo2_prev' in buttons and buttons['algo2_prev'].collidepoint(pos):
                        idx = self.agent_names.index(self.selected_algo2)
                        self.selected_algo2 = self.agent_names[(idx - 1) % len(self.agent_names)]
                    elif 'algo2_next' in buttons and buttons['algo2_next'].collidepoint(pos):
                        idx = self.agent_names.index(self.selected_algo2)
                        self.selected_algo2 = self.agent_names[(idx + 1) % len(self.agent_names)]
                    elif 'first_player_prev' in buttons and buttons['first_player_prev'].collidepoint(pos):
                        self.first_player = 1 if self.first_player == -1 else -1
                    elif 'first_player_next' in buttons and buttons['first_player_next'].collidepoint(pos):
                        self.first_player = -1 if self.first_player == 1 else 1
                    elif 'start_button' in buttons and buttons['start_button'].collidepoint(pos):
                        self.games_per_match = self.selected_games
                        self.algo1 = self.selected_algo1
                        self.algo2 = self.selected_algo2
                        self.results = {name: {"wins": 0, "losses": 0, "draws": 0, "avg_pieces": 0, "games_played": 0} 
                                      for name in [self.algo1, self.algo2]}
                        self.menu_active = False
                        
            await asyncio.sleep(0.016)

                    if 'maps_prev' in buttons and buttons['maps_prev'].collidepoint(pos):
                        if self.available_maps:
                            current_idx = self.available_maps.index(self.map_file) if self.map_file in self.available_maps else 0
                            self.map_file = self.available_maps[(current_idx - 1) % len(self.available_maps)]
                            self.initial_board = read_map_file(self.map_file)
                    elif 'maps_next' in buttons and buttons['maps_next'].collidepoint(pos):
                        if self.available_maps:
                            current_idx = self.available_maps.index(self.map_file) if self.map_file in self.available_maps else 0
                            self.map_file = self.available_maps[(current_idx + 1) % len(self.available_maps)]
                            self.initial_board = read_map_file(self.map_file)
                    elif 'games_prev' in buttons and buttons['games_prev'].collidepoint(pos):
                        self.selected_games = max(1, self.selected_games - 1)
                    elif 'games_next' in buttons and buttons['games_next'].collidepoint(pos):
                        self.selected_games = min(10, self.selected_games + 1)
                    elif 'algo1_prev' in buttons and buttons['algo1_prev'].collidepoint(pos):
                        idx = self.agent_names.index(self.selected_algo1)
                        self.selected_algo1 = self.agent_names[(idx - 1) % len(self.agent_names)]
                    elif 'algo1_next' in buttons and buttons['algo1_next'].collidepoint(pos):
                        idx = self.agent_names.index(self.selected_algo1)
                        self.selected_algo1 = self.agent_names[(idx + 1) % len(self.agent_names)]
                    elif 'algo2_prev' in buttons and buttons['algo2_prev'].collidepoint(pos):
                        idx = self.agent_names.index(self.selected_algo2)
                        self.selected_algo2 = self.agent_names[(idx - 1) % len(self.agent_names)]
                    elif 'algo2_next' in buttons and buttons['algo2_next'].collidepoint(pos):
                        idx = self.agent_names.index(self.selected_algo2)
                        self.selected_algo2 = self.agent_names[(idx + 1) % len(self.agent_names)]
                    elif 'first_player_prev' in buttons and buttons['first_player_prev'].collidepoint(pos):
                        self.first_player = 1 if self.first_player == -1 else -1
                    elif 'first_player_next' in buttons and buttons['first_player_next'].collidepoint(pos):
                        self.first_player = -1 if self.first_player == 1 else 1
                    elif 'start_button' in buttons and buttons['start_button'].collidepoint(pos):
                        self.menu_active = False
                        self.games_per_match = self.selected_games
                        self.algo1 = self.selected_algo1
                        self.algo2 = self.selected_algo2
                        self.results = {name: {"wins": 0, "losses": 0, "draws": 0, "avg_pieces": 0, "games_played": 0} 
                                        for name in [self.algo1, self.algo2]}

    async def play_game(self, agent1_name, agent2_name, forward=True):
        self.state = AtaxxState(initial_board=self.initial_board, current_player=self.first_player)
        
        if self.map_file and self.map_file in self.available_maps:
            map_idx = self.available_maps.index(self.map_file)
            map_name = f"Map {map_idx}: {self.map_file.replace('.txt', '')}"
        else:
            map_name = "Map 1: Default"
            
        print(f"\nGame ({'Forward' if forward else 'Reverse'}) on {map_name}")
        legal_moves = self.state.get_legal_moves()
        if not legal_moves:
            print(f"Warning: No initial legal moves for player {self.state.current_player}")
        
        control_buttons = self.draw_board()
        
        move_count = 0
        x_pieces, o_pieces = 0, 0

        while not self.state.is_game_over() and self.running and not self.menu_active:
            if self.display == 'pygame':
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        return 
                    elif event.type == pygame.KEYDOWN:
                        self.handle_keyboard_events(event)
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if control_buttons:
                            if 'pause' in control_buttons and control_buttons['pause'].collidepoint(event.pos):
                                self.toggle_pause()
                                control_buttons = self.draw_board()  
                            elif 'menu' in control_buttons and control_buttons['menu'].collidepoint(event.pos):
                                self.return_to_menu()
                                return
            
            if not self.running or self.menu_active:
                return
            
            if self.paused:
                if self.display == 'pygame':
                    control_buttons = self.draw_board() 
                await asyncio.sleep(0.1)
                continue
                
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

            if not self.running or self.menu_active:
                return

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
                control_buttons = self.draw_board()
                
                await asyncio.sleep(2.0)
                
                if self.display == 'pygame':
                    sleep_time = max(0.1, self.delay)
                    sleep_steps = int(sleep_time / 0.1)
                    for i in range(sleep_steps):
                        await asyncio.sleep(0.1)
                        
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.running = False
                                return
                            elif event.type == pygame.KEYDOWN:
                                self.handle_keyboard_events(event)
                            elif event.type == pygame.MOUSEBUTTONDOWN:
                                if control_buttons:
                                    if 'pause' in control_buttons and control_buttons['pause'].collidepoint(event.pos):
                                        self.toggle_pause()
                                        control_buttons = self.draw_board()
                                    elif 'menu' in control_buttons and control_buttons['menu'].collidepoint(event.pos):
                                        self.return_to_menu()
                                        return
                        
                        if not self.running or self.menu_active:
                            return
                        
                        while self.paused and self.running and not self.menu_active:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    self.running = False
                                    return
                                elif event.type == pygame.KEYDOWN:
                                    self.handle_keyboard_events(event)
                                elif event.type == pygame.MOUSEBUTTONDOWN:
                                    if control_buttons:
                                        if 'pause' in control_buttons and control_buttons['pause'].collidepoint(event.pos):
                                            self.toggle_pause()
                                            control_buttons = self.draw_board()
                                        elif 'menu' in control_buttons and control_buttons['menu'].collidepoint(event.pos):
                                            self.return_to_menu()
                                            return
                            control_buttons = self.draw_board()
                            await asyncio.sleep(0.1)
            else:
                print(f"\n{agent_name} has no legal moves - PASS")
                self.state.current_player = -self.state.current_player
        
        if not self.running or self.menu_active:
            return
            
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
        
        if self.display == 'pygame' and self.running and not self.menu_active:
            self.draw_game_result(winner, p1_name, p2_name)
            for i in range(20):  
                await asyncio.sleep(0.1)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        return
                    elif event.type == pygame.KEYDOWN:
                        self.handle_keyboard_events(event)
                if not self.running or self.menu_active:
                    return

    def draw_game_result(self, winner, p1_name, p2_name):
        screen_width, screen_height = self.screen.get_size()
        
        overlay = pygame.Surface((screen_width, screen_height))
        overlay.set_alpha(180)
        overlay.fill(COLORS['text_dark'])
        self.screen.blit(overlay, (0, 0))
        
        box_width = min(screen_width//2, 400)
        box_height = 200
        result_rect = pygame.Rect(screen_width//2 - box_width//2, screen_height//2 - box_height//2, box_width, box_height)
        
        self.draw_shadow(self.screen, result_rect, 5)
        self.draw_gradient_rect(self.screen, result_rect, COLORS['panel'], COLORS['panel_light'])
        pygame.draw.rect(self.screen, COLORS['panel'], result_rect, border_radius=20)
        pygame.draw.rect(self.screen, COLORS['accent'], result_rect, 4, border_radius=20)
        
        if winner == 0:
            result_text = "DRAW!"
            color = COLORS['warning']
        elif winner == 1:
            result_text = f"{p1_name} WINS!"
            color = COLORS['success']
        else:
            result_text = f"{p2_name} WINS!"
            color = COLORS['success']
        
        text_surface = self.font_large.render(result_text, True, color)
        text_rect = text_surface.get_rect(center=result_rect.center)
        self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()

    def save_results(self):
        data = []
        
        if self.map_file and self.map_file in self.available_maps:
            map_idx = self.available_maps.index(self.map_file)
            map_name = f"Map_{map_idx}_{self.map_file.replace('.txt', '')}"
        else:
            map_name = "Map_1_Default"
            
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for name in self.results:
            if self.results[name]["games_played"] > 0:
                avg_pieces = self.results[name]['avg_pieces'] / self.results[name]['games_played']
                data.append({
                    'Timestamp': timestamp,
                    'Agent': name,
                    'Wins': self.results[name]['wins'],
                    'Losses': self.results[name]['losses'],
                    'Draws': self.results[name]['draws'],
                    'AvgPieces': round(avg_pieces, 2),
                    'TotalGames': self.results[name]['games_played'],
                    'Map': map_name,
                    'Opponent': self.algo2 if name == self.algo1 else self.algo1,
                    'MatchID': f"{self.algo1}_vs_{self.algo2}_{map_name}_{timestamp.replace(':', '-').replace(' ', '_')}"
                })
        
        new_df = pd.DataFrame(data)
        output_path = '/kaggle/working/results.csv' if os.path.exists('/kaggle/working') else 'results.csv'
        
        if platform.system() == "Emscripten":
            print("Pyodide: Cannot save CSV. Results:")
            print(new_df.to_string(index=False))
            return new_df
        
        try:
            if os.path.exists(output_path):
                print(f"📖 Reading existing results from {output_path}")
                existing_df = pd.read_csv(output_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                print(f"📊 Added {len(new_df)} new records to existing {len(existing_df)} records")
            else:
                print(f"📝 Creating new results file {output_path}")
                combined_df = new_df
            
            combined_df.to_csv(output_path, index=False)
            print(f"💾 Saved {len(combined_df)} total records to {output_path}")
            
            print(f"\n📈 Latest Match Results ({map_name}):")
            for _, row in new_df.iterrows():
                win_rate = (row['Wins'] / row['TotalGames'] * 100) if row['TotalGames'] > 0 else 0
                print(f"  {row['Agent']}: {row['Wins']}W-{row['Losses']}L-{row['Draws']}D "
                    f"({win_rate:.1f}% win rate, {row['AvgPieces']} avg pieces)")
            
            if len(combined_df) > len(new_df):
                print(f"\n📊 Historical Summary (All Matches):")
                historical_summary = combined_df.groupby('Agent').agg({
                    'Wins': 'sum',
                    'Losses': 'sum', 
                    'Draws': 'sum',
                    'TotalGames': 'sum',
                    'AvgPieces': 'mean'
                }).round(2)
                
                for agent, stats in historical_summary.iterrows():
                    total_games = stats['TotalGames']
                    win_rate = (stats['Wins'] / total_games * 100) if total_games > 0 else 0
                    print(f"  {agent}: {int(stats['Wins'])}W-{int(stats['Losses'])}L-{int(stats['Draws'])}D "
                        f"({win_rate:.1f}% win rate, {stats['AvgPieces']:.2f} avg pieces, {int(total_games)} total games)")
            
            return combined_df
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            print("Fallback - Results printed to console:")
            print(new_df.to_string(index=False))
            return new_df

    async def run_tournament(self):
        while self.running:
            await self.run_menu()
            if not self.running:
                return
            
            print(f"\nMatch: {self.algo1} (X) vs {self.algo2} (O)")
            for game_num in range(self.games_per_match):
                if not self.running or self.menu_active:
                    break
                await self.play_game(self.algo1, self.algo2, forward=True)
                if not self.running or self.menu_active:
                    break
            
            if not self.running or self.menu_active:
                if self.menu_active:
                    continue
                else:
                    return
                
            print(f"\nMatch: {self.algo2} (X) vs {self.algo1} (O)")
            for game_num in range(self.games_per_match):
                if not self.running or self.menu_active:
                    break
                await self.play_game(self.algo1, self.algo2, forward=False)
                if not self.running or self.menu_active:
                    break
            
            if not self.running or self.menu_active:
                if self.menu_active:
                    continue
                else:
                    return
                
            print(f"\nTournament Results ({self.algo1} vs {self.algo2}):")
            
            if self.display == 'pygame':
                if self.running and not self.menu_active:
                    self.draw_final_results()
                while self.running and not self.menu_active:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                            break
                        elif event.type == pygame.KEYDOWN:
                            self.handle_keyboard_events(event)
                    await asyncio.sleep(0.1)
            else:
                for name in self.results:
                    if self.results[name]["games_played"] > 0:
                        self.results[name]["avg_pieces"] /= self.results[name]["games_played"]
                        result_text = (f"{name}: Wins={self.results[name]['wins']}, Losses={self.results[name]['losses']}, "
                                    f"Draws={self.results[name]['draws']}, Avg Pieces={self.results[name]['avg_pieces']:.2f}")
                        print(result_text)
            
            if self.running and not self.menu_active:
                self.save_results()
                break

    def draw_final_results(self):
        screen_width, screen_height = self.screen.get_size()
        
        self.draw_gradient_rect(self.screen, pygame.Rect(0, 0, screen_width, screen_height),
                               COLORS['bg'], (COLORS['bg'][0]+10, COLORS['bg'][1]+10, COLORS['bg'][2]+10))
        
        margin = 50
        result_rect = pygame.Rect(margin, margin, screen_width - 2*margin, screen_height - 2*margin)
        self.draw_shadow(self.screen, result_rect, 5)
        self.draw_gradient_rect(self.screen, result_rect, COLORS['panel'], COLORS['panel_light'])
        pygame.draw.rect(self.screen, COLORS['panel'], result_rect, border_radius=20)
        pygame.draw.rect(self.screen, COLORS['accent'], result_rect, 4, border_radius=20)
        
        title = self.font_large.render("Tournament Results", True, COLORS['text'])
        title_rect = title.get_rect(center=(screen_width//2, result_rect.y + 50))
        self.screen.blit(title, title_rect)
        
        y_offset = result_rect.y + 120
        
        for name in self.results:
            if self.results[name]["games_played"] > 0:
                avg_pieces = self.results[name]['avg_pieces'] / self.results[name]['games_played']
                wins = self.results[name]['wins']
                losses = self.results[name]['losses']
                draws = self.results[name]['draws']
                
                agent_rect = pygame.Rect(result_rect.x + 30, y_offset - 10, result_rect.width - 60, 80)
                pygame.draw.rect(self.screen, (255, 255, 255, 20), agent_rect, border_radius=12)
                
                agent_text = self.font_large.render(name, True, COLORS['text'])
                self.screen.blit(agent_text, (agent_rect.x + 20, y_offset))
                
                stats_text = f"{wins}W  {losses}L  {draws}D  {avg_pieces:.1f} avg pieces"
                stats_surface = self.font.render(stats_text, True, COLORS['text'])
                self.screen.blit(stats_surface, (agent_rect.x + 20, y_offset + 35))
                
                total_games = wins + losses + draws
                if total_games > 0:
                    win_rate = wins / total_games
                    bar_width = min(300, agent_rect.width - 40)
                    bar_rect = pygame.Rect(agent_rect.x + 20, y_offset + 60, bar_width, 8)
                    pygame.draw.rect(self.screen, COLORS['grid'], bar_rect, border_radius=4)
                    fill_width = int(bar_width * win_rate)
                    if fill_width > 0:
                        fill_rect = pygame.Rect(agent_rect.x + 20, y_offset + 60, fill_width, 8)
                        color = COLORS['success'] if win_rate > 0.5 else COLORS['warning'] if win_rate == 0.5 else COLORS['player_x']
                        pygame.draw.rect(self.screen, color, fill_rect, border_radius=4)
                
                y_offset += 120
        
        pygame.display.flip()

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
    except ValueError as e:
        print(e)
        exit(1)
    finally:
        if game.display == 'pygame':
            pygame.quit()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())