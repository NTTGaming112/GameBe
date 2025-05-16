#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the Ataxx API.

This script sends a sample request to the /get_move endpoint to verify
the API is working correctly.
"""
import json
import requests
import time
from app.ai.constants import BOARD_SIZE, PLAYER_ONE, PLAYER_TWO, EMPTY_CELL

# Define API URL
API_URL = "http://localhost:5000/get_move"  # Using default Flask port

# Create a sample 7x7 Ataxx board using constants
# Empty cells are represented by None, red cells by 'red', and yellow cells by 'yellow'
sample_board = [
    [None] * BOARD_SIZE for _ in range(BOARD_SIZE)
]

# Set initial positions (standard Ataxx setup with pieces in corners)
sample_board[0][0] = 'red'     # Top-left
sample_board[0][6] = 'yellow'  # Top-right
sample_board[6][0] = 'yellow'  # Bottom-left
sample_board[6][6] = 'red'     # Bottom-right
sample_board[3][3] = 'red'     # Center
sample_board[3][4] = 'yellow'  # Center-right
sample_board[4][3] = 'yellow'  # Center-left
sample_board[4][4] = 'red'     # Center-bottom
sample_board[2][2] = 'red'     # Near center
sample_board[2][4] = 'yellow'  # Near center-right
sample_board[4][2] = 'yellow'  # Near center-left
sample_board[2][3] = 'yellow'  # Near center-bottom
sample_board[3][2] = 'red'     # Near center-top
sample_board[5][5] = 'red'     # Near bottom-right
sample_board[5][1] = 'yellow'  # Near bottom-left
sample_board[1][5] = 'yellow'  # Near top-right
sample_board[1][1] = 'red'     # Near top-left
sample_board[3][1] = 'red'     # Near center-left
sample_board[1][3] = 'yellow'  # Near top-center
sample_board[5][3] = 'yellow'  # Near bottom-center
sample_board[3][5] = 'red'     # Near center-right

# Convert board to numeric for API
mapping = {'red': 1, 'yellow': -1, None: 0}
numeric_board = [[mapping[cell] for cell in row] for row in sample_board]

# Request data
request_data = {
    "board": numeric_board,
    "current_player": "red",
    "algorithm": "MCD",  # Using Basic Monte Carlo
    "iterations": 10,  # Using fewer iterations for faster testing
    "depth": 2,  # Search depth for Minimax algorithm
    "policy_args": {
        "use_simulation_formula": False,
        "s1_ratio": 1.0,
        "s2_ratio": 1.0,
        "s3_ratio": 0.5
    }
}

print("Sending request to API...")
print(f"Request data: {json.dumps(request_data)[:100]}...")

try:
    # Send POST request to API
    start_time = time.time()
    response = requests.post(API_URL, json=request_data)
    end_time = time.time()
    
    print(f"Request completed in {end_time - start_time:.2f} seconds")
    print(f"Status code: {response.status_code}")
    
    # Print response
    if response.status_code == 200:
        response_data = response.json()
        print(f"Response: {json.dumps(response_data, indent=2)}")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Error making request: {str(e)}")