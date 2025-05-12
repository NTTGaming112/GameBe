#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

def minimax(board, state, depth_minimax):
	"""Basic Minimax implementation without heuristics."""
	def max_value(state, depth, alpha, beta):
		if depth == 0 or board.is_gameover(state):
			return board.get_score(state, state.player), None
			
		moves = board.legal_plays(state)
		if not moves:
			return board.get_score(state, state.player), None
			
		best_score = float('-inf')
		best_move = None
		
		for move in moves:
			next_state = board.next_state(state, move)
			score, _ = min_value(next_state, depth - 1, alpha, beta)
			
			if score > best_score:
				best_score = score
				best_move = move
				
			alpha = max(alpha, best_score)
			if beta <= alpha:
				break
				
		return best_score, best_move

	def min_value(state, depth, alpha, beta):
		if depth == 0 or board.is_gameover(state):
			return board.get_score(state, state.player), None
			
		moves = board.legal_plays(state)
		if not moves:
			return board.get_score(state, state.player), None
			
		best_score = float('inf')
		best_move = None
		
		for move in moves:
			next_state = board.next_state(state, move)
			score, _ = max_value(next_state, depth - 1, alpha, beta)
			
			if score < best_score:
				best_score = score
				best_move = move
				
			beta = min(beta, best_score)
			if beta <= alpha:
				break
				
		return best_score, best_move

	# Start the search
	_, move = max_value(state, depth_minimax, float('-inf'), float('inf'))
	return move