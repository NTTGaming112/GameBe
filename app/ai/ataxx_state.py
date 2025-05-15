#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
from copy import deepcopy
from collections import deque as dl
## 1 é o branco
## -1 é o preto
class Ataxx:
	"""
	Lớp đại diện cho trò chơi Ataxx (còn gọi là Infection hoặc Reversi cải tiến).
	
	Quy tắc:
	- Người chơi có thể Clone (sao chép) hoặc Jump (nhảy) quân của mình
	- Clone: tạo một quân mới ở ô liền kề, quân cũ vẫn giữ nguyên
	- Jump: di chuyển quân hiện có đến ô cách 2 bước, quân cũ biến mất
	- Sau mỗi nước đi, các quân đối phương liền kề với quân mới sẽ bị chiếm
	- Người chơi không thể di chuyển sẽ bị mất lượt
	- Trò chơi kết thúc khi bàn cờ đầy hoặc một người chơi không còn quân nào
	- Người có nhiều quân hơn sẽ thắng
	
	Cải tiến mới:
	- Trò chơi cũng kết thúc nếu một vị trí bàn cờ xuất hiện lần thứ 3 với cùng người chơi
	"""
	def __init__(self):
		self.balls = {}
		self.moves = {}
		self.balls[1] = 0
		self.balls[-1] = 0
		self.moves[1] = 0
		self.moves[-1] = 0
		self.n_fields = 7
		self.board = [[0 for x in range(self.n_fields)] for y in range(self.n_fields)]
		
		self.turn_player = 1

		self.add_ball(0,0)
		self.add_ball(self.n_fields-1,self.n_fields-1)

		self.turn_player = -1

		self.add_ball(0,self.n_fields-1)
		self.add_ball(self.n_fields-1,0)

		self.turn_player = 1

		self.last_pos = -1
		self.last_player = 0
		self.cont_last_pos = 0
		self.stop_game = False
		
		# Dictionary để lưu trữ lịch sử trạng thái bàn cờ
		# Key: (tuple của board làm phẳng, người chơi hiện tại)
		# Value: số lần trạng thái này xuất hiện
		self.position_history = {}


	def get_all_possible_moves(self):
		"""Lấy tất cả các nước đi hợp lệ cho người chơi hiện tại.
		
		Returns:
			list: Danh sách các nước đi có thể, định dạng:
				- ('c', (x, y)) cho Clone move
				- ('j', (x_dest, y_dest), (x_src, y_src)) cho Jump move
		"""
		pos_copy = [(-1,1),(0,1),(1,1),(-1,0),(1,0),(-1,-1),(0,-1),(1,-1)]
		pos_jump = [(-2,2),(-2,1),(-2,0),(-2,-1),(-2,-2),(-1,2),(-1,-2),
		(0,2),(0,-2),(1,2),(1,-2),(2,2),(2,1),(2,0),(2,-1),(2,-2)]

		possible_moves = []
		for x in range(self.n_fields):
			for y in range(self.n_fields):
				if(self.board[x][y] != self.turn_player):
					continue
				b = (x,y)
				
				# Thêm tất cả nước Clone hợp lệ
				possible_moves.extend([('c', p) for p in self.get_empty_pos(b, pos_copy)])
				
				# Thêm tất cả nước Jump hợp lệ
				possible_moves.extend([('j', p, b) for p in self.get_empty_pos(b, pos_jump)])

		return possible_moves

	def update_board(self,state):
		self.board = state.board
		self.turn_player = state.player
		
		self.balls[-1] = state.balls[-1]
		self.balls[1] = state.balls[1]

	def toggle_player(self):
		self.turn_player = -1 * self.turn_player

	def current_player(self):
		return self.turn_player

	def move_with_position(self,position):
		if(position[0] == 'c'):
			self.copy_stone_position(position[1])
		else:
			self.jump_stone_position(position[1],position[2])
			
		# Cập nhật lịch sử trạng thái sau mỗi nước đi
		self.update_position_history()

	def copy_stone_position(self,p):
		x,y = p[0],p[1]
		self.add_ball(x,y)
		self.take_stones(x,y)
		self.increase_move()

	def jump_stone_position(self,p,b):
		x,y = p[0],p[1]
		self.add_ball(x,y)
		self.remove_ball(b[0],b[1])
		self.take_stones(x,y)
		self.increase_move()

	####### monte carlo

	def move(self):
		"""Thực hiện một nước đi ngẫu nhiên cho người chơi hiện tại.
		Nếu không có nước đi hợp lệ, trả về False.
		
		Returns:
			bool: True nếu thực hiện thành công, False nếu không có nước đi hợp lệ
		"""
		moves = self.get_all_possible_moves()
		if not moves:
			return False
			
		idx = np.random.randint(0, len(moves))
		self.move_with_position(moves[idx])
		return True


	def increase_move(self):
		self.moves[self.turn_player] += 1

	def get_amount_moves(self):
		return self.moves[1] + self.moves[-1]

	def get_empty_pos(self,b,pos):
		"""Lấy danh sách các vị trí trống từ các điểm tương đối so với vị trí b.
		
		Args:
			b (tuple): Vị trí gốc (x, y)
			pos (list): Danh sách các vị trí tương đối cần kiểm tra
			
		Returns:
			list: Danh sách các vị trí trống hợp lệ
		"""
		return [(b[0]+x, b[1]+y) for x, y in pos 
				if 0 <= b[0]+x < self.n_fields 
				and 0 <= b[1]+y < self.n_fields 
				and self.is_empty(b[0]+x, b[1]+y)]

	def get_full_pos(self,b,pos):
		"""Lấy danh sách các vị trí đã có quân từ các điểm tương đối so với vị trí b.
		
		Args:
			b (tuple): Vị trí gốc (x, y)
			pos (list): Danh sách các vị trí tương đối cần kiểm tra
			
		Returns:
			list: Danh sách các vị trí đã có quân
		"""
		return [(b[0]+x, b[1]+y) for x, y in pos 
				if 0 <= b[0]+x < self.n_fields 
				and 0 <= b[1]+y < self.n_fields 
				and not self.is_empty(b[0]+x, b[1]+y)]


	def get_copy_position(self,b):
		pos = [(-1,1),(0,1),(1,1),(-1,0),(1,0),(-1,-1),(0,-1),(1,-1)]

		vs = self.get_empty_pos(b,pos)

		if(len(vs) == 0):
			return b

		x_m,y_m = vs[np.random.randint(0,len(vs))]
	
		return x_m,y_m

	def get_jump_position(self,b):
		pos = [(-2,2),(-2,1),(-2,0),(-2,-1),(-2,-2),(-1,2),(-1,-2),
		(0,2),(0,-2),(1,2),(1,-2),(2,2),(2,1),(2,0),(2,-1),(2,-2)]

		vs = self.get_empty_pos(b,pos)

		if(len(vs) == 0):
			return b

		x_m,y_m = vs[np.random.randint(0,len(vs))]
		return x_m,y_m


	def choose_ball(self):
		#balls = self.balls[self.turn_player]
		idx = np.random.randint(1,self.balls[self.turn_player]+1)
		cont = 1
		for x in range(self.n_fields):
			for y in range(self.n_fields):
				if(self.board[x][y] == self.turn_player):
					if(cont == idx):
						return [x,y]
					cont += 1		

	def copy_stone(self):
		b = self.choose_ball()
		[x,y] = b
		#print "copy b",b
		while([x,y] == b):
			x,y = self.get_copy_position(b)
		
		self.add_ball(x,y)
		self.take_stones(x,y)
		self.increase_move()

	def jump_stone(self):
		b = self.choose_ball()
		#print "move b",b
		x,y = self.get_jump_position(b)
		if([x,y] != b):
			self.add_ball(x,y)
			self.remove_ball(b[0],b[1])
			self.take_stones(x,y)
			self.increase_move()

	def take_stones(self,x,y):
		b = [x,y]
		pos = [(-1,1),(0,1),(1,1),(-1,0),(1,0),(-1,-1),(0,-1),(1,-1)]
		pos = self.get_full_pos(b,pos)
		for x,y in pos:
			# bola não é do jogador atual
			if(self.turn_player == -1 and self.board[x][y] == 1):
				self.change_ball_player(x,y)
			elif(self.turn_player == 1 and self.board[x][y] == -1):
				self.change_ball_player(x,y)

	def change_ball_player(self,x,y):
		self.board[x][y] = -1*self.board[x][y]
		self.balls[self.turn_player] += 1
		self.balls[-1*self.turn_player] -= 1
		assert self.balls[-1] >= 0
		assert self.balls[1] >= 0


	def is_empty(self,x,y):
		return self.board[x][y] == 0

	def add_ball(self,x,y):
		assert self.board[x][y] == 0
		self.board[x][y] = self.turn_player
		self.balls[self.turn_player] += 1

	def remove_ball(self,x,y):
		assert self.board[x][y] != 0
		self.board[x][y] = 0
		self.balls[self.turn_player] -= 1

	def full_squares(self):
		if(self.balls[1] + self.balls[-1] >= (self.n_fields)*(self.n_fields)):
			return True
		return False

	def is_game_over(self):
		if(self.stop_game):
			return True
		if(self.balls[1] == 0 or self.balls[-1] == 0):
			return True
		if(self.full_squares()):
			return True
		# Kiểm tra nếu vị trí hiện tại lặp lại lần thứ 3 với cùng người chơi
		if(self.position_repeated_three_times()):
			return True
		return False

	def print_winner(self):
		if self.balls[1] > self.balls[-1]:
			print("Winner: Branco. Gamer 1")
		elif self.balls[-1] > self.balls[1]:
			print("Winner: Preto. Gamer 2")
		else:
			print("Draw")

	def get_winner(self):
		#self.print_winner()
		if(self.balls[-1] > self.balls[1]):
			return -1
		elif(self.balls[1] > self.balls[-1]):
			return 1
		else:
			return 100

	def get_winner_without_gameover(self):
	#self.print_winner()
		if(self.balls[-1] > self.balls[1]):
			return -1
		elif(self.balls[1] > self.balls[-1]):
			return 1
		else:
			return 0

	def get_score(self,player):
		assert self.balls[1] >=0
		assert self.balls[-1] >=0
		return (self.balls[player] * 1.) / (self.balls[1] + self.balls[-1])

	def get_score_pieces(self,player):
		return (self.balls[player] * 1.) / (self.n_fields**2)
		
	def board_to_tuple(self):
		"""Chuyển đổi bảng 2D thành tuple 1D để lưu trữ trong dictionary"""
		flat_board = []
		for row in self.board:
			flat_board.extend(row)
		return tuple(flat_board)
		
	def update_position_history(self):
		"""Cập nhật lịch sử vị trí bàn cờ"""
		position_key = (self.board_to_tuple(), self.turn_player)
		
		if position_key in self.position_history:
			self.position_history[position_key] += 1
		else:
			self.position_history[position_key] = 1
			
	def position_repeated_three_times(self):
		"""Kiểm tra xem vị trí hiện tại có xuất hiện 3 lần không"""
		position_key = (self.board_to_tuple(), self.turn_player)
		return self.position_history.get(position_key, 0) >= 3

	def print_board(self):
		print("Player:", self.turn_player)
		s = [[str(e) for e in row] for row in self.board]
		lens = [max(map(len, col)) for col in zip(*s)]
		fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
		table = [fmt.format(*row) for row in s]
		print('\n'.join(table))
		print("Peças Branco: " + str(self.balls[1]) + " Peças Preto: " + str(self.balls[-1]))

	def show_board(self):
		msg = "Player: "+str(self.turn_player)+"\n"
		s = [[str(e) for e in row] for row in self.board]
		lens = [max(map(len, col)) for col in zip(*s)]
		fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
		table = [fmt.format(*row) for row in s]
		msg += '\n'.join(table)
		msg += "\n White pieces: "+str(self.balls[1])
		msg += " | Black pieces: "+str(self.balls[-1])
		return msg

	def play(self):
		"""Tự động chơi cả trò chơi đến khi kết thúc.
		Sử dụng các nước đi ngẫu nhiên cho cả hai người chơi.
		"""
		no_moves_count = 0  # Đếm số lần liên tiếp không có nước đi hợp lệ
		
		while not self.is_game_over():
			# Thực hiện nước đi
			if not self.move():
				# Không có nước đi hợp lệ, tăng biến đếm
				no_moves_count += 1
				if no_moves_count >= 2:
					# Cả hai người chơi đều không có nước đi, kết thúc trò chơi
					self.stop_game = True
					break
			else:
				# Có nước đi hợp lệ, reset biến đếm
				no_moves_count = 0
				
			# Chuyển lượt
			self.toggle_player()