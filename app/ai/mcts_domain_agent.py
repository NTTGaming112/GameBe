import random
import numpy as np
from heuristics import evaluate, heuristic
from constants import DEFAULT_MCTS_DOMAIN_ITERATIONS
from mcts_agent import MCTSNode
import math

class MCTSDomainNode(MCTSNode):
    def __init__(self, state, parent=None, move=None):
        super().__init__(state, parent, move)

    def expand(self):
        if not self.untried_moves:
            return None
        
        scored_moves = [(move, heuristic(move, self.state, self.state.current_player)) 
                    for move in self.untried_moves]
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        threshold = scored_moves[0][1] * 0.7
        good_moves = [move for move, score in scored_moves if score >= threshold]
        
        if good_moves:
            move = good_moves[0]  
        else:
            move = random.choice(self.untried_moves) 
        
        self.untried_moves.remove(move)
        next_state = self.state.copy()
        next_state.make_move(move)
        child = MCTSDomainNode(state=next_state, parent=self, move=move)
        self.children.append(child)
        return child

    def tournament_rollout(self, root_player, tournament_params):
        moves = self.state.get_legal_moves()
        if not moves:
            return evaluate(self.state, root_player)
        
        move_scores = []
        best_score_so_far = float('-inf')
        
        for move in moves:
            total_score = 0
            simulations_run = 0
            
            for sim_idx in range(tournament_params[0]['num_sim']):
                sim_state = self.state.copy()
                sim_state.make_move(move)
                
                while not sim_state.is_game_over():
                    legal_moves = sim_state.get_legal_moves()
                    if not legal_moves:
                        sim_state.current_player = -sim_state.current_player
                        continue
                    
                    scored_moves = [(m, heuristic(m, sim_state, sim_state.current_player)) 
                                for m in legal_moves]
                    scored_moves.sort(key=lambda x: x[1], reverse=True)
                    best_move = scored_moves[0][0]
                    sim_state.make_move(best_move)
                
                total_score += evaluate(sim_state, root_player)
                simulations_run += 1
                
                if sim_idx >= 5:
                    current_avg = total_score / simulations_run
                    
                    if current_avg < best_score_so_far - 0.3 and sim_idx >= 10:
                        break
                    
                    if current_avg > best_score_so_far + 0.2 and sim_idx >= 15:
                        break
            
            avg_score = total_score / simulations_run if simulations_run > 0 else 0
            move_scores.append((move, avg_score, simulations_run))
            best_score_so_far = max(best_score_so_far, avg_score)
        
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        if len(move_scores) > 1 and move_scores[0][1] - move_scores[1][1] > 0.4:
            return move_scores[0][1]  
        
        top_moves_r1 = [move for move, _, _ in move_scores[:tournament_params[0]['top_k']]]
        
        move_scores_r2 = []
        for i, move in enumerate(top_moves_r1):
            adaptive_sims = tournament_params[1]['num_sim'] + (len(top_moves_r1) - i - 1) * 10
            
            total_score = 0
            simulations_run = 0
            
            for sim_idx in range(adaptive_sims):
                sim_state = self.state.copy()
                sim_state.make_move(move)
                
                while not sim_state.is_game_over():
                    legal_moves = sim_state.get_legal_moves()
                    if not legal_moves:
                        sim_state.current_player = -sim_state.current_player
                        continue
                    
                    scored_moves = [(m, heuristic(m, sim_state, sim_state.current_player)) 
                                for m in legal_moves]
                    scored_moves.sort(key=lambda x: x[1], reverse=True)
                    best_move = scored_moves[0][0]
                    sim_state.make_move(best_move)
                
                total_score += evaluate(sim_state, root_player)
                simulations_run += 1
                
                if sim_idx >= 8:
                    current_avg = total_score / simulations_run
                    if sim_idx >= 20 and abs(current_avg - (total_score - evaluate(sim_state, root_player)) / (simulations_run - 1)) < 0.05:
                        break
            
            avg_score = total_score / simulations_run if simulations_run > 0 else 0
            move_scores_r2.append((move, avg_score, simulations_run))
        
        move_scores_r2.sort(key=lambda x: x[1], reverse=True)
        
        if len(move_scores_r2) > 1 and move_scores_r2[0][1] - move_scores_r2[1][1] > 0.3:
            return move_scores_r2[0][1]
        
        top_moves_r2 = [move for move, _, _ in move_scores_r2[:tournament_params[1]['top_k']]]
        
        move_scores_r3 = []
        for move in top_moves_r2:
            total_score = 0
            simulations_run = 0
            variance_sum = 0
            scores = []
            
            for sim_idx in range(tournament_params[2]['num_sim']):
                sim_state = self.state.copy()
                sim_state.make_move(move)
                
                while not sim_state.is_game_over():
                    legal_moves = sim_state.get_legal_moves()
                    if not legal_moves:
                        sim_state.current_player = -sim_state.current_player
                        continue
                    
                    scored_moves = [(m, heuristic(m, sim_state, sim_state.current_player)) 
                                for m in legal_moves]
                    scored_moves.sort(key=lambda x: x[1], reverse=True)
                    best_move = scored_moves[0][0]
                    sim_state.make_move(best_move)
                
                score = evaluate(sim_state, root_player)
                total_score += score
                scores.append(score)
                simulations_run += 1
                
                if sim_idx >= 15:
                    current_avg = total_score / simulations_run
                    if len(scores) > 1:
                        variance = np.var(scores)
                        std_error = math.sqrt(variance / len(scores))
                        
                        if std_error < 0.1: 
                            break
            
            avg_score = total_score / simulations_run if simulations_run > 0 else 0
            move_scores_r3.append((move, avg_score, simulations_run))
        
        if move_scores_r3:
            return max(move_scores_r3, key=lambda x: x[1])[1]
        elif move_scores_r2:
            return max(move_scores_r2, key=lambda x: x[1])[1]
        else:
            return max(move_scores, key=lambda x: x[1])[1]

class MCTSDomainAgent:
    def __init__(self, iterations=DEFAULT_MCTS_DOMAIN_ITERATIONS, tournament_params=None):
        self.iterations_per_round = iterations * 6 // 10
        tournament_total = iterations * 4 // 10
        
        self.tournament_params = tournament_params if tournament_params else [
            {'num_sim': tournament_total // 4, 'top_k': 5},      
            {'num_sim': tournament_total // 3, 'top_k': 3},      
            {'num_sim': tournament_total * 5 // 12, 'top_k': 1}  
        ]

    def get_move(self, state):
        root = MCTSDomainNode(state)
        root_player = state.current_player
        
        best_move_history = []
        convergence_threshold = 5 
        
        for iteration in range(self.iterations_per_round):
            node = root
            
            if node.untried_moves:
                node = node.expand()
            
            result = node.tournament_rollout(root_player, self.tournament_params)
            node.backpropagate(result, root_player)
            
            if iteration >= 10:  
                current_best = max(root.children, 
                                key=lambda c: c.wins / c.visits if c.visits > 0 else 0)
                best_move_history.append(current_best.move)
                
                if len(best_move_history) > convergence_threshold:
                    best_move_history.pop(0)
                
                if len(best_move_history) == convergence_threshold:
                    if all(move == best_move_history[0] for move in best_move_history):
                        confidence = current_best.wins / current_best.visits
                        if confidence > 0.7: 
                            break
                
                if len(root.children) > 1:
                    sorted_children = sorted(root.children, 
                                        key=lambda c: c.wins / c.visits if c.visits > 0 else 0, 
                                        reverse=True)
                    if len(sorted_children) >= 2:
                        best_rate = sorted_children[0].wins / sorted_children[0].visits if sorted_children[0].visits > 0 else 0
                        second_rate = sorted_children[1].wins / sorted_children[1].visits if sorted_children[1].visits > 0 else 0
                        
                        if (best_rate - second_rate > 0.3 and 
                            sorted_children[0].visits > 20 and 
                            iteration > 15):
                            break

        if not root.children:
            return None
        return max(root.children, key=lambda c: c.wins / c.visits if c.visits > 0 else 0).move