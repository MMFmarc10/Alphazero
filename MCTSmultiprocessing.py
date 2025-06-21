import copy
import math

import numpy as np
import torch


class MCTS:

    class Node:
        def __init__(self,prior,game,move=None, C=1.5):
            self.prior = prior
            self.visit_count= 0
            self.value_sum = 0
            self.children=[]
            self.game= game
            self.move = move
            self.C = C

        def value(self):
            if self.visit_count == 0:
                return 0
            return self.value_sum / self.visit_count

        def is_expanded(self):
            return len(self.children)>0

        def expand(self,distribution):
            
            for move,prob in enumerate(distribution):

                if prob != 0:
                    game_child = copy.deepcopy(self.game)
                    game_child.make_move(move)
                    
                    node = MCTS.Node(prob,game_child,move, C=self.C)
                    self.children.append(node)
                


        def select(self):

            mejor_ucb = -np.inf
            mejor_nodo= None
            for node in self.children:
                ucb_score = self.get_ucb_score(node)
                if ucb_score>mejor_ucb:
                    mejor_ucb = ucb_score
                    mejor_nodo = node
            return mejor_nodo
        

        def get_ucb_score(self, child):
            q_value = -child.value()
            u_value = self.C * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))
            return q_value + u_value
        
    class MCTSInfo:
        def __init__(self, game, C):
            self.root = MCTS.Node(1, game, C=C)
            self.search_path = []
            self.selected_node = None
            self.value = None
            self.terminada = None
    
    def __init__(self, games ,num_simulations, C ,request_model_queue, response_model_queue, worker_id):
        self.games = games
        self.num_parallel_mcts = len(games)
        self.simulations = num_simulations
        self.request_model_queue = request_model_queue
        self.response_model_queue = response_model_queue
        self.worker_id = worker_id
        self.C = C
        
    def iniciar(self):

        parallel_mcts = [self.MCTSInfo(game, self.C) for game in self.games]
        
        for _ in range(self.simulations):

            # Selection
            for mcts_i in parallel_mcts:

                mcts_i.search_path = [mcts_i.root]
                mcts_i.selected_node = self.select_node(mcts_i.root, mcts_i.search_path)

                mcts_i.terminada = mcts_i.selected_node.game.is_game_over()
                if mcts_i.terminada:
                    mcts_i.value = abs(mcts_i.selected_node.game.get_game_result())*-1

            # Expansion
            nodos_a_expandir = [mcts_i for mcts_i in parallel_mcts if not mcts_i.terminada]
            self.expand_nodes(nodos_a_expandir)
        

            # Backpropagation
            for mcts_instance in parallel_mcts:
               
                self.backpropagate(mcts_instance.search_path, mcts_instance.value)

        resultados = []

        for mcts_instance in parallel_mcts:
            root = mcts_instance.root
            moves = []
            distribution = np.zeros(self.games[0].ACTION_SIZE)

            for child in root.children:
                moves.append(child.move)
                distribution[child.move] = child.visit_count

            if np.sum(distribution) != 0:
                distribution /= np.sum(distribution)
            else:
                print("error")
                root.game.print_board()

            resultados.append((moves, distribution, root.value()))
        

        return resultados  # lista de tuplas (moves, distribution, value)

        
    
    def select_node(self,node,search_path):

        while node.is_expanded():
            node = node.select()
            search_path.append(node)
        
        return node


    def expand_nodes(self, expandable_pararlel_mcts):

        if len(expandable_pararlel_mcts)>0:
            nodos_a_expandir = [mcts.selected_node for mcts in expandable_pararlel_mcts]
        
            distribuciones, valores = self.obtener_distribuciones_batch(nodos_a_expandir)

            for mcts_i, distribucion, value in zip(
            [m for m in expandable_pararlel_mcts if not m.terminada], distribuciones, valores
            ):
                mcts_i.selected_node.expand(distribucion)
                mcts_i.value = value


    def backpropagate(self, search_path, value):
        oponente = 1
        for node in reversed(search_path):
            node.value_sum += value*oponente
            node.visit_count += 1
            oponente = oponente*-1


    def obtener_distribuciones_batch(self, nodos):
        encoded_boards = [torch.tensor(node.game.encode_board(), dtype=torch.float32) for node in nodos]

        self.request_model_queue.put((encoded_boards, self.worker_id))

        policy_tensors, value_tensors = self.response_model_queue.get()

        distribuciones = [torch.softmax(policy, dim=0).numpy() for policy in policy_tensors]
        valores = [v.item() for v in value_tensors]

        distribuciones_legales = []
        for i, policy in enumerate(distribuciones):
            board_obj = nodos[i].game
            legal_mask = board_obj.legal_moves_mask()
            legal_policy = policy * legal_mask

            if np.sum(legal_policy) == 0:
                legal_policy = legal_mask

            legal_policy /= np.sum(legal_policy)
            distribuciones_legales.append(legal_policy)

        return distribuciones_legales, valores





