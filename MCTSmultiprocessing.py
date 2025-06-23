import copy
import math

import numpy as np
import torch
from collections import OrderedDict



class MCTSCache:
    def __init__(self, capacity=10000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):

        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


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
    
    def __init__(self, games, num_simulations ,config,request_model_queue, response_model_queue, worker_id, mode="selfplay",model_id = "model1"):
        self.games = games
        self.num_parallel_mcts = len(games)
        self.simulations = num_simulations
        self.request_model_queue = request_model_queue
        self.response_model_queue = response_model_queue
        self.worker_id = worker_id
        self.C = config.C
        self.dirichlet_alpha = config.dirichlet_alpha
        self.exploration_fraction = config.exploration_fraction
        self.mode = mode
        self.model_id = model_id
        self.cache = MCTSCache()
        self.cache_hits = 0
        self.cache_misses = 0
        
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
            self.expand_nodes_cache(nodos_a_expandir)
        

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
        
        #print(f"[MCTS Cache] Hits: {self.cache_hits} | Misses: {self.cache_misses}")

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
                if mcts_i.selected_node is mcts_i.root and self.mode == "selfplay":
                    distribucion = self.aplicar_ruido(distribucion)
                mcts_i.selected_node.expand(distribucion)
                mcts_i.value = value

    def expand_nodes_cache(self, expandable_pararlel_mcts):

        if not expandable_pararlel_mcts:
            return

        mcts_pendientes_prediccion = []

        # Mirar qué nodos ya están en cache
        for mcts_i in expandable_pararlel_mcts:
            node = mcts_i.selected_node
            encoded = node.game.encode_board()
            key = encoded.tobytes()

            cached = self.cache.get(key)
            if cached:

                distribucion, value = cached
                self.cache_hits +=1
                if node is mcts_i.root and self.mode == "selfplay":
                    distribucion = self.aplicar_ruido(distribucion)
                node.expand(distribucion)
                mcts_i.value = value
            else:
                self.cache_misses+=1
                mcts_pendientes_prediccion.append(mcts_i)

        if not mcts_pendientes_prediccion:
            return

        nodos_a_expandir = [mcts.selected_node for mcts in mcts_pendientes_prediccion]

        distribuciones, valores = self.obtener_distribuciones_batch(nodos_a_expandir)

        for mcts_i, distribucion, value in zip(mcts_pendientes_prediccion, distribuciones, valores):
            if mcts_i.selected_node is mcts_i.root:
                distribucion = self.aplicar_ruido(distribucion)
            mcts_i.selected_node.expand(distribucion)
            mcts_i.value = value

            # Guardamos en caché
            encoded = mcts_i.selected_node.game.encode_board()
            key = encoded.tobytes()
            self.cache.put(key, (distribucion, value))


    def backpropagate(self, search_path, value):
        oponente = 1
        for node in reversed(search_path):
            node.value_sum += value*oponente
            node.visit_count += 1
            oponente = oponente*-1


    def obtener_distribuciones_batch(self, nodos):
        encoded_boards = [torch.tensor(node.game.encode_board(), dtype=torch.float32) for node in nodos]

        if self.mode == "selfplay":
            self.request_model_queue.put((encoded_boards, self.worker_id))
        else:
            self.request_model_queue.put((encoded_boards, self.worker_id,self.model_id))

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

    def aplicar_ruido(self, distribucion):

        alpha = self.dirichlet_alpha
        epsilon = self.exploration_fraction

        # Aplica Dirichlet solo sobre las acciones legales (>0 en la distribución)
        legal_indices = np.where(distribucion > 0)[0]
        dir_noise = np.zeros_like(distribucion)

        if len(legal_indices) > 0:
            dirichlet = np.random.dirichlet([alpha] * len(legal_indices))
            dir_noise[legal_indices] = dirichlet

            distribucion = (1 - epsilon) * distribucion + epsilon * dir_noise
            distribucion /= np.sum(distribucion)  # Re-normalizar

        return distribucion





