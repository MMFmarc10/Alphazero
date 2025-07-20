import math
from collections import OrderedDict

import numpy as np
import torch


# Clase de caché LRU (Least Recently Used) para almacenar predicciones del modelo
# para enviar menos peticiones a la red neuronal
class MCTSCache:
    def __init__(self, capacity=50000):
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
            print("epaaaa")
            self.cache.popitem(last=False)


# Almacena la información del MCTS de una partida.
class MCTSInfo:
    def __init__(self, game):
        self.root = Node(game, visit_count=1)
        self.search_path = []
        self.selected_node = None
        self.value = None
        self.terminada = None


# Implementación de Monte Carlo Tree Search (MCTS) para AlphaZero.
# Realiza simulaciones sobre múltiples juegos en paralelo
class MCTS2:
    
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
        Node.C = config.C
        Node.ACTION_SIZE = games[0].get_action_size()

    # Ejecuta todas las simulaciones MCTS para todos los juegos en paralelo.
    # Realiza las fases de selección, expansión, backpropagation y devuelve distribuciones de política de cada partida.
    def iniciar(self):

        parallel_mcts = [MCTSInfo(game) for game in self.games]

        self.expand_root_nodes(parallel_mcts)
        
        for _ in range(self.simulations):

            # Selection
            for mcts_i in parallel_mcts:

                mcts_i.search_path = [mcts_i.root]
                mcts_i.selected_node = self.select_node(mcts_i.root, mcts_i.search_path)

                mcts_i.terminada = mcts_i.selected_node.game.is_game_over()
                if mcts_i.terminada:

                    winner = mcts_i.selected_node.game.get_game_result()
                    player = mcts_i.selected_node.game.player

                    if winner == 0:
                        mcts_i.value = 0
                    elif player == winner:
                        mcts_i.value = 1
                    else:
                        mcts_i.value = -1

            # Expansion
            nodos_a_expandir = [mcts_i for mcts_i in parallel_mcts if not mcts_i.terminada]
            self.expand_nodes_cache(nodos_a_expandir)
        

            # Backpropagation
            for mcts_instance in parallel_mcts:

                self.backpropagate(mcts_instance.search_path, mcts_instance.value)

                for node in reversed(mcts_instance.search_path[1:]):  # No hace falta deshacer el root
                    node.game.undo_move()

        # Resultados
        resultados = []

        for mcts_instance in parallel_mcts:
            root = mcts_instance.root
            moves = []
            distribution = np.zeros(self.games[0].ACTION_SIZE)

            for child in root.children:

                if child is not None:
                    moves.append(child.move)
                    distribution[child.move] = child.visit_count

            if np.sum(distribution) != 0:
                distribution /= np.sum(distribution)
            else:
                print("error")
                root.game.print_board()

            resultados.append((moves, distribution, 0))

        return resultados

    # Navega por el árbol MCTS de la partida desde la raíz hasta un nodo hoja sin expandir.
    def select_node(self,node,search_path):

        while node.is_expanded():
            node = node.select()
            node.game.make_move(node.move)
            search_path.append(node)
        
        return node

    # Expande los nodos seleccionados usando la red neuronal.
    def expand_root_nodes(self, expandable_pararlel_mcts):

        nodos_a_expandir = [mcts.root for mcts in expandable_pararlel_mcts]
        
        distribuciones, valores = self.obtener_distribuciones_batch(nodos_a_expandir)

        for mcts_i, distribucion, value in zip(
        [m for m in expandable_pararlel_mcts if not m.terminada], distribuciones, valores
        ):

            distribucion = self.aplicar_ruido_dirichlet(distribucion)
            mcts_i.root.expand(distribucion)
            mcts_i.value = value

    # Igual que expand_nodes
    # Pero usando la caché para evitar recomputar distribuciones para tableros vistos previamente.
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

                board_obj = node.game
                legal_mask = board_obj.legal_moves_mask()
                legal_policy = distribucion * legal_mask

                if np.sum(legal_policy) == 0:
                    print("ueeepapaa")
                    legal_policy = legal_mask

                legal_policy /= np.sum(legal_policy)

                node.expand(legal_policy)
                mcts_i.value = value
            else:

                mcts_pendientes_prediccion.append(mcts_i)

        if not mcts_pendientes_prediccion:
            return

        nodos_a_expandir = [mcts.selected_node for mcts in mcts_pendientes_prediccion]

        distribuciones, valores = self.obtener_distribuciones_batch(nodos_a_expandir)

        for mcts_i, distribucion, value in zip(mcts_pendientes_prediccion, distribuciones, valores):

            mcts_i.selected_node.expand(distribucion)
            mcts_i.value = value

        # Propaga el valor estimado por el modelo hacia atrás desde el nodo hoja hasta la raíz

    def backpropagate(self, search_path, value):
        """Propaga el valor desde la hoja hasta la raíz."""
        invert = 1

        for i in reversed(range(1, len(search_path))):
            node = search_path[i]
            parent = search_path[i - 1]
            move = node.move

            # Actualiza las estadísticas del padre en el movimiento que llevó al hijo
            parent.visit_counts[move] += 1
            parent.total_values[move] += value * invert

            node.visit_count += 1

            invert *= -1  # Cambia la perspectiva del jugador

        # ✔️ No hace falta actualizar el root: ya se actualizó en la última iteración

    # Solicita al modelo una predicción de política y valor para varios tableros en batch.
    # Aplica una máscara de movimientos legales a la política resultante.
    def obtener_distribuciones_batch(self, nodos):

        encoded_boards = np.stack([node.game.encode_board() for node in nodos])  # Más rápido
        encoded_boards_tensor = torch.from_numpy(encoded_boards).float()  # Un solo tensor


        if self.mode == "selfplay":
            self.request_model_queue.put((encoded_boards_tensor, self.worker_id))
        else:
            self.request_model_queue.put((encoded_boards_tensor, self.worker_id,self.model_id))

        policy_tensors, value_tensors = self.response_model_queue.get()

        policy_tensors_np = policy_tensors.numpy()

        valores =  value_tensors.numpy()


        distribuciones_legales = []
        for i, policy in enumerate(policy_tensors_np):

            board_obj = nodos[i].game
            legal_mask = board_obj.legal_moves_mask()
            legal_policy = policy * legal_mask

            if np.sum(legal_policy) == 0:
                print("ueeepapaa")
                legal_policy = legal_mask

            legal_policy /= np.sum(legal_policy)
            distribuciones_legales.append(legal_policy)
            encoded = encoded_boards[i]
            key = encoded.tobytes()
            self.cache.put(key, (policy, valores[i]))


        return distribuciones_legales, valores

    # Añade ruido de Dirichlet a la política en la raíz del árbol para fomentar la exploración
    def aplicar_ruido_dirichlet(self, distribucion):

        alpha = self.dirichlet_alpha
        epsilon = self.exploration_fraction

        legal_indices = np.where(distribucion > 0)[0]
        dir_noise = np.zeros_like(distribucion)

        if len(legal_indices) > 0:
            dirichlet = np.random.dirichlet([alpha] * len(legal_indices))
            dir_noise[legal_indices] = dirichlet

            distribucion = (1 - epsilon) * distribucion + epsilon * dir_noise
            distribucion /= np.sum(distribucion)  # Re-normalizar

        return distribucion


# Representa un nodo del árbol MCTS
class Node:

    C = 1.5
    ACTION_SIZE = None

    def __init__(self, game, move=None, visit_count=0):
        self.visit_count = visit_count  # Total de veces que este nodo fue visitado
        self.policy = None  # Política (al expandir)
        self.visit_counts = None  # Lista de visitas por acción
        self.total_values = None  # Suma de valores por acción
        self.children = None  # Lista de hijos
        self.game = game  # Estado tras aplicar move
        self.move = move  # El movimiento que llevó a este estado

    # Devuelve el valor del nodo
    @staticmethod
    def value(value_sum, visit_count):
        if visit_count == 0:
            return 0
        return value_sum / visit_count

    # Indica si el nodo ya ha sido expandido
    def is_expanded(self):
        return self.children is not None

    def expand(self, policy):

        self.policy = policy  # Ya es lista, no hay que convertir

        size = Node.ACTION_SIZE

        # Inicializa las listas de estadísticas y de hijos
        self.visit_counts = [0] * size
        self.total_values = [0.0] * size
        self.children = [None] * size


    def select(self):
        mejor_ucb = -float('inf')
        mejor_move = None

        for move in range(Node.ACTION_SIZE):
            if self.policy[move] == 0:
                continue  # No calcules UCB en movimientos con probabilidad 0

            ucb = self.get_ucb_score(move)

            if ucb > mejor_ucb:
                mejor_ucb = ucb
                mejor_move = move

        # Lazy expand: crea el hijo solo si aún no existe
        if self.children[mejor_move] is None:

            self.children[mejor_move] = Node(self.game, move=mejor_move)

        return self.children[mejor_move]


    def get_ucb_score(self, move):

        visits = self.visit_counts[move]
        total = self.total_values[move]
        prior = self.policy[move]

        # Q: valor promedio acumulado (negado porque AlphaZero usa perspectiva del jugador actual)
        q_value = -Node.value(total, visits)

        # U: incentivo por explorar
        u_value = Node.C * prior * (math.sqrt(self.visit_count) / (1 + visits))

        return q_value + u_value



import numpy as np

class Node:
    C = 1.5
    ACTION_SIZE = None  # Debe configurarse desde fuera

    def __init__(self, game, move=None):
        self.visit_count = 0
        self.game = game
        self.move = move

        # Se inicializan cuando se expande
        self.policy = None
        self.legal_moves = None
        self.visit_counts = None
        self.total_values = None
        self.children = None
        self.move_to_index = None

    def expand(self, policy):
        # Guardamos la policy completa (opcional)
        self.policy = np.array(policy, dtype=np.float32)

        # Calculamos los movimientos legales (donde la probabilidad > 0)
        self.legal_moves = np.nonzero(self.policy)[0]
        size = len(self.legal_moves)

        # Creamos arrays solo del tamaño de legal_moves
        self.visit_counts = np.zeros(size, dtype=np.int32)
        self.total_values = np.zeros(size, dtype=np.float32)
        self.children = np.full(size, None, dtype=object)

        # Mapeo rápido move (global) -> índice en estos arrays
        self.move_to_index = np.full(Node.ACTION_SIZE, -1, dtype=np.int16)
        for idx, move in enumerate(self.legal_moves):
            self.move_to_index[move] = idx

    def is_expanded(self):
        return self.legal_moves is not None

    def select(self):
        # Vectorizamos el cálculo del UCB
        visits = self.visit_counts
        totals = self.total_values
        priors = self.policy[self.legal_moves]

        q_values = -np.divide(totals, visits, out=np.zeros_like(totals), where=visits != 0)
        u_values = Node.C * priors * np.sqrt(self.visit_count) / (1 + visits)

        ucb_scores = q_values + u_values

        # Elegimos el movimiento con mayor UCB
        best_index = np.argmax(ucb_scores)
        best_move = self.legal_moves[best_index]

        # Lazy expand
        if self.children[best_index] is None:
            self.children[best_index] = Node(self.game, move=best_move)

        return self.children[best_index]

    @staticmethod
    def value(value_sum, visit_count):
        if visit_count == 0:
            return 0
        return value_sum / visit_count
