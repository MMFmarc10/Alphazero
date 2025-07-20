import math
from collections import OrderedDict

import numpy as np
import torch


# Clase de caché LRU (Least Recently Used) para almacenar predicciones del modelo
# para enviar menos peticiones a la red neuronal
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

    def __init__(self, games, num_simulations, config, request_model_queue, response_model_queue, worker_id,
                 mode="selfplay", model_id="model1"):
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

        Node.C = self.C


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
        # Resultados
        resultados = []

        for mcts_instance in parallel_mcts:
            root = mcts_instance.root

            distribution = np.zeros(self.games[0].ACTION_SIZE, dtype=np.float32)

            if root.legal_moves is not None:
                for move, visits in zip(root.legal_moves, root.children_visit_counts):
                    distribution[move] = visits

            total_visits = np.sum(distribution)

            if total_visits != 0:
                distribution /= total_visits
            else:
                print("Error: distribución vacía")
                root.game.print_board()

            # Calcula el valor medio de los hijos (opcionalmente puedes aplicar - promedio porque el root está desde su perspectiva)
            total_child_visits = np.sum(root.children_visit_counts)
            if total_child_visits != 0:
                avg_value = - np.sum(root.children_value_sums) / total_child_visits
            else:
                avg_value = 0.0

            resultados.append((root.legal_moves, distribution, avg_value))

        return resultados

    # Navega por el árbol MCTS de la partida desde la raíz hasta un nodo hoja sin expandir.
    def select_node(self, node, search_path):

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
            if self.mode == "selfplay":
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

                node.expand(distribucion)
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

            # Guardamos en caché
            encoded = mcts_i.selected_node.game.encode_board()
            key = encoded.tobytes()
            self.cache.put(key, (distribucion, value))

    def backpropagate(self, search_path, value):
        oponente = 1

        for i in reversed(range(1, len(search_path))):
            parent = search_path[i - 1]
            child = search_path[i]

            # Obtenemos el índice del movimiento que llevó al hijo
            idx = parent.selected_children

            # Actualizamos las estadísticas del padre hacia el hijo
            parent.children_visit_counts[idx] += 1
            parent.children_value_sums[idx] += value * oponente

            # Sumar la visita al hijo
            child.visit_count += 1

            oponente *= -1

        # No olvides sumar la visita al root
        search_path[0].visit_count += 1

        # Solicita al modelo una predicción de política y valor para varios tableros en batch.
        # Aplica una máscara de movimientos legales a la política resultante.

    def obtener_distribuciones_batch(self, nodos):

        #encoded_boards = np.stack([node.game.encode_board() for node in nodos])  # Más rápido
        #encoded_boards_tensor = torch.from_numpy(encoded_boards).float()  # Un solo tensor
        print("************")
        print(len(nodos))
        print(self.worker_id)
        encoded_boards = np.stack([node.game.encode_board() for node in nodos], dtype=np.uint8)
        encoded_boards_tensor = torch.from_numpy(encoded_boards)
        encoded_boards_tensor.share_memory_()
        print(encoded_boards_tensor.shape)
        print("************")

        if self.mode == "selfplay":
            self.request_model_queue.put((encoded_boards_tensor, self.worker_id))
        else:
            self.request_model_queue.put((encoded_boards_tensor, self.worker_id, self.model_id))

        policy_tensors, value_tensors = self.response_model_queue.get()

        policy_tensors_np = policy_tensors.numpy()

        valores = value_tensors.numpy()

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

    def __init__(self, game, move=None, visit_count=0):
        self.game = game
        self.move = move
        self.visit_count = visit_count

        self.legal_moves = None
        self.children_visit_counts = None
        self.children_value_sums = None
        self.children_priors = None
        self.children_list = None

        self.selected_children = None


    # Indica si el nodo ya ha sido expandido
    def is_expanded(self):
        return self.children_list is not None

    # Genera nodos hijos a partir de la política del modelo
    def expand(self, policy_distribution):
        # policy_distribution: lista o array del tamaño del action space
        self.legal_moves = [i for i, prob in enumerate(policy_distribution) if prob != 0]

        self.children_priors = np.array([policy_distribution[move] for move in self.legal_moves], dtype=np.float32)
        self.children_visit_counts = np.zeros(len(self.legal_moves), dtype=np.float32)
        self.children_value_sums = np.zeros(len(self.legal_moves), dtype=np.float32)
        self.children_list = [None] * len(self.legal_moves)  # Ningún hijo creado aún

    def select(self):
        # Paso 1: Calcular los valores Q
      #  q_values = -np.divide(
      #      self.children_value_sums,
      #      np.maximum(self.children_visit_counts, 1),
      #      where=self.children_visit_counts != 0
      #  )

        q_values = np.zeros_like(self.children_value_sums, dtype=np.float32)
        np.divide(
            self.children_value_sums,
            self.children_visit_counts,
            out=q_values,
            where=self.children_visit_counts != 0
        )
        q_values = -q_values

        # Paso 2: Calcular los valores U
        u_values = Node.C * self.children_priors * (math.sqrt(self.visit_count) / (1 + self.children_visit_counts))

        # Paso 3: Sumar UCB
        ucb_scores = q_values + u_values

        # Paso 4: Elegir el mejor movimiento
        best_idx = np.argmax(ucb_scores)
        best_move = self.legal_moves[best_idx]

        # Paso 5: Crear el nodo hijo solo si no existe
        if self.children_list[best_idx] is None:
            # ⚠️ IMPORTANTE: el game debe clonarse, porque cada nodo tiene su propio estado
            child_game = self.game

            child_node = Node(game=child_game, move=best_move)
            self.children_list[best_idx] = child_node

        # Opcionalmente puedes guardar el índice seleccionado
        self.selected_children = best_idx

        return self.children_list[best_idx]