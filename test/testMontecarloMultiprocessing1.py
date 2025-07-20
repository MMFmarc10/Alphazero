import time

from configs.go_5x5_config_test import Go5x5ConfigTest
from games.Go5x5 import Go

import math
from collections import OrderedDict

import numpy as np

from games.Go9x9 import Go9x9


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
    def __init__(self, game, C):
        self.root = Node(1, game, C=C, visit_count=1)
        self.search_path = []
        self.selected_node = None
        self.value = None
        self.terminada = None


# Implementación de Monte Carlo Tree Search (MCTS) para AlphaZero.
# Realiza simulaciones sobre múltiples juegos en paralelo
class MCTS:

    def __init__(self, games, num_simulations, mode="test"):
        self.games = games
        self.num_parallel_mcts = len(games)
        self.simulations = num_simulations
        self.C = config.C
        self.dirichlet_alpha = config.dirichlet_alpha
        self.exploration_fraction = config.exploration_fraction
        self.mode = mode
        self.cache = MCTSCache()
        self.cache_hits = 0
        self.cache_misses = 0


    # Ejecuta todas las simulaciones MCTS para todos los juegos en paralelo.
    # Realiza las fases de selección, expansión, backpropagation y devuelve distribuciones de política de cada partida.
    def iniciar(self):

        parallel_mcts = [MCTSInfo(game, self.C) for game in self.games]

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
                moves.append(child.move)
                distribution[child.move] = child.visit_count

            if np.sum(distribution) != 0:
                distribution /= np.sum(distribution)
            else:
                print("error")
                root.game.print_board()

            resultados.append((moves, distribution, root.value()))

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

        distribuciones, valores = self.obtener_distribuciones_batch_mock(nodos_a_expandir)

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

        distribuciones, valores = self.obtener_distribuciones_batch_mock(nodos_a_expandir)

        for mcts_i, distribucion, value in zip(mcts_pendientes_prediccion, distribuciones, valores):
            mcts_i.selected_node.expand(distribucion)
            mcts_i.value = value

            # Guardamos en caché
            encoded = mcts_i.selected_node.game.encode_board()
            key = encoded.tobytes()
            self.cache.put(key, (distribucion, value))

    # Propaga el valor estimado por el modelo hacia atrás desde el nodo hoja hasta la raíz
    def backpropagate(self, search_path, value):
        oponente = 1
        for node in reversed(search_path):
            node.value_sum += value * oponente
            node.visit_count += 1
            oponente = oponente * -1

    # Solicita al modelo una predicción de política y valor para varios tableros en batch.
    # Aplica una máscara de movimientos legales a la política resultante.
    def obtener_distribuciones_batch_mock(self, nodos, distribucion=None, value=0.3):
        distribuciones_legales = []
        valores = []

        # Si no pasas ninguna distribución, crea una plana
        for node in nodos:
            legal_mask = node.game.legal_moves_mask()

            if distribucion is None:
                legal_policy = legal_mask.astype(np.float32)
                legal_policy /= np.sum(legal_policy)
            else:
                # Distribución que pasas tú, ajustada a los movimientos legales
                legal_policy = np.array(distribucion) * legal_mask
                if np.sum(legal_policy) == 0:
                    legal_policy = legal_mask  # fallback si la distribución es ilegal
                legal_policy /= np.sum(legal_policy)

            distribuciones_legales.append(legal_policy)
            valores.append(value)  # Valor fijo que tú defines

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
    def __init__(self, prior, game, move=None, C=1.5, visit_count=0):
        self.prior = prior
        self.visit_count = visit_count
        self.value_sum = 0
        self.children = []
        self.game = game
        self.move = move
        self.C = C

    # Devuelve el valor del nodo
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    # Indica si el nodo ya ha sido expandido
    def is_expanded(self):
        return len(self.children) > 0

    # Genera nodos hijos a partir de la política del modelo
    def expand(self, distribution):

        for move, prob in enumerate(distribution):

            if prob != 0:
                node = Node(prob, self.game, move, C=self.C)
                self.children.append(node)

    # Elige el hijo con mejor puntuación UCB (Upper Confidence Bound).
    def select(self):

        mejor_ucb = -np.inf
        mejor_nodo = None
        for node in self.children:
            ucb_score = self.get_ucb_score(node)
            if ucb_score > mejor_ucb:
                mejor_ucb = ucb_score
                mejor_nodo = node
        return mejor_nodo

    # Calcula la puntuación UCB
    def get_ucb_score(self, child):
        q_value = -child.value()
        u_value = self.C * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))
        return q_value + u_value

# Aplica temperatura a la política del selfplay para fomentar la exploración durante los primeros turnos
def aplicar_temperatura(probs, turn, config):

    if turn < config.temperature_threshold:
        temperature = config.selfplay_temperature
        logits = np.log(probs + 1e-8) / temperature
        policy = np.exp(logits) / np.sum(np.exp(logits))
        return policy
    else:
        return probs



if __name__ == '__main__':
    start = time.time()
    SEED = 42


    np.random.seed(SEED)

    config = Go5x5ConfigTest()


    games = [Go9x9() for _ in range(50)]
    mc= MCTS(games,2000)
    results = mc.iniciar()
    moves,probs,value = results[0]
    print(probs)
    print(value)



    end = time.time()

    print(f"Tiempo total de ejecución: {end - start:.3f} segundos")
