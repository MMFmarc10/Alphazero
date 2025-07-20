import math
import time
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

from AlphaZeroModel import AlphaZeroModel
from configs.go_5x5_config_test import Go5x5ConfigTest

from test.Go9x9Fast import Go9x9Fast


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
        self.root = Node(1, game, C=C)
        self.search_path = []
        self.selected_node = None
        self.value = None
        self.terminada = None


# Implementación de Monte Carlo Tree Search (MCTS) para AlphaZero.
# Realiza simulaciones sobre múltiples juegos en paralelo
class MCTS:

    def __init__(self, games, num_simulations, config, model,device):
        self.games = games
        self.num_parallel_mcts = len(games)
        self.simulations = num_simulations
        self.model = model
        self.device = device
        self.C = config.C
        self.dirichlet_alpha = config.dirichlet_alpha
        self.exploration_fraction = config.exploration_fraction

        self.cache = MCTSCache()
        self.cache_hits = 0
        self.cache_misses = 0

    # Ejecuta todas las simulaciones MCTS para todos los juegos en paralelo.
    # Realiza las fases de selección, expansión, backpropagation y devuelve distribuciones de política de cada partida.
    def iniciar(self):

        parallel_mcts = [MCTSInfo(game, self.C) for game in self.games]

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
            search_path.append(node)

        return node

    # Expande los nodos seleccionados usando la red neuronal.
    def expand_nodes(self, expandable_pararlel_mcts):

        if len(expandable_pararlel_mcts) > 0:

            nodos_a_expandir = [mcts.selected_node for mcts in expandable_pararlel_mcts]

            distribuciones, valores = self.obtener_distribuciones_batch(nodos_a_expandir)

            for mcts_i, distribucion, value in zip(
                    [m for m in expandable_pararlel_mcts if not m.terminada], distribuciones, valores
            ):

                distribucion = self.aplicar_ruido_dirichlet(distribucion)
                mcts_i.selected_node.expand(distribucion)
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
                self.cache_hits += 1

                distribucion = self.aplicar_ruido_dirichlet(distribucion)
                node.expand(distribucion)
                mcts_i.value = value
            else:
                self.cache_misses += 1
                mcts_pendientes_prediccion.append(mcts_i)

        if not mcts_pendientes_prediccion:
            return

        nodos_a_expandir = [mcts.selected_node for mcts in mcts_pendientes_prediccion]

        distribuciones, valores = self.obtener_distribuciones_batch(nodos_a_expandir)

        for mcts_i, distribucion, value in zip(mcts_pendientes_prediccion, distribuciones, valores):
            if mcts_i.selected_node is mcts_i.root:
                distribucion = self.aplicar_ruido_dirichlet(distribucion)
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

    def obtener_distribuciones_batch(self, nodos):
        encoded_boards = [torch.tensor(node.game.encode_board(), dtype=torch.float32) for node in nodos]
        batch_tensor = torch.stack(encoded_boards).to(self.device)

        with torch.no_grad():
            policy_tensors, value_tensors = self.model(batch_tensor)



            policy_softmax = torch.softmax(policy_tensors, dim=1)


        policy_softmax_cpu = policy_softmax.cpu()
        value_tensors_cpu = value_tensors.cpu()

        distribuciones = [p.numpy() for p in policy_softmax_cpu]
        valores = [v.item() for v in value_tensors_cpu]

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
    def __init__(self, prior, game, move=None, C=1.5):
        self.prior = prior
        self.visit_count = 0
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
                game_child = fast_clone(self.game)
                game_child.make_move(move)

                node = Node(prob, game_child, move, C=self.C)
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


# Contiene el estado de una partida en self-play
class GameInfo:
    def __init__(self, game):
        self.game = game
        self.history = []
        self.winner = None
        self.terminado = False
        self.num_turn = 0

# Proceso que ejecuta múltiples partidas en paralelo con MCTS y devuelve datos de entrenamiento
def self_play_worker(game_class,mcts_class,config, model,device):

    datos = []
    for _ in tqdm(range(config.num_selfplay_games), desc="Self-play", leave=False):
        games_info = [GameInfo(game_class()) for _ in range(config.simultaneous_games_per_worker)]

        while any(not g.terminado for g in games_info):

            games_info_active = [g for g in games_info if not g.terminado]


            games = [g.game for g in games_info_active]

            resultados_mcts = mcts_class(games, config.num_mcts_simulations,config, model,device).iniciar()


            for game_info, (_, probs, _) in zip(games_info_active, resultados_mcts):


                probs_temperature = aplicar_temperatura(probs, game_info.num_turn, config)

                encoded_board = game_info.game.encode_board()
                jugador_actual = game_info.game.player

                game_info.history.append((encoded_board, probs_temperature, jugador_actual))

                move = np.random.choice(game_info.game.ACTION_SIZE, p=probs_temperature)
                game_info.game.make_move(move)

                game_info.num_turn  += 1

                terminado = game_info.game.is_game_over()

                if terminado:
                    game_info.terminado = True
                    game_info.winner = game_info.game.get_game_result()

        for game_info in games_info:
            for encoded_board, probs, player_history in game_info.history:
                z = 0 if game_info.winner == 0 else (1 if player_history == game_info.winner else -1)
                datos.append((encoded_board, probs, z))

    return datos

# Al final del archivo Go9x9Fast.py
def fast_clone(game: Go9x9Fast) -> Go9x9Fast:
    nuevo = Go9x9Fast.__new__(Go9x9Fast)

    nuevo.board = game.board.copy()
    nuevo.player = game.player
    nuevo.turns = game.turns
    nuevo.passes = game.passes
    nuevo.komi = game.komi
    nuevo.last_board_hash = game.last_board_hash
    nuevo.group_id_board = game.group_id_board.copy()
    nuevo.next_group_id = game.next_group_id

    nuevo.black_groups = {gid: fast_clone_group(g) for gid, g in game.black_groups.items()}
    nuevo.white_groups = {gid: fast_clone_group(g) for gid, g in game.white_groups.items()}

    return nuevo
class Group:
    def __init__(self, id, color):

        self.id = id
        self.color = color  # 1 para negro, -1 para blanco
        self.stones = set()
        self.liberties = set()

    def add_stone(self, r, c):
        self.stones.add((r, c))

    def add_liberty(self, r, c):
        self.liberties.add((r, c))

    def remove_liberty(self, r, c):
        self.liberties.discard((r, c))

    def merge_with(self, other):
        self.stones |= other.stones
        self.liberties |= other.liberties
        # Las libertades no pueden incluir posiciones ocupadas por piedras del grupo
        self.liberties -= self.stones

    def clone(self):
        nuevo = Group(self.id, self.color)
        nuevo.stones = self.stones.copy()
        nuevo.liberties = self.liberties.copy()
        return nuevo


def fast_clone_group(g: Group) -> Group:
    nuevo = Group.__new__(Group)  # Evita llamar a __init__

    nuevo.id = g.id
    nuevo.color = g.color
    nuevo.stones = g.stones.copy()
    nuevo.liberties = g.liberties.copy()

    return nuevo





if __name__ == '__main__':
    start = time.time()
    go = Go9x9Fast()
    config = Go5x5ConfigTest()
    cuatro = Go9x9Fast
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroModel(Go9x9Fast(), 8, 128).to(device)  # <- modelo en GPU si hay
    model.eval()
    all_data = self_play_worker(game_class=Go9x9Fast, mcts_class=MCTS,config=config, model=model, device=device)

    end = time.time()
    print(f"⏱ Tiempo transcurrido: {end - start:.2f} segundos")
    print("📦 Total de posiciones generadas:", len(all_data))