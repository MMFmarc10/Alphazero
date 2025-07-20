import queue
import time

import numpy as np
import torch



# Contiene el estado de una partida en self-play
class GameInfo:
    def __init__(self, game):
        self.game = game
        self.history = []
        self.winner = None
        self.terminado = False
        self.num_turn = 1


# Aplica temperatura a la política del selfplay para fomentar la exploración durante los primeros turnos
def aplicar_temperatura(probs, turn, config):
    if turn < config.temperature_threshold:
        temperature = config.selfplay_temperature
        logits = np.log(probs + 1e-8) / temperature
        policy = np.exp(logits) / np.sum(np.exp(logits))
        return policy
    else:
        return probs


# Proceso que ejecuta múltiples partidas en paralelo con MCTS y devuelve datos de entrenamiento
def self_play_worker(game_class, mcts_class, config, result_queue, request_model_queue, response_model_queue, wid):
    datos = []

    for _ in range(config.games_for_worker // config.simultaneous_games_per_worker):
        games_info = [GameInfo(game_class()) for _ in range(config.simultaneous_games_per_worker)]

        while any(not g.terminado for g in games_info):

            games_info_active = [g for g in games_info if not g.terminado]

            games = [g.game for g in games_info_active]

            resultados_mcts = mcts_class(games, config.num_mcts_simulations, config, request_model_queue,
                                         response_model_queue, wid).iniciar()

            for game_info, (_, probs, _) in zip(games_info_active, resultados_mcts):

                probs_temperature = aplicar_temperatura(probs, game_info.num_turn, config)

                encoded_board = game_info.game.encode_board()
                jugador_actual = game_info.game.player

                game_info.history.append((encoded_board, probs_temperature, jugador_actual))

                move = np.random.choice(game_info.game.ACTION_SIZE, p=probs_temperature)
                game_info.game.make_move(move)

                game_info.num_turn += 1

                terminado = game_info.game.is_game_over()

                if terminado:
                    game_info.terminado = True
                    game_info.winner = game_info.game.get_game_result()

        for game_info in games_info:
            for encoded_board, probs, player_history in game_info.history:
                z = 0 if game_info.winner == 0 else (1 if player_history == game_info.winner else -1)
                datos.append((encoded_board, probs, z))

    result_queue.put(datos)


# Proceso que ejecuta múltiples partidas en paralelo con MCTS y devuelve datos de entrenamiento
def self_play_worker1(game_class, mcts_class, config, result_queue, request_model_queue, response_model_queue, wid):
    datos = []

    for _ in range(config.games_for_worker // config.simultaneous_games_per_worker):
        games_info = [GameInfo(game_class()) for _ in range(config.simultaneous_games_per_worker)]

        while any(not g.terminado for g in games_info):

            games_info_active = [g for g in games_info if not g.terminado]

            # Separar en dos grupos: uno para jugadas directas, otro para MCTS
            mcts_games_info = []
            direct_games_info = []

            for g in games_info_active:
                if len(g.game.legal_moves()) == 1:
                    direct_games_info.append(g)
                else:
                    mcts_games_info.append(g)

            # Procesar los que requieren MCTS
            if mcts_games_info:
                games = [g.game for g in mcts_games_info]

                resultados_mcts = mcts_class(games, config.num_mcts_simulations, config, request_model_queue,
                                             response_model_queue, wid).iniciar()

                for game_info, (_, probs, _) in zip(mcts_games_info, resultados_mcts):
                    probs_temperature = aplicar_temperatura(probs, game_info.num_turn, config)
                    encoded_board = game_info.game.encode_board()
                    jugador_actual = game_info.game.player
                    game_info.history.append((encoded_board, probs_temperature, jugador_actual))
                    move = np.random.choice(game_info.game.ACTION_SIZE, p=probs_temperature)
                    game_info.game.make_move(move)
                    game_info.num_turn += 1
                    if game_info.game.is_game_over():
                        game_info.terminado = True
                        game_info.winner = game_info.game.get_game_result()

            if direct_games_info:
                # Procesar directamente los que solo tienen una acción
                for game_info in direct_games_info:
                    legal = game_info.game.legal_moves()
                    move = legal[0]
                    probs = np.zeros(game_info.game.ACTION_SIZE)
                    probs[move] = 1.0
                    encoded_board = game_info.game.encode_board()
                    jugador_actual = game_info.game.player
                    game_info.history.append((encoded_board, probs, jugador_actual))
                    game_info.game.make_move(move)
                    game_info.num_turn += 1
                    if game_info.game.is_game_over():
                        game_info.terminado = True
                        game_info.winner = game_info.game.get_game_result()

        for game_info in games_info:
            for encoded_board, probs, player_history in game_info.history:
                z = 0 if game_info.winner == 0 else (1 if player_history == game_info.winner else -1)
                datos.append((encoded_board, probs, z))

    result_queue.put(datos)


# Proceso centralizado que realiza inferencia del modelo para todos los workers de self-play
def inference_worker(game_class, model_class, model_path, device, config, request_queue, response_queues):
    model = model_class(game_class(), config.num_residual_blocks, config.num_filters)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    while True:
        try:
            item = request_queue.get(timeout=0.1)

            if item is None:
                break

            board_batch, wid = item
            # Conversión aquí, cuando ya está dentro del proceso correcto
            batch_tensor = board_batch.to(dtype=torch.float32, device=device)
            #print(f"[InferenceWorker2] Batch shape: {batch_tensor.shape}")

            #batch_tensor = board_batch.to(device)

            with torch.no_grad():
                policy_batch, value_batch = model(batch_tensor)
                policy_batch = torch.softmax(policy_batch, dim=1)

            response_queues[wid].put((
                policy_batch.cpu(),
                value_batch.cpu()
            ))

        except:
            continue




def inference_worker2(game_class, model_class, model_path, device, config,
                     request_queue, response_queues,
                     max_batch_size=128, max_wait_time=0.01):
    # Cargar modelo
    model = model_class(game_class(), config.num_residual_blocks, config.num_filters)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    while True:
        batch = []
        start = time.time()

        # 🔁 Acumular peticiones hasta formar un batch o agotar tiempo
        while  (time.time() - start) < max_wait_time:
            try:
                item = request_queue.get(timeout=max_wait_time)
                if item is None:
                    return  # Señal de parada
                board_batch, wid = item
                for i in range(board_batch.shape[0]):
                    # Convertimos a float y almacenamos con pin_memory
                    sample = board_batch[i].to(dtype=torch.float32).pin_memory()
                    batch.append((sample, wid))
            except queue.Empty:
                break  # Sal del bucle si no hay más

        if not batch:
            continue

        # 📦 Agrupar batch final
        inputs, wids = zip(*batch)
        input_tensor = torch.stack(inputs).to(device, non_blocking=True)
        #print(f"[InferenceWorker2] Batch shape: {input_tensor.shape}")
        with torch.no_grad():
            policy_batch, value_batch = model(input_tensor)
            policy_batch = torch.softmax(policy_batch, dim=1)

        # ⚡ Una sola conversión a .cpu() por tensor completo
        policies_cpu = policy_batch.cpu()
        values_cpu = value_batch.cpu()

        # 📨 Agrupar respuestas por worker
        outputs_per_worker = {wid: [] for wid in set(wids)}
        for i, wid in enumerate(wids):
            outputs_per_worker[wid].append((policies_cpu[i], values_cpu[i]))

        # Enviar respuestas agrupadas por worker
        for wid, outputs in outputs_per_worker.items():
            policies, values = zip(*outputs)
            response_queues[wid].put((torch.stack(policies), torch.stack(values)))


# Proceso que ejecuta múltiples partidas en paralelo con MCTS y devuelve que modelo ha ganado
def evaluation_worker(game_class, mcts_class, config, result_queue, request_model_queue, response_model_queue,
                              wid):

    datos = []

    for _ in range((config.num_test_games // config.num_test_workers) // config.test_games_per_worker):


        player_model_1_starts = {1: "model1", -1: "model2"}
        player_model_2_starts = {1: "model2", -1: "model1"}

        start_conf = (player_model_1_starts, player_model_2_starts)

        for player_model in start_conf:

            games_info = [GameInfo(game_class()) for _ in range(config.test_games_per_worker)]

            while any(not g.terminado for g in games_info):

                games_info_active = [g for g in games_info if not g.terminado]

                games_info_model1 = [g for g in games_info_active if g.game.player == 1]
                games_info_model2 = [g for g in games_info_active if g.game.player == -1]

                games_model1 = [g.game for g in games_info_model1]
                games_model2 = [g.game for g in games_info_model2]

                if games_model1:


                    resultados_mcts_model1 = mcts_class(games_model1, config.test_num_simulations, config,
                                                            request_model_queue, response_model_queue,
                                                            wid, "test", player_model[1]).iniciar()
                else:
                    resultados_mcts_model1 = []

                if games_model2:

                    resultados_mcts_model2 = mcts_class(games_model2, config.test_num_simulations, config,
                                                            request_model_queue, response_model_queue,
                                                            wid, "test", player_model[-1]).iniciar()
                else:
                    resultados_mcts_model2 = []


                games_info_active = games_info_model1 + games_info_model2
                resultados_mcts = resultados_mcts_model1 + resultados_mcts_model2

                for game_info, (_, probs, _) in zip(games_info_active, resultados_mcts):

                    probs_temperature = aplicar_temperatura_test(probs, config, game_info.num_turn)

                    move = np.random.choice(game_info.game.ACTION_SIZE, p=probs_temperature)
                    game_info.game.make_move(move)

                    game_info.num_turn += 1

                    terminado = game_info.game.is_game_over()

                    if terminado:
                        game_info.terminado = True
                        game_info.winner = game_info.game.get_game_result()

                        if game_info.winner != 0:
                                datos.append(player_model[game_info.winner])
                        else:
                            datos.append("empate")


    result_queue.put(datos)

def inference_evaluation_worker(game_class, model_info_1, model_info_2, device, config, request_queue, response_queues):


    model_class_1, model_path_1 = model_info_1
    model_class_2, model_path_2 = model_info_2

    model1 = model_class_1(game_class(), config.num_residual_blocks, config.num_filters)
    model1.load_state_dict(torch.load(model_path_1, map_location=device))
    model1.to(device)
    model1.eval()

    model2 = model_class_1(game_class(), config.num_residual_blocks, config.num_filters)
    model2.load_state_dict(torch.load(model_path_2, map_location=device))
    model2.to(device)
    model2.eval()

    while True:
        try:
            item = request_queue.get(timeout=0.1)
            if item is None:
                break

            board_batch, wid, turn_owner = item  # 'model1' o 'model2'

            #batch_tensor = board_batch.to(device)

            batch_tensor = board_batch.to(dtype=torch.float32, device=device)


            with torch.no_grad():

                if turn_owner == "model1":
                    policy_batch, value_batch = model1(batch_tensor)

                else:
                    policy_batch, value_batch = model2(batch_tensor)


                policy_batch = torch.softmax(policy_batch, dim=1)

            response_queues[wid].put((
                policy_batch.cpu(),
                value_batch.cpu()
            ))

        except:
            continue

def aplicar_temperatura_test(probs, config,turn):

    if turn > config.test_temperature_threshold:
        temperature = config.test_temperature_after
    else:
        temperature = config.test_temperature_before

    logits = np.log(probs + 1e-8) / temperature
    policy = np.exp(logits) / np.sum(np.exp(logits))
    return policy