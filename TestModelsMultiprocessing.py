import os

from AlphaZeroModel import AlphaZeroModel
from MCTSmultiprocessing import MCTS

from configs.tres_en_raya_config import TresEnRayaConfig
from configs.cuatro_en_raya_config import CuatroEnRayaConfig
from collections import Counter

import numpy as np
import torch
import torch.multiprocessing as mp

from games.TresEnRaya import TresEnRaya
from games.CuatroEnRaya import CuatroEnRaya



def aplicar_temperatura(probs,config):

    temperature = config.test_temperature

    # Aplicar softmax con temperatura
    logits = np.log(probs + 1e-8) / temperature
    policy = np.exp(logits) / np.sum(np.exp(logits))
    return policy

# Proceso que ejecuta múltiples partidas en paralelo con MCTS y devuelve datos de entrenamiento
def self_play_test_worker(game_class, mcts_class, config, result_queue, request_model_queue, response_model_queue, wid):
    class GameInfo:
        def __init__(self, game):
            self.game = game
            self.winner = None
            self.terminado = False

    datos = []

    print((config.num_test_games// config.num_test_workers) // config.test_games_per_worker)
    for _ in range((config.num_test_games// config.num_test_workers) // config.test_games_per_worker):

        player_model_1_starts = {1: "model1", -1: "model2"}
        player_model_2_starts = {1: "model2", -1: "model1"}

        start_conf = (player_model_1_starts,player_model_2_starts)

        for player_model in start_conf:

            games_info = [GameInfo(game_class()) for _ in range(config.test_games_per_worker)]

            while any(not g.terminado for g in games_info):

                games_info_active = [g for g in games_info if not g.terminado]

                # Divides los juegos activos por modelo
                games_info_model1 = [g for g in games_info_active if g.game.player == 1]
                games_info_model2 = [g for g in games_info_active if g.game.player == -1]

                # Ejecutas los MCTS por separado
                games_model1 = [g.game for g in games_info_model1]
                games_model2 = [g.game for g in games_info_model2]

                if games_model1:
                    resultados_mcts_model1 = mcts_class(games_model1, config.test_num_simulations, config,request_model_queue, response_model_queue,
                                                        wid, "test", player_model[1]).iniciar()
                else:
                    resultados_mcts_model1 = []

                if games_model2:
                    resultados_mcts_model2 = mcts_class(games_model2, config.test_num_simulations,config, request_model_queue, response_model_queue,
                                                        wid, "test",player_model[-1]).iniciar()
                else:
                    resultados_mcts_model2 = []

                # Juntas ambos
                games_info_active = games_info_model1 + games_info_model2
                resultados_mcts = resultados_mcts_model1 + resultados_mcts_model2

                for game_info, (_, probs, _) in zip(games_info_active, resultados_mcts):

                    probs_temperature = aplicar_temperatura(probs, config)

                    move = np.random.choice(game_info.game.ACTION_SIZE, p=probs_temperature)
                    game_info.game.make_move(move)


                    terminado = game_info.game.is_game_over()

                    if terminado:
                        game_info.terminado = True
                        game_info.winner = game_info.game.get_game_result()

                        if game_info.winner != 0:
                            datos.append(player_model[game_info.winner])
                        else:
                            datos.append("empate")



    result_queue.put(datos)


def inference_test_worker(game_class, model_info_1, model_info_2, device, config, request_queue, response_queues):
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
            batch_tensor = torch.stack(board_batch).to(device)

            with torch.no_grad():
                if turn_owner == "model1":
                    policy_batch, value_batch = model1(batch_tensor)
                else:
                    policy_batch, value_batch = model2(batch_tensor)

            response_queues[wid].put((
                [p.cpu() for p in policy_batch],
                [v.cpu() for v in value_batch]
            ))

        except:
            continue


def evaluate_models(game_class,mcts_class,model_class1,model_class2,model_path1,model_path2, config, device):

    request_model_queue = mp.Queue()
    response_model_queues = [mp.Queue() for _ in range(config.num_test_workers)]
    result_queue = mp.Queue()


    # Proceso de inferencia del modelo
    inference_proc = mp.Process(
        target=inference_test_worker,
        args=(game_class, (model_class1,model_path1),(model_class2,model_path2),device, config, request_model_queue,
              response_model_queues)
    )
    inference_proc.start()

    # procesos para generar partidas
    workers = []
    for wid in range(config.num_test_workers):
        p = mp.Process(
            target=self_play_test_worker,
            args=(
                game_class,
                mcts_class,
                config,
                result_queue,
                request_model_queue,
                response_model_queues[wid],
                wid
            )
        )
        p.start()
        workers.append(p)

    all_data = []
    for _ in range(config.num_test_workers):
        all_data.extend(result_queue.get())

    for p in workers:
        p.join()

    request_model_queue.put(None)
    inference_proc.join()

    conteo = Counter(all_data)

    resultados = {
        "modelo1": conteo.get("model1", 0),
        "modelo2": conteo.get("model2", 0),
        "empates": conteo.get("empate", 0)
    }

    print("\n--- Resultados ---")
    print(f"Modelo 1 ({model_path_1}): {resultados['modelo1']} victorias")
    print(f"Modelo 2 ({model_path_2}): {resultados['modelo2']} victorias")
    print(f"Empates: {resultados['empates']}")

    return all_data


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = CuatroEnRayaConfig()

    import os


    model_path_1 = "model_versions/4EnRalla_model_/4EnRalla_model_0.pth"
    model_path_2 = "model_versions/4EnRalla_model_/4EnRalla_model_15.pth"

    resultados = evaluate_models(
        CuatroEnRaya,
        MCTS,
        AlphaZeroModel,
        AlphaZeroModel,
        model_path_1,
        model_path_2,
        config,
        device
    )