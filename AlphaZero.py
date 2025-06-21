import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from AlphaZeroModel import AlphaZeroModel
from MCTSmultiprocessing import MCTS
from games.TresEnRaya import TresEnRaya


# Clase de configuración con todos los parámetros de AlphaZero
class AlphaZeroConfig:

    def __init__(self):
        # Entrenamiento general
        self.num_iterations = 10

        # SelfPlay
        self.num_selfplay_games = 400
        self.num_selfplay_workers = 4
        self.simultaneous_games_per_worker = 50
        self.games_for_worker = self.num_selfplay_games // self.num_selfplay_workers
        self.selfplay_temperature = 1.25
        self.temperature_threshold = 3
        
        # Entrenamiento de red neuronal
        self.batch_size = 64
        self.learning_rate = 0.001
        self.num_epochs = 4
        self.weight_decay=0.0001

        # MCTS
        self.num_mcts_simulations = 90
        self.C = 1.5

        # Model
        self.num_residual_blocks = 4
        self.num_filters = 64


# Clase principal que coordina el ciclo de entrenamiento AlphaZero (self-play y entrenamiento)
class AlphaZero:

    def __init__(self,game_class,mcts_class,model_class,model,device,optimizer,scheduler,config):
        self.game_class = game_class
        self.mcts_class = mcts_class
        self.model_class = model_class
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config


    # Crea un Dataset personalizado para AlphaZero a partir de los datos generados
    class AlphaZeroDataset(Dataset):
        def __init__(self, data):
            self.data = data  

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            board, probs, result = self.data[idx]

            board = np.array(board, dtype=np.float32)
            probs = np.array(probs, dtype=np.float32)
            result = np.array(result, dtype=np.float32)

            board = torch.tensor(board)
            probs = torch.tensor(probs)
            result = torch.tensor(result)

            return board, probs, result

    # Bucle principal de AlphaZero: genera partidas, entrena el modelo y finalmente guarda el modelo entrenado
    def run(self):

        for iteration in range(self.config.num_iterations):

            print(f"\nIteración {iteration + 1}/{self.config.num_iterations}")

            start_time = time.time()

            self.model.eval()
            print("Generando partidas...")

            data = self.generate_games()
            #save_selfplay_data("logs/selfplay_workers_paralel3.txt", data,iteration)
            
            print(f"Total de posiciones generadas: {len(data)}")

            dataset = self.AlphaZeroDataset(data)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

            self.model.train()
            for epoch in range(self.config.num_epochs):
                avg_loss = self.train(dataloader,iteration,epoch)
                print(f"   Epoch {epoch + 1}/{self.config.num_epochs} - Pérdida media: {avg_loss:.4f}")

            self.save_model(iteration)
            print(f"Modelo guardado como: model_iter_{iteration}.pth")

            self.scheduler.step()

            duration = time.time() - start_time
            print(f"Duración de la iteración: {duration:.2f} segundos")

    # Genera partidas mediante self-play en paralelo y devuelve los datos recolectados
    def generate_games(self):
  
        request_model_queue = mp.Queue()
        response_model_queues = [mp.Queue() for _ in range(self.config.num_selfplay_workers)]
        result_queue = mp.Queue()
        model_path = f"temp_model_iter_{int(time.time())}.pth"
        torch.save(self.model.state_dict(), model_path)

        # Proceso de inferencia del modelo
        inference_proc = mp.Process(
            target=inference_worker,
            args=(self.game_class,self.model_class,model_path,self.device, self.config,request_model_queue, response_model_queues)
        )
        inference_proc.start()

        # procesos para generar partidas
        workers = []
        for wid in range(self.config.num_selfplay_workers):
            p = mp.Process(
                target=self_play_worker,
                args=(
                    self.game_class,
                    self.mcts_class,
                    self.config,
                    result_queue,
                    request_model_queue,
                    response_model_queues[wid],
                    wid
                )
            )
            p.start()
            workers.append(p)


        all_data = []
        for _ in range(self.config.num_selfplay_workers):
            all_data.extend(result_queue.get())

        for p in workers:
            p.join()

        request_model_queue.put(None)
        inference_proc.join()
        os.remove(model_path)

        return all_data

    # Entrena el modelo con los datos recolectados
    def train(self, dataloader,iter_num,epoch):

        self.model.to(device)
        self.model.train()
        total_loss = 0
    
        for i, (boards, probs, results) in enumerate(dataloader):

            boards, probs,results = boards.to(self.device), probs.to(self.device), results.to(self.device)

            self.optimizer.zero_grad()

            pred_pi, pred_v = self.model(boards)

            #log_training_batch("logs/train_debug_paralel3.txt", boards[:2], pred_pi[:2], pred_v[:2], probs[:2], results[:2], iter_num, epoch)

            loss_pi = F.cross_entropy(pred_pi, probs)
            loss_v = F.mse_loss(pred_v, results.unsqueeze(1))
            loss = loss_pi + loss_v

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        total_loss /= len(dataloader)
        return total_loss

    # Guarda el modelo entrenado
    def save_model(self, iteration):

        os.makedirs("model_versions", exist_ok=True)
        
        filename = f"model_versions/model_temperature1_{iteration}.pth"
        
        torch.save(self.model.state_dict(), filename)


def aplicar_temperatura(probs, turn, config):

    if turn < config.temperature_threshold:
        temperature = config.selfplay_temperature

        # Aplicar softmax con temperatura
        logits = np.log(probs + 1e-8) / temperature
        policy = np.exp(logits) / np.sum(np.exp(logits))
        return policy

    else:
        # No aplicar temperatura, usar directamente las probabilidades
        return probs


# Proceso que ejecuta múltiples partidas en paralelo con MCTS y devuelve datos de entrenamiento
def self_play_worker(game_class,mcts_class,config, result_queue, request_model_queue,response_model_queue, wid):

    class GameInfo:
        def __init__(self, game):
            self.game = game
            self.history = []
            self.winner = None
            self.terminado = False
            self.num_turn = 0

    datos = []

    for _ in range(config.games_for_worker // config.simultaneous_games_per_worker):
        games_info = [GameInfo(game_class()) for _ in range(config.simultaneous_games_per_worker)]

        while any(not g.terminado for g in games_info):

            games_info_active = [g for g in games_info if not g.terminado]
            games = [g.game for g in games_info_active]

            resultados_mcts = mcts_class(games, config.num_mcts_simulations, config.C, request_model_queue,response_model_queue, wid).iniciar()


            for game_info, (_, probs, _) in zip(games_info_active, resultados_mcts):


                probs_temperature = aplicar_temperatura(probs, game_info.num_turn, config)
                #print("**************************")
                #print("probs:")
                #print(np.array2string(probs, precision=3, suppress_small=True, separator=", "))

                #print("probs_temperature:")
                #print(np.array2string(probs_temperature, precision=3, suppress_small=True, separator=", "))
                #print("**************************")

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

    result_queue.put(datos)


# Proceso que ejecuta inferencia centralizada: el modelo realiza las predicciones que recibe en la cola
def inference_worker(game_class,model_class,model_path, device, config,request_queue, response_queues):
    model = model_class(game_class(),config.num_residual_blocks, config.num_filters)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    while True:
        try:
            item = request_queue.get(timeout=0.1)

            if item is None:
                break

            board_batch, wid = item
    
            batch_tensor = torch.stack(board_batch).to(device)

            with torch.no_grad():
                policy_batch, value_batch = model(batch_tensor)


            response_queues[wid].put((
                [p.cpu() for p in policy_batch],
                [v.cpu() for v in value_batch]
            ))

        except:
            continue


def save_selfplay_data(path, data, iter_num):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(f"\n=========== SELFPLAY - ITERACIÓN {iter_num} ===========\n\n")
        for i, (board, probs, result) in enumerate(data):
            f.write(f"--- EJEMPLO {i} ---\n")
            f.write("Board:\n")
            f.write(np.array2string(np.array(board), separator=', '))
            f.write("\nProbs:\n")
            f.write(np.array2string(np.array(probs), separator=', ', precision=3))
            f.write(f"\nSuma probs: {np.sum(probs):.4f}\n")
            f.write(f"Resultado Z: {result}\n\n")


def log_training_batch(path, boards, pred_pi, pred_v, target_pi, target_v, iter_num, epoch_num):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(f"\n=========== ITERACIÓN {iter_num} - EPOCH {epoch_num} ===========\n\n")
        for i in range(len(boards)):
            f.write(f"--- BATCH EJEMPLO {i} ---\n")
            f.write("Board:\n")
            f.write(np.array2string(boards[i].cpu().numpy(), separator=', '))
            f.write("\nTarget probs:\n")
            f.write(np.array2string(target_pi[i].cpu().numpy(), separator=', ', precision=3))
            f.write("\nPredicted logits (sin softmax):\n")
            f.write(np.array2string(pred_pi[i].detach().cpu().numpy(), separator=', ', precision=3))
            f.write(f"\nTarget value: {target_v[i].item():.3f}\n")
            f.write(f"Predicted value: {pred_v[i].detach().item():.3f}\n\n")
   



if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    config = AlphaZeroConfig()
    game = TresEnRaya()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroModel(game, config.num_residual_blocks, config.num_filters)
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    alphazero = AlphaZero(TresEnRaya,MCTS,AlphaZeroModel,model,device,optimizer,scheduler,config)
    alphazero.run()



