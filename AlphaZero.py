import json
import os
import shutil
import time
import random
from collections import Counter
import logging
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from configs.cuatro_en_raya_config import CuatroEnRayaConfig
from configs.go_9x9_config import Go9x9Config
from configs.tres_en_raya_config import TresEnRayaConfig
from games.CuatroEnRayaFast import CuatroEnRayaFast
from games.Go9x9 import Go9x9
from games.TresEnRaya import TresEnRaya
from workers import self_play_worker, inference_worker, evaluation_worker, inference_evaluation_worker, \
    inference_worker2

from AlphaZeroModel import AlphaZeroModel
from MCTSmultiprocessing3 import MCTS3
from configs.go_5x5_config import Go5x5Config
from games.Go5x5 import Go5x5


# Clase principal que coordina el ciclo de entrenamiento AlphaZero (self-play y entrenamiento)
class AlphaZero:

    def __init__(self,game_class,mcts_class,model_class,model,device,optimizer,scheduler,config):

        self.game_class = game_class
        self.mcts_class = mcts_class
        self.model_class = model_class

        self.model = model
        self.current_model_path = ""
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        os.makedirs("model_versions", exist_ok=True)
        self.directory_name = "Go_9x9_shared"

        self.directory_path = os.path.join("model_versions", self.directory_name)
        self.best_model_path =os.path.join("model_versions", self.directory_name ,self.directory_name + "Best.pth")
        self.last_model_path = os.path.join("model_versions",self.directory_name,"iterations")

        os.makedirs(self.directory_path, exist_ok=True)
        os.makedirs(self.last_model_path, exist_ok=True)

        self.replay_buffer = ReplayBuffer(config.max_size)
        self.save_configuration()

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.StreamHandler(),  # Terminal
                logging.FileHandler(os.path.join(self.directory_path, "training.log"), mode="a", encoding="utf-8"),

            ]
        )


    # Bucle principal de AlphaZero: genera partidas, entrena el modelo y finalmente guarda el modelo entrenado
    def run(self):

        file = self.save_model(0)
        self.save_best_model(file)


        for iteration in range(self.config.num_iterations):

            logging.info(f"\nIteración {iteration + 1}/{self.config.num_iterations}")

            start_time = time.time()

            self.model.eval()
            logging.info("Generando partidas...")

            data = self.generate_games(iteration)

            # Al añadir datos de self-play
            self.replay_buffer.add(data)

            # Para samplear 25000 posiciones aleatorias:
            sample_size = min(len(self.replay_buffer), self.config.train_sample)
            buffer_data = self.replay_buffer.sample(sample_size)

            logging.info(f"Total de posiciones generadas: {len(data)}")
            logging.info(f"Total de posiciones guardadas en el buffer: {len(self.replay_buffer)}")
            logging.info(f"Total de posiciones utilizadas del buffer para entrenar: {len(buffer_data)}")

            dataset = AlphaZeroDataset(buffer_data)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

            self.model.train()
            for epoch in range(self.config.num_epochs):
                avg_loss = self.train(dataloader)
                logging.info(f"   Epoch {epoch + 1}/{self.config.num_epochs} - Pérdida media: {avg_loss:.4f}")

            self.save_model(iteration + 1)
            filename = os.path.join(self.last_model_path, f"{self.directory_name}{iteration + 1}.pth")
            logging.info(f"Modelo guardado como: {filename}")

            self.scheduler.step()
            if iteration % self.config.evaluation_frequency == 0:
                self.evaluate()


            duration = time.time() - start_time
            logging.info(f"Duración de la iteración: {duration:.2f} segundos")






    # Genera partidas mediante self-play en paralelo y devuelve los datos recolectados
    def generate_games(self,iteration):
  
        request_model_queue = mp.Queue()
        response_model_queues = [mp.Queue() for _ in range(self.config.num_selfplay_workers)]
        result_queue = mp.Queue()
        model_path = self.current_model_path

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

        return all_data


    # Entrena el modelo con los datos recolectados
    def train(self, dataloader):

        self.model.to(self.device)
        self.model.train()
        total_loss = 0
    
        for i, (boards, probs, results) in enumerate(dataloader):

            boards, probs,results = boards.to(self.device), probs.to(self.device), results.to(self.device)

            self.optimizer.zero_grad()

            pred_p, pred_v = self.model(boards)

            loss_p = F.cross_entropy(pred_p, probs)
            loss_v = F.mse_loss(pred_v, results.unsqueeze(1))
            loss = loss_p + loss_v

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        total_loss /= len(dataloader)
        return total_loss

    def evaluate(self):

        current_model_path =  self.current_model_path

        request_model_queue = mp.Queue()
        response_model_queues = [mp.Queue() for _ in range(config.num_test_workers)]
        result_queue = mp.Queue()

        # Proceso de inferencia del modelo
        inference_proc = mp.Process(
            target=inference_evaluation_worker,
            args=(self.game_class, (self.model_class, current_model_path), (self.model_class, self.best_model_path), self.device, self.config,
                  request_model_queue,
                  response_model_queues)
        )
        inference_proc.start()

        # procesos para generar partidas
        workers = []
        for wid in range(config.num_test_workers):
            p = mp.Process(
                target=evaluation_worker,
                args=(
                    self.game_class,
                    self.mcts_class,
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


        wins1 = resultados['modelo1']
        wins2 = resultados['modelo2']
        draws = resultados['empates']

        # Total de partidas jugadas
        total_games = wins1 + wins2 + draws

        # Score del modelo actual (modelo1): victoria = 1, empate = 0.5, derrota = 0
        score_actual = (wins1 + 0.5 * draws) / total_games if total_games > 0 else 0

        # Score del modelo best (modelo2), opcional
        score_best = (wins2 + 0.5 * draws) / total_games if total_games > 0 else 0

        logging.info("\n--- Evaluación ---")
        logging.info(f"Modelo Actual ({self.current_model_path}): {wins1} victorias")
        logging.info(f"Modelo best ({self.best_model_path}): {wins2} victorias")
        logging.info(f"Empates: {draws}")
        logging.info(f"Winrate: Modelo Actual: {score_actual * 100:.2f}%, Modelo Best: {score_best * 100:.2f}%")
        logging.info("")

        if score_actual >= self.config.test_win_rate_threshold:
            self.save_best_model(self.current_model_path)
            logging.info(" - Nuevo modelo guardado como Best.")
            logging.info("")

        else:
            logging.info(" - El nuevo modelo no ha superado el threshold. Restaurando modelo Best.")
            logging.info("")
            best_weights = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(best_weights)
            self.current_model_path = self.best_model_path


    # Guarda el modelo actual en un archivo .pth
    def save_model(self, iteration):

        filename = os.path.join(self.last_model_path, f"{self.directory_name}{iteration}.pth")
        torch.save(self.model.state_dict(), filename)
        self.current_model_path = filename
        return filename


    def save_best_model(self, model_path):
        """Guarda el modelo ganador como el nuevo best."""
        shutil.copy(model_path, self.best_model_path)

    # Guarda la configuración actual en un archivo config.json
    def save_configuration(self):

        # Guardar configuración como config.json
        config_dict = vars(self.config)
        config_path = os.path.join(self.directory_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        pass


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = []   # Guarda tuplas: (encoded_board, probs, z)
        self.ptr = 0       # Índice circular para sobrescribir cuando esté lleno

    def add(self, data):
        """Añade una lista de ejemplos al buffer."""
        for item in data:
            if len(self.buffer) < self.max_size:
                self.buffer.append(item)  # Aún hay espacio
            else:
                # Sobrescribe usando índice circular
                self.buffer[self.ptr] = item
                self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, sample_size):
        """Devuelve una muestra aleatoria del buffer."""
        if sample_size > len(self.buffer):
            raise ValueError(f"Intentas samplear {sample_size}, pero solo hay {len(self.buffer)} elementos.")
        return random.sample(self.buffer, sample_size)

    def __len__(self):
        """Permite usar len(buffer)."""
        return len(self.buffer)


# Dataset personalizado que prepara los datos para el entrenamiento de AlphaZero
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



if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)

    config = Go9x9Config()

    game = Go9x9()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlphaZeroModel(game, config.num_residual_blocks, config.num_filters)

    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    alphazero = AlphaZero(Go9x9,MCTS3,AlphaZeroModel,model,device,optimizer,scheduler,config)

    alphazero.run()



