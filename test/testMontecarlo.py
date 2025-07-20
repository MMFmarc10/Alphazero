from torch.utils.data import Dataset

import time
import numpy as np
import torch

from gui.MCTSsimple import MCTS
from test.Go5x5Fast import Go5x5Fast
from test.CuatroEnRaya import CuatroEnRaya
from AlphaZeroModel import AlphaZeroModel
from tqdm import tqdm

class AlphaZeroDataset(Dataset):
    def __init__(self, data):
        self.data = data  # Lista de tuplas (encoded_board, probs, z)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, probs, result = self.data[idx]

        board = torch.tensor(board, dtype=torch.float32)
        probs = torch.tensor(probs, dtype=torch.float32)
        result = torch.tensor(result, dtype=torch.float32)

        return board, probs, result


def self_play(num_partidas, game_class, mcts_class, model, device):
    datos = []

    for _ in tqdm(range(num_partidas), desc="Self-play", leave=False):
        history = []
        winner_player = 0
        game = game_class()

        while True:

            mcts = mcts_class(game, 100, 1.25, model, device)
            _, probs, _ = mcts.iniciar()
            encoded_board = game.encode_board()

            history.append((encoded_board, probs, game.player))

            move = np.random.choice(game.ACTION_SIZE, p=probs)

            game.make_move(move)

            terminado= game.is_game_over()

            if terminado:
                winner_player = game.get_game_result()
                break

        for encoded_board, probs, player_history in history:

            if winner_player == 0:
                z = 0  # empate
            else:
                z = 1 if player_history == winner_player else -1
            datos.append((encoded_board, probs, z))

    return datos


if __name__ == '__main__':
    start = time.time()
    go = Go5x5Fast()
    cuatro = CuatroEnRaya
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroModel(Go5x5Fast(), 8, 128).to(device)  # <- modelo en GPU si hay
    all_data = self_play(20, game_class=Go5x5Fast, mcts_class=MCTS, model=model, device=device)

    end = time.time()
    print(f"⏱ Tiempo transcurrido: {end - start:.2f} segundos")
    print("📦 Total de posiciones generadas:", len(all_data))