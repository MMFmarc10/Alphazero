import numpy as np
import torch

from AlphaZeroModel import AlphaZeroModel
from games.TresEnRaya import TresEnRaya


def jugar_vs_modelo(model, device):

    juego = TresEnRaya()

    while True:

        if juego.player == -1:

            print("")
            juego.print_board()
            legal = juego.legal_moves()
            print(f"\nTu turno. Jugadas legales: {legal}")
            while True:
                try:
                    accion = int(input(" - Elige una acción: "))
                    if accion in legal:
                        break
                    else:
                        print("Movimiento ilegal, intenta otra vez.")
                except ValueError:
                    print("Entrada inválida, pon un número.")

            juego.make_move(accion)


        else:
            print("\nTurno del modelo...")

            encoded = torch.tensor(juego.encode_board(), dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = model(encoded)
                probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

            mask = np.array(juego.legal_moves_mask())
            probs *= mask
            if probs.sum() > 0:
                probs /= probs.sum()
            else:
                print("No hay movimientos legales.")
                break

            print(probs)
            action = np.argmax(probs)
            juego.make_move(action)
            print(f" - El modelo jugó la jugada {action}")

        terminado = juego.is_game_over()

        if terminado:
            juego.print_board()
            resultado = juego.get_game_result()
            if resultado == 0:
                print("¡Empate!")
            elif resultado == 1:
                print("¡Tú ganas!")
            else:
                print("El modelo gana.")
            break


model = AlphaZeroModel(TresEnRaya(), num_residual_blocks=4, num_filters=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load("model_versions/model_ruido_9.pth", map_location=device))
model.to(device)
model.eval()

jugar_vs_modelo(model, device)
