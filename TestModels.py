import numpy as np
import torch

from AlphaZeroModel import AlphaZeroModel
from games.TresEnRaya import TresEnRaya


def suavizar_probs(probs, temperatura):


    logits = np.log(probs + 1e-8) / temperatura
    suavizadas = np.exp(logits) / np.sum(np.exp(logits))
    return suavizadas


def seleccionar_accion(model, juego, device):
    model.eval()
    encoded = torch.tensor(juego.encode_board(), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(encoded)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

    mask = np.array(juego.legal_moves_mask())
    probs *= mask
    if probs.sum() > 0:
        probs /= probs.sum()
    else:
        return None

    probs = suavizar_probs(probs,0.3)

    return np.random.choice(len(probs), p=probs)


def jugar_partida(modelo1, modelo2, juego_clase, device, verbose=False):
    juego = juego_clase()

    modelos = {1: modelo1, -1: modelo2}

    while not juego.is_game_over():
        jugador = juego.player
        modelo = modelos[jugador]

        accion = seleccionar_accion(modelo, juego, device)
        if accion is None:
            break
        juego.make_move(accion)

        if verbose:
            juego.print_board()
            print("")

    return juego.get_game_result()


def evaluar_modelos(juego_clase, model_path_1, model_path_2, device,n_partidas=100, verbose=False):


    modelo1 = AlphaZeroModel(juego_clase(), num_residual_blocks=4, num_filters=64)
    modelo2 = AlphaZeroModel(juego_clase(), num_residual_blocks=4, num_filters=64)

    modelo1.load_state_dict(torch.load(model_path_1, map_location=device))
    modelo2.load_state_dict(torch.load(model_path_2, map_location=device))

    modelo1.to(device).eval()
    modelo2.to(device).eval()

    resultados = {"modelo1": 0, "modelo2": 0, "empates": 0}

    for i in range(n_partidas):
        resultado = jugar_partida(modelo1, modelo2, juego_clase, device, verbose)
        if resultado == 1:
            resultados["modelo1"] += 1
        elif resultado == -1:
            resultados["modelo2"] += 1
        else:
            resultados["empates"] += 1

        resultado = jugar_partida(modelo2, modelo1, juego_clase, device, verbose)
        if resultado == 1:
            resultados["modelo2"] += 1
        elif resultado == -1:
            resultados["modelo1"] += 1
        else:
            resultados["empates"] += 1

    print("\n--- Resultados ---")
    print(f"Modelo 1 ({model_path_1}): {resultados['modelo1']} victorias")
    print(f"Modelo 2 ({model_path_2}): {resultados['modelo2']} victorias")
    print(f"Empates: {resultados['empates']}")


if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluar_modelos(
        juego_clase=TresEnRaya,
        model_path_1="model_versions/model_ruido_0.pth",
        model_path_2="model_versions/model_ruido_9.pth",
        device= device,
        n_partidas=1000,
        verbose=False
    )
