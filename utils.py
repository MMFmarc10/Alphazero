# Guarda los datos generados durante el self-play en un archivo de texto
import os

import numpy as np


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


# Guarda los datos de un batch de entrenamiento
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

