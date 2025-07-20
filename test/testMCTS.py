import numpy as np

from MCTSmultiprocessing1 import MCTS1
from configs.cuatro_en_raya_config import CuatroEnRayaConfig
from test.CuatroEnRaya import CuatroEnRaya
import torch

# Inicialización
config = CuatroEnRayaConfig()  # O la que toque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simula 5 partidas
games = [CuatroEnRaya() for _ in range(5)]

# Configura las colas y modelo de prueba (simplificado)
request_queue = []
response_queue = []
worker_id = 0

# Crea MCTS
mcts = MCTS1(
    games,
    num_simulations=10,  # pocas simulaciones para prueba rápida
    config=config,
    request_model_queue=request_queue,
    response_model_queue=response_queue,
    worker_id=worker_id,
    mode="selfplay"
)

# ⚠️ Esto normalmente lanza el worker de inferencia y la red, aquí deberías simular las respuestas
# Para probar sin modelo, necesitas mockear obtener_distribuciones_batch()

# Para probar solo estructura, podrías hacer expand_nodes_cache() devolver distribuciones fijas

# Ejecuta el MCTS
resultados = mcts.iniciar()

# Muestra los resultados
for i, (moves, distribution, value) in enumerate(resultados):
    print(f"\nJuego {i}:")
    print(f"  Movimientos explorados: {moves}")
    print(f"  Distribución: {np.round(distribution, 3)}")
    print(f"  Valor estimado: {value}")