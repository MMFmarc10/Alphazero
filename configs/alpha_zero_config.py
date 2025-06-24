class AlphaZeroConfig:

    def __init__(self):
        # ==== Entrenamiento general ====
        # Número de iteraciones
        self.num_iterations = 15

        # ==== Self-Play ====
        # Partidas por iteración
        self.num_selfplay_games = 600
        # Número de procesos paralelos
        self.num_selfplay_workers = 4
        # Número de partidas que un worker juega a la vez
        self.simultaneous_games_per_worker = 50
        # Partidas que juga cada worker
        self.games_for_worker = self.num_selfplay_games // self.num_selfplay_workers
        # Temperatura para elegir jugadas en el selfplay
        self.selfplay_temperature = 1.25
        # Determina hasta qué turno se usa la temperatura
        self.temperature_threshold = 10

        # ==== Entrenamiento de la red neuronal ====
        # Batch Size
        self.batch_size = 128
        # Learning Rate
        self.learning_rate = 0.001
        # Epochs por iteración
        self.num_epochs = 5
        # Weight Decay
        self.weight_decay = 0.0001

        # ==== MCTS ====
        # Cantidad de simulaciones por movimiento durante la búsqueda en el mcts
        self.num_mcts_simulations = 300
        # Influye en el equilibrio entre explorar nuevos nodos y explotar los conocidos
        self.C = 1.5
        # Define la dispersión del ruido añadido en la raíz del mcts
        self.dirichlet_alpha = 0.3
        # Controla cuánto se mezcla el ruido con las probabilidades originales en el nodo raíz del mcts
        self.exploration_fraction = 0.25

        # ==== Red neuronal ====
        # Número de bloques residuales usados en la red
        self.num_residual_blocks = 8
        # Número de filtros en cada capa convolucional
        self.num_filters = 128

        # ==== Evaluación de modelos ====
        # Número de partidas jugadas por cada modelo
        self.num_test_games = 200
        # Número de procesos paralelos en evaluación
        self.num_test_workers = 4
        # Número de partidas jugadas por cada proceso en evalución
        self.test_games_per_worker = 25
        # Número de simulaciones de Monte Carlo en evaluación
        self.test_num_simulations = 80
        # Temperatura en evaluación
        self.test_temperature = 0.2
        # Umbral mínimo de victorias requerido para aceptar un nuevo modelo
        self.test_win_rate_threshold = 0.55
