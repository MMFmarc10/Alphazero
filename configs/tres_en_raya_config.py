# Configuración utilizada para aprender el juego TresEnRaya
class TresEnRayaConfig:

    def __init__(self):
        # Entrenamiento general
        self.num_iterations = 10

        # SelfPlay
        self.num_selfplay_games = 400
        self.num_selfplay_workers = 5
        self.simultaneous_games_per_worker = 40
        self.games_for_worker = self.num_selfplay_games // self.num_selfplay_workers
        self.selfplay_temperature = 1.25
        self.temperature_threshold = 3

        # Entrenamiento de red neuronal
        self.batch_size = 64
        self.learning_rate = 0.001
        self.num_epochs = 4
        self.weight_decay = 0.0001

        # MCTS
        self.num_mcts_simulations = 80
        self.C = 1.5
        self.dirichlet_alpha = 0.3
        self.exploration_fraction = 0.25

        # Model
        self.num_residual_blocks = 4
        self.num_filters = 64

        # Evaluation
        self.num_test_games = 100
        self.num_test_workers = 4
        self.test_games_per_worker = 25
        self.test_num_simulations = 50
        self.test_temperature = 0.1
        self.test_win_rate_threshold = 0.55