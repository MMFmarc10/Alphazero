# Configuración utilizada para aprender el juego TresEnRaya
class Go5x5ConfigTest:

    def __init__(self):
        # Entrenamiento general
        self.num_iterations = 10

        # SelfPlay
        self.num_selfplay_games = 1
        self.num_selfplay_workers = 4
        self.simultaneous_games_per_worker = 20
        self.games_for_worker = self.num_selfplay_games // self.num_selfplay_workers
        self.selfplay_temperature = 1.25
        self.temperature_threshold = 12

        # Entrenamiento de red neuronal
        self.batch_size = 128
        self.learning_rate = 0.001
        self.num_epochs = 5
        self.weight_decay = 0.0001

        # MCTS
        self.num_mcts_simulations = 50
        self.C = 1.5
        self.dirichlet_alpha = 0.3
        self.exploration_fraction = 0.25

        # Model
        self.num_residual_blocks = 8
        self.num_filters = 128

        # Evaluation
        self.num_test_games = 100
        self.num_test_workers = 4
        self.test_games_per_worker = 25
        self.test_num_simulations = 150
        self.test_temperature = 0.2
        self.test_win_rate_threshold = 0.55