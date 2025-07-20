# Configuración utilizada para aprender el juego TresEnRaya
class Go5x5Config:

    def __init__(self):
        # Entrenamiento general
        self.num_iterations = 40

        # SelfPlay
        self.num_selfplay_games = 600
        self.num_selfplay_workers = 5
        self.simultaneous_games_per_worker = 120
        self.games_for_worker = self.num_selfplay_games // self.num_selfplay_workers
        self.selfplay_temperature = 1.25
        self.temperature_threshold = 10

        # Entrenamiento de red neuronal
        self.batch_size = 128
        self.learning_rate = 0.001
        self.num_epochs = 5
        self.weight_decay = 0.0001

        # MCTS
        self.num_mcts_simulations = 350
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
        self.test_num_simulations = 300
        self.test_temperature_before = 0.8
        self.test_temperature_threshold = 6
        self.test_temperature_after = 0.1
        self.test_win_rate_threshold = 0.55
        self.evaluation_frequency = 2

        # Replay Buffer
        self.max_size = 150000
        self.train_sample = 35000