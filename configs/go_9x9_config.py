# Configuración utilizada para aprender el juego TresEnRaya
class Go9x9Config:

    def __init__(self):
        # Entrenamiento general
        self.num_iterations = 60  # más iteraciones porque el espacio de estados es mayor

        # SelfPlay
        self.num_selfplay_games = 720  # menos partidas por iteración, pero más largas
        self.num_selfplay_workers = 6
        self.simultaneous_games_per_worker = 40  # menos juegos simultáneos, más memoria por juego
        self.games_for_worker = self.num_selfplay_games // self.num_selfplay_workers
        self.selfplay_temperature = 1.25
        self.temperature_threshold = 20

        # Entrenamiento de red neuronal
        self.batch_size = 128
        self.learning_rate = 0.001
        self.num_epochs = 5
        self.weight_decay = 0.0001

        # MCTS
        self.num_mcts_simulations = 500  # más simulaciones para decisiones más sólidas
        self.C = 1.5
        self.dirichlet_alpha = 0.15  # menor alpha porque hay más acciones posibles
        self.exploration_fraction = 0.25

        # Model
        self.num_residual_blocks = 10
        self.num_filters = 128

        # Evaluation
        self.num_test_games = 80
        self.num_test_workers = 4
        self.test_games_per_worker = 20
        self.test_num_simulations = 500
        self.test_temperature_before = 0.8
        self.test_temperature_threshold = 10
        self.test_temperature_after = 0.1
        self.test_win_rate_threshold = 0.55
        self.evaluation_frequency = 2

        # Replay Buffer
        self.max_size = 350000  # más estados únicos por partida
        self.train_sample  = 100000