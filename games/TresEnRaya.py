import numpy as np

from games.BaseGame import BaseGame

# Implementación del juego Tres en Raya compatible con la interfaz AlphaZero (BaseGame).
class TresEnRaya(BaseGame):

    ROWS = 3
    COLS = 3
    ACTION_SIZE = ROWS * COLS

    def __init__(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.player = 1
        self.result = None
        
    def legal_moves(self):
        return [r * self.COLS + c for r in range(self.ROWS) for c in range(self.COLS) if self.board[r][c] == 0]

    def legal_moves_mask(self):
        return (self.board.flatten() == 0).astype(np.float32)

    def make_move(self, action):
        row, col = divmod(action, self.COLS)
        if self.board[row][col] == 0:
            self.board[row][col] = self.player
            self.player *= -1
            return True
        return False

    def check_win(self) -> bool:
        b = self.board

        for player in [1, -1]:
            # Filas y columnas
            for i in range(3):
                if np.all(b[i, :] == player) or np.all(b[:, i] == player):
                    return True

            # Diagonales
            if np.all(np.diag(b) == player) or np.all(np.diag(np.fliplr(b)) == player):
                return True

        return False

    def is_game_over(self) -> bool:
        if self.check_win():
            self.result = self.get_opposite_player()
            return True

        if len(self.legal_moves()) == 0:
            self.result = 0  # empate
            return True

        return False

    def get_game_result(self) -> int:
        if self.result is None:
            raise Exception("Game is not finished")
        return self.result

    def get_action_size(self):
        return self.ACTION_SIZE

    def encode_board(self):
        if self.player == 1:
            x_layer = (self.board == 1).astype(int)
            o_layer = (self.board == -1).astype(int)
        else:
            x_layer = (self.board == -1).astype(int)
            o_layer = (self.board == 1).astype(int)
        empty_layer = (self.board == 0).astype(int)
        return np.stack([x_layer, empty_layer, o_layer], axis=0)


    def get_opposite_player(self):
        return -self.player

    def print_board(self):
        symbol_map = {0: '.', 1: 'X', -1: 'O'}
        for row in range(self.ROWS):
            print("  " + " ".join(symbol_map[int(self.board[row][col])] for col in range(self.COLS)))

