from enum import Enum

import numpy as np

from games.BaseGame import BaseGame


class Damas(BaseGame):

    ROWS = 8
    COLS = 8
    ACTION_SIZE = ROWS * COLS * ROWS * COLS  # (from_row, from_col, to_row, to_col)

    class Piece(Enum):
        RED_MAN = 1
        RED_KING = 2
        BLACK_MAN = -1
        BLACK_KING = -2

    def __init__(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.player = 1
        self.result = None
        self._initialize_board()

    def _initialize_board(self):
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 != 0:
                    if row < 3:
                        self.board[row][col] = Damas.Piece.RED_MAN.value
                    elif row > 4:
                        self.board[row][col] = Damas.Piece.BLACK_MAN.value

    def legal_moves(self):
       pass

    def legal_moves_mask(self):
       pass

    def make_move(self, action):
       pass

    def is_game_over(self):
       pass

    def get_game_result(self):
       pass

    def get_action_space(self):
        return self.ACTION_SIZE

    def encode_board(self):
        pass

    def get_opposite_player(self):
        return -self.player

    def print_board(self):
        pass


