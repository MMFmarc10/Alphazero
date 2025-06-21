
import numpy as np

from games.BaseGame import BaseGame


class CuatroEnRaya(BaseGame):

    ROWS = 6
    COLS = 7
    ACTION_SIZE = COLS
    
    def __init__(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.player = 1
        self.result= None

    def legal_moves(self):
         return [col for col in range(self.COLS) if self.board[0][col] == 0]
    
    def legal_moves_mask(self):
          return (self.board[0] == 0).astype(np.float32)

    def make_move(self,column):
        if self.board[0][column]== 0 and column< self.COLS:

            for row in range(self.ROWS-1, -1, -1):

                if self.board[row][column] == 0:
                     self.board[row][column] = self.player
                     break
                    
            self.player = self.player *-1

            return True
        else:
            print("movimiento equivocado")
            return False

    def check_win(self):
        b = self.board

        # Horizontal
        for row in range(self.ROWS):
            for col in range(self.COLS - 3):
                window = b[row, col:col + 4]
                if abs(np.sum(window)) == 4 and len(set(window)) == 1:
                    return True

        # Vertical
        for row in range(self.ROWS - 3):
            for col in range(self.COLS):
                window = b[row:row + 4, col]
                if abs(np.sum(window)) == 4 and len(set(window)) == 1:
                    return True

        # Diagonal descendente (\)
        for row in range(self.ROWS - 3):
            for col in range(self.COLS - 3):
                window = [b[row + i, col + i] for i in range(4)]
                if abs(sum(window)) == 4 and len(set(window)) == 1:
                    return True

        # Diagonal ascendente (/)
        for row in range(3, self.ROWS):
            for col in range(self.COLS - 3):
                window = [b[row - i, col + i] for i in range(4)]
                if abs(sum(window)) == 4 and len(set(window)) == 1:
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

    def get_action_space(self):
        return self.ROWS * self.COLS

    def encode_board(self):
        """
        Convierte el tablero (np.array de forma [ROWS, COLS]) 
        en un tensor de forma [2, ROWS, COLS] para entrada a la red.
        """
        if self.player == 1: 
            x_layer = (self.board == 1).astype(int)
            o_layer = (self.board == -1).astype(int)
        else:
            x_layer = (self.board == -1).astype(int)
            o_layer = (self.board == 1).astype(int)
    
        empty_layer = (self.board == 0).astype(int)

        return np.stack([x_layer, empty_layer, o_layer], axis=0)


    def get_opposite_player(self):
        return self.player*-1
        

    def print_board(self):
        symbol_map = {0: '.', 1: 'X', -1: 'O'}

        print("  " + " ".join(str(col) for col in range(self.COLS)))
        for row in range(self.ROWS):
            print("  " + " ".join(symbol_map[self.board[row][col]] for col in range(self.COLS)))

