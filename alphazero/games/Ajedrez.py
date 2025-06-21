import chess
import numpy as np

from games.BaseGame import BaseGame


class Ajedrez(BaseGame):

    ACTION_SIZE = 64*64

    def __init__(self):
        self.board = chess.Board()
        self.result = None

    def legal_moves(self):
        return [
            move.uci()[:4]
            for move in self.board.legal_moves
            if not move.promotion or move.promotion == chess.QUEEN
        ]

    def legal_moves_mask(self):

        mask = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        for uci_move in self.legal_moves():
            idx = self.move_to_index(uci_move)
            mask[idx] = 1.0
        return mask

    def make_move(self, index: int) -> bool:
        from_square = index // 64
        to_square = index % 64


        if self.board.turn == chess.BLACK:
            from_square = self.flip_vertical(from_square)
            to_square = self.flip_vertical(to_square)

        move = chess.Move(from_square, to_square)
        print(move.uci())

        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        else:
            return False

    def make_move_uci(self, move_uci: str):
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
            return False
        except:
            return False

    def is_game_over(self):
        if self.board.is_game_over():
            result = self.board.result()
            if result == '1-0':
                self.result = 1
            elif result == '0-1':
                self.result = -1
            else:
                self.result = 0
            return True
        return False

    def get_game_result(self):
        if self.result is None:
            raise Exception("La partida no ha terminado todavía")
        return self.result

    def get_action_space(self):
        return self.ACTION_SIZE

    def encode_board(self):
        pass

    def get_opposite_player(self):
        return 1 if self.board.turn == chess.WHITE else -1

    def print_board(self):
        print(self.board)

    def flip_vertical(self,square: int) -> int:
        fila = square // 8
        col = square % 8
        return (7 - fila) * 8 + col

    def move_to_index(self, uci_move):
        move = chess.Move.from_uci(uci_move)

        if self.board.turn == chess.WHITE:
            return move.from_square * 64 + move.to_square
        else:
            from_sq = self.flip_vertical(move.from_square)
            to_sq = self.flip_vertical(move.to_square)
            return from_sq * 64 + to_sq


b = chess.Board()
a = Ajedrez()
print(a.move_to_index("e2e4"))
a.make_move(a.move_to_index("e2e4"))
a.print_board()

print(a.move_to_index("e7e5"))
a.make_move(a.move_to_index("e7e5"))
a.print_board()


print(a.move_to_index("d1g4"))
a.make_move(a.move_to_index("d1g4"))
a.print_board()

print(a.move_to_index("d8g5"))
a.make_move(a.move_to_index("d8g5"))
a.print_board()








