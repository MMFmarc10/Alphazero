import numpy as np
from games.BaseGame import BaseGame

class Go5x5(BaseGame):

    ROWS = 5
    COLS = 5
    SIZE = 5
    ACTION_SIZE = SIZE * SIZE + 1

    def __init__(self):
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=int)
        self.player = 1
        self.result = None
        self.passes = 0
        self.komi = 5.5
        self.last_board_hash = self.hash_board()

    def legal_moves(self):
        moves = []
        for i in range(self.SIZE * self.SIZE):
            row, col = divmod(i, self.SIZE)

            if self.board[row, col] != 0 or self.is_suicide(row, col):
                continue

            # Crear una instancia mínima con tablero simulado
            temp_game = Go5x5()
            temp_game.board = self.board.copy()
            temp_game.player = self.player
            temp_game.board[row, col] = self.player
            temp_game.remove_captured_stones(-self.player)
            temp_game.player *= -1

            if temp_game.hash_board() != self.last_board_hash:
                moves.append(i)

        moves.append(self.ACTION_SIZE - 1)  # Pasar
        return moves

    def legal_moves_mask(self):
        mask = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        for move in self.legal_moves():
            mask[move] = 1.0
        return mask

    def make_move(self, action):
        if action not in self.legal_moves():
            print("Movimiento ilegal")
            return False

        self.last_board_hash = self.hash_board()

        if action == self.ACTION_SIZE - 1:  # Pasar
            self.passes += 1
            self.player *= -1
            return True

        row, col = divmod(action, self.SIZE)
        self.passes = 0
        self.board[row, col] = self.player
        self.remove_captured_stones(-self.player)
        self.player *= -1

        return True


    def is_game_over(self):
        if self.passes >= 2:
            self.result = self.get_result_score()
            return True
        return False


    def get_game_result(self):
        if self.result is None:
            raise Exception("Game not finished")
        return self.result

    def get_action_size(self):
        return self.ACTION_SIZE

    def encode_board(self):
        if self.player == 1:
            p_layer = (self.board == 1).astype(int)
            o_layer = (self.board == -1).astype(int)
        else:
            p_layer = (self.board == -1).astype(int)
            o_layer = (self.board == 1).astype(int)
        empty = (self.board == 0).astype(int)
        return np.stack([p_layer, empty, o_layer], axis=0)

    def get_opposite_player(self):
        return -self.player



    def print_board(self):
        symbol_map = {0: '.', 1: 'X', -1: 'O'}
        print("  " + " ".join(str(i) for i in range(self.SIZE)))
        for r in range(self.SIZE):
            print(f"{r} " + " ".join(symbol_map[self.board[r][c]] for c in range(self.SIZE)))


    def hash_board(self):
        player_byte = 0 if self.player == -1 else 1
        return hash(self.board.tobytes() + bytes([player_byte]))

    def get_neighbors(self, r, c):
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                yield nr, nc

    def has_liberty(self, board, r, c, visited=None):
        if visited is None:
            visited = set()
        visited.add((r, c))
        for nr, nc in self.get_neighbors(r, c):
            if board[nr, nc] == 0:
                return True
            if board[nr, nc] == board[r, c] and (nr, nc) not in visited:
                if self.has_liberty(board, nr, nc, visited):
                    return True
        return False

    def remove_captured_stones(self, color):
        visited = set()
        to_remove = []
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if self.board[r, c] == color and (r, c) not in visited:
                    group = []
                    if not self.has_liberty_group(r, c, color, visited, group):
                        to_remove.extend(group)

        for r, c in to_remove:
            self.board[r, c] = 0


    def is_suicide(self, row, col):
        if self.board[row, col] != 0:
            return True
        temp_board = self.board.copy()
        temp_board[row, col] = self.player
        temp_game = Go5x5()
        temp_game.board = temp_board
        temp_game.player = self.player
        temp_game.remove_captured_stones(-self.player)
        return not temp_game.has_liberty(temp_board, row, col)

    def has_liberty_group(self, r, c, color, visited, group):
        stack = [(r, c)]
        has_liberty = False
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            group.append((x, y))
            for nx, ny in self.get_neighbors(x, y):
                if self.board[nx, ny] == 0:
                    has_liberty = True
                elif self.board[nx, ny] == color:
                    stack.append((nx, ny))
        return has_liberty


    def get_result_score(self):
        black_score = 0
        white_score = self.komi
        visited = set()

        for r in range(self.SIZE):
            for c in range(self.SIZE):
                val = self.board[r, c]
                if val == 1:
                    black_score += 1
                elif val == -1:
                    white_score += 1
                elif val == 0 and (r, c) not in visited:
                    territory, owner = self.flood_fill_territory(r, c, visited)
                    if owner == 1:
                        black_score += territory
                    elif owner == -1:
                        white_score += territory
                    # Neutral no suma a nadie

        if black_score > white_score:
            return 1
        elif white_score > black_score:
            return -1
        else:
            return 0

    def flood_fill_territory(self, r, c, visited):
        queue = [(r, c)]
        visited.add((r, c))
        territory = [(r, c)]
        owners = set()

        while queue:
            x, y = queue.pop()
            for nx, ny in self.get_neighbors(x, y):
                if self.board[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    territory.append((nx, ny))
                elif self.board[nx, ny] != 0:
                    owners.add(self.board[nx, ny])

        if len(owners) == 1:
            return len(territory), owners.pop()
        else:
            return 0, 0  # territorio neutral


