import numpy as np

from games.BaseGame import BaseGame

class Go5x5(BaseGame):


    SIZE = 5
    ROWS = SIZE
    COLS = SIZE
    MAX_TURNS = SIZE*SIZE * 2
    ACTION_SIZE = SIZE * SIZE + 1

    neighbor_table = {}  # Variable de clase

    def __init__(self):

        self.board = np.zeros((self.SIZE, self.SIZE), dtype=np.int8)

        self.group_id_board = np.full((self.SIZE, self.SIZE), -1, dtype=np.int16)

        self.group_liberties = np.zeros((self.MAX_TURNS, self.SIZE, self.SIZE), dtype=bool)

        self.liberties_per_group = np.zeros(self.MAX_TURNS, dtype = np.int8)

        self.ko_position = None

        self.black_stones = 0
        self.white_stones = 0

        self.next_id = 0
        self.history = []

        self.player = 1  # 1 = negro, -1 = blanco
        self.result = None
        self.passes = 0
        self.komi = 2.5
        self.turns = 1

        if not Go5x5.neighbor_table:
            Go5x5.neighbor_table = self._build_neighbor_table()

    def legal_moves(self):
        mask = self.legal_moves_mask()  # ← Reutilizas el cálculo eficiente
        # Los movimientos legales son las posiciones donde mask es 1.0
        return [i for i, v in enumerate(mask) if v == 1.0]

    def legal_moves_mask(self):
        mask = np.zeros(self.ACTION_SIZE, dtype=np.float32)

        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if self.board[r, c] != 0:
                    continue  # Ocupado

                if self.ko_position == (r, c):
                    continue  # Ko

                if self.is_sacrifice(r, c):
                    continue  # Suicidio

                mask[r * self.SIZE + c] = 1.0

        mask[self.ACTION_SIZE - 1] = 1.0  # Acción de "pasar"
        return mask

    def make_move(self, action):

        move_deltas = []

        if action == self.ACTION_SIZE - 1:  # Pasar

            move_deltas.append(("pass",0,self.passes))
            self.passes += 1
            self.player *= -1
            self.turns += 1

            self.history.append((action,move_deltas))


            return True

        if self.player == 1:
            move_deltas.append(("black_stone", 0, -1))
            self.black_stones +=1
        else:

            move_deltas.append(("white_stone", 0, -1))
            self.white_stones += 1

        row, col = divmod(action, self.SIZE)
        move_deltas.append(("board", (row,col), 0))
        self.board[row, col] = self.player

        new_group_id = self.next_id
        self.next_id +=1
        move_deltas.append(("id", 0, -1))

        move_deltas.append(("group_id_board", (row, col),  self.group_id_board[row][col]))
        self.group_id_board[row][col] = new_group_id

        new_group_liberties  = np.zeros((self.SIZE, self.SIZE), dtype=bool)
        self.group_liberties[new_group_id] = new_group_liberties

        neighbors= self.get_neighbors(row, col)

        gids_amigos = set()
        gids_enemigos = set()
        all_captured = []

        # 1. Recorres vecinos y clasificas
        for nr, nc in neighbors:
            neighbor = self.board[nr, nc]
            if neighbor == 0:
                self.group_liberties[new_group_id][nr][nc] = True
                self.liberties_per_group[new_group_id] += 1
            elif neighbor == self.player:
                gid = self.group_id_board[nr, nc]
                gids_amigos.add(gid)
            else:
                gid = self.group_id_board[nr, nc]
                gids_enemigos.add(gid)

        # 2. Procesas cada grupo enemigo solo una vez
        for gid in gids_enemigos:
            move_deltas.append(("liberty", (gid, row, col), 1))
            self.group_liberties[gid, row, col] = False
            self.liberties_per_group[gid] -= 1

            if self.liberties_per_group[gid] == 0:
                captured = self.eliminar_grupo(gid, move_deltas)
                all_captured.extend(captured)

        self.juntar_grupos(new_group_id, gids_amigos, row, col, move_deltas)

        move_deltas.append(("ko", 0, self.ko_position))  # Guarda el Ko anterior por si haces undo
        if len(all_captured) == 1 and len(gids_amigos) == 0:

            self.ko_position = all_captured[0]
        else:
            self.ko_position = None

        move_deltas.append(("pass", 0, self.passes))
        self.passes = 0
        self.player *= -1
        self.turns += 1

        self.history.append((action,move_deltas))


        return True

    def undo_move(self):
        action,move_deltas = self.history.pop()

        for tipo, pos, old_value in reversed(move_deltas):
            if tipo == "board":
                r, c = pos
                self.board[r, c] = old_value

            elif tipo == "group_id_board":
                r, c = pos
                self.group_id_board[r, c] = old_value

            elif tipo == "liberty":
                gid, r, c = pos

                if old_value == 1:
                    self.group_liberties[gid,r,c] = True
                else:
                    self.group_liberties[gid, r, c] = False

                self.liberties_per_group[gid] += old_value

            elif tipo == "id":

               self.next_id += old_value

            elif tipo == "pass":
                self.passes= old_value

            elif tipo == "white_stone":
                self.white_stones += old_value

            elif tipo == "black_stone":
                self.black_stones += old_value
            elif tipo == "ko":
                self.ko_position = old_value

        self.player *= -1
        self.turns -= 1
        return action




    def juntar_grupos(self, new_gid, gids,row,col,move_deltas):
        # Supongamos que tienes los arrays de libertades de dos grupos

        for gid in gids:
        # Combinas las libertades con un OR lógico (elemento a elemento)
            self.group_liberties[new_gid] |= self.group_liberties[gid]  # In-plac

            for r, c in np.argwhere(self.group_id_board == gid):
                move_deltas.append(("group_id_board", (r, c), gid))  # Sabes que el valor previo es gid
                self.group_id_board[r, c] = new_gid

        self.group_liberties[new_gid, row, col] = False
        self.liberties_per_group[new_gid] = np.sum(self.group_liberties[new_gid])




    def eliminar_grupo(self, gid,move_deltas):
        # Encuentra las coordenadas del grupo capturado
        stones = np.argwhere(self.group_id_board == gid)
        captured_positions = []
        # Elimina las piedras del tablero y del group_id_board
        for r, c in stones:
            captured_positions.append((r, c))
            move_deltas.append(("board", (r, c), self.board[r, c]))
            self.board[r, c] = 0
            move_deltas.append(("group_id_board", (r, c), self.group_id_board[r, c]))
            self.group_id_board[r, c] = -1
            if self.player == 1:
                move_deltas.append(("white_stone", 0, 1))
                self.white_stones -= 1
            else:
                move_deltas.append(("black_stone", 0, 1))
                self.black_stones -= 1


            # Añade esta posición como libertad a los grupos vecinos
            for nr, nc in self.get_neighbors(r, c):
                if  self.board[nr, nc] == self.player:
                    neighbor_gid = self.group_id_board[nr, nc]
                    move_deltas.append(("liberty", (neighbor_gid, r, c), -1))
                    self.group_liberties[neighbor_gid, r, c] = True
                    self.liberties_per_group[neighbor_gid] +=1

        return captured_positions





    def is_game_over(self):
        # Condición 1: ambos jugadores han pasado
        if self.passes >= 2:
            self.result = self.get_result_score()


            return True

        # Condición 2: se alcanzó el número máximo de turnos
        if self.turns >= self.MAX_TURNS:
            self.result = self.get_result_score()


            return True


        if self.turns > self.ACTION_SIZE:

            black_score = self.black_stones
            white_score = self.white_stones + self.komi//2

            if black_score >= 2 * white_score + 1:
                self.result = 1

                return True

            if white_score >= 2 * black_score + 1:
                self.result = -1

                return True

        return False

    def get_game_result(self):
        if self.result is None:
            raise Exception("Game not finished")
        return self.result

    def get_action_size(self):
        return self.ACTION_SIZE

    def encode_board1(self):
        if self.player == 1:
            p_layer = (self.board == 1).astype(np.uint8)
            o_layer = (self.board == -1).astype(np.uint8)
        else:
            p_layer = (self.board == -1).astype(np.uint8)
            o_layer = (self.board == 1).astype(np.uint8)

        empty = (self.board == 0).astype(np.uint8)
        return np.stack([p_layer, empty, o_layer], axis=0)  # dtype = np.uint8

    def encode_board(self, history_length=2):
        """
        Devuelve una representación [2*(1+history_length) + 2, SIZE, SIZE]:
        [player_now, opponent_now, player_t-1, opponent_t-1, ..., empty, passed]
        """
        layers = []

        actual_player = self.player
        # Estado actual
        if actual_player == 1:
            player_now = (self.board == 1).astype(np.uint8)
            opponent_now = (self.board == -1).astype(np.uint8)
        else:
            player_now = (self.board == -1).astype(np.uint8)
            opponent_now = (self.board == 1).astype(np.uint8)

        layers.append(player_now)
        layers.append(opponent_now)

        # Historia deshecha
        undone_actions = []
        for _ in range(history_length):
            if not self.history:
                layers.append(np.zeros((self.SIZE, self.SIZE), dtype=np.uint8))  # player
                layers.append(np.zeros((self.SIZE, self.SIZE), dtype=np.uint8))  # opponent
            else:
                action = self.undo_move()
                undone_actions.append(action)

                if actual_player == 1:
                    player_layer = (self.board == 1).astype(np.uint8)
                    opponent_layer = (self.board == -1).astype(np.uint8)
                else:
                    player_layer = (self.board == -1).astype(np.uint8)
                    opponent_layer = (self.board == 1).astype(np.uint8)

                layers.append(player_layer)
                layers.append(opponent_layer)

        # Restaurar estado original
        for action in reversed(undone_actions):
            self.make_move(action)

        # Capa de vacías
        empty_layer = (self.board == 0).astype(np.uint8)
        layers.append(empty_layer)

        # Capa de si el oponente pasó
        pass_layer = np.full((self.SIZE, self.SIZE), np.uint8(self.passes >= 1), dtype=np.uint8)
        layers.append(pass_layer)

        return np.stack(layers, axis=0)

    def get_opposite_player(self):
        return -self.player


    def print_board(self):
        symbol_map = {0: '.', 1: 'X', -1: 'O'}
        print("  " + " ".join(str(i) for i in range(self.SIZE)))
        for r in range(self.SIZE):
            print(f"{r} " + " ".join(symbol_map[self.board[r][c]] for c in range(self.SIZE)))


    def get_neighbors(self, r, c):
        return Go5x5.neighbor_table[(r, c)]


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

    def _build_neighbor_table(self):
        table = {}
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.SIZE and 0 <= nc < self.SIZE:
                        neighbors.append((nr, nc))
                table[(r, c)] = neighbors
        return table

    def print_groups(self):
        print("\nGrupos del jugador negro (X):")
        for gid, group in self.black_groups.items():
            print(f"  ID {gid}:")
            print(f"    Piedras: {sorted(group.stones)}")
            print(f"    Libertades: {sorted(group.liberties)}")

        print("\nGrupos del jugador blanco (O):")
        for gid, group in self.white_groups.items():
            print(f"  ID {gid}:")
            print(f"    Piedras: {sorted(group.stones)}")
            print(f"    Libertades: {sorted(group.liberties)}")

    def is_sacrifice(self, r, c):
        neighbors = self.get_neighbors(r, c)

        # 1. ¿Tiene libertades directas?
        for nr, nc in neighbors:
            if self.board[nr, nc] == 0:
                return False  # No es suicidio: hay libertades

        # 2. ¿Captura algún grupo enemigo sin libertades?
        for nr, nc in neighbors:
            if self.board[nr, nc] == -self.player:
                gid = self.group_id_board[nr, nc]
                if self.liberties_per_group[gid]==1:
                    return False  # No es suicidio: captura al enemigo


        # 3. ¿Se une a un grupo propio con otras libertades?
        for nr, nc in neighbors:
            if self.board[nr, nc] == self.player:
                gid = self.group_id_board[nr, nc]

                if self.liberties_per_group[gid]>1:

                    return False  # No es suicidio: captura al enemigo


        # Si no tiene libertades, no captura, y no se une a grupo libre → suicidio
        return True

    def print_estado(self):
        print("\n========== ESTADO DEL JUEGO ==========")

        # Tablero
        print("Tablero (1 = X, -1 = O, 0 = vacío):")
        self.print_board()

        # Contadores de piedras
        print(f"\nPiedras negras (X): {self.black_stones}")
        print(f"Piedras blancas (O): {self.white_stones}")

        # Group ID Board
        print("\nGroup ID Board:")
        for r in range(self.SIZE):
            print(" ".join(f"{self.group_id_board[r, c]:2}" for c in range(self.SIZE)))

        # Libertades por grupo
        print("\nLibertades por grupo:")
        for gid in range(self.MAX_TURNS):
            if np.any(self.group_id_board == gid):
                print(f"  Grupo {gid}:")
                liberties = self.group_liberties[gid]
                coords = np.argwhere(liberties)
                print(f"    Libertades en {[(int(r), int(c)) for r, c in coords]}")
                print(f"    Total libertades: {self.liberties_per_group[gid]}")

        # Jugador actual y estado general
        print("\nJugador actual:", "Negro (X)" if self.player == 1 else "Blanco (O)")
        print("Turno:", self.turns)
        print("Pases consecutivos:", self.passes)
        print("======================================\n")






