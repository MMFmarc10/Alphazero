import numpy as np
from games.BaseGame import BaseGame


class Group:
    def __init__(self, id, color):
        self.id = id
        self.color = color  # 1 para negro, -1 para blanco
        self.stones = set()
        self.liberties = set()

    def add_stone(self, r, c):
        self.stones.add((r, c))

    def add_liberty(self, r, c):
        self.liberties.add((r, c))

    def remove_liberty(self, r, c):
        self.liberties.discard((r, c))

    def merge_with(self, other):
        self.stones |= other.stones
        self.liberties |= other.liberties
        # Las libertades no pueden incluir posiciones ocupadas por piedras del grupo
        self.liberties -= self.stones

    def clone(self):
        nuevo = Group(self.id, self.color)
        nuevo.stones = self.stones.copy()
        nuevo.liberties = self.liberties.copy()
        return nuevo



class Go5x5Faster(BaseGame):

    ROWS = 5
    COLS = 5
    SIZE = 5
    MAX_TURNS = 80
    ACTION_SIZE = SIZE * SIZE + 1

    neighbor_table = {}  # Variable de clase

    def __init__(self):
        self.board = np.zeros((self.SIZE, self.SIZE), dtype=int)
        self.player = 1  # 1 = negro, -1 = blanco
        self.result = None
        self.passes = 0
        self.komi = 1.5
        self.turns = 0
        self.last_board_hash = self.hash_board()

        self.history = []

        self.group_id_board = np.full((self.SIZE, self.SIZE), -1, dtype=int)

        # Diccionarios separados para grupos de cada color
        self.black_groups = {}  # group_id -> Group
        self.white_groups = {}  # group_id -> Group

        self.next_group_id = 0

        if not Go5x5Faster.neighbor_table:
            Go5x5Faster.neighbor_table = self._build_neighbor_table()

    def get_group_dict(self, color):
        return self.black_groups if color == 1 else self.white_groups

    def legal_moves(self):
        moves = []
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if self.board[r, c] != 0:
                    continue  # Ocupado

                if self.is_sacrifice(r, c):
                    continue

                if self.ko_rule(r,c):
                    continue

                moves.append(r * self.SIZE + c)

        moves.append(self.ACTION_SIZE - 1)  # Acción de "pasar"
        return moves

    def legal_moves_mask(self):
        mask = np.zeros(self.ACTION_SIZE, dtype=np.float32)
        for move in self.legal_moves():
            mask[move] = 1.0
        return mask

    def make_move(self, action):
        # Guardar el hash anterior para regla de ko
        self.last_board_hash = self.hash_board()

        undo_entry = {
            'player': self.player,
            'action': action,
            'last_board_hash': self.last_board_hash,
            'passes': self.passes,  # 👈 AÑADIR ESTA LÍNEA
            'captured': [],
            'merged_groups': [],
            'created_group_id': self.next_group_id,
            'group_id_board_snapshot': {},
        }

        if action == self.ACTION_SIZE - 1:  # Pasar
            self.passes += 1
            self.player *= -1
            self.turns += 1
            self.history.append(undo_entry)
            return True

        row, col = divmod(action, self.SIZE)
        if self.board[row, col] != 0:
            print(action)
            print(self.legal_moves())
            print(self.legal_moves_mask())
            print(self.board)
            print(self.group_id_board)
            raise RuntimeError(
                f"❌ Movimiento ilegal: la casilla ({row},{col}) ya está ocupada por {self.board[row, col]}")

        self.passes = 0
        self.board[row, col] = self.player

        new_group = Group(self.next_group_id, self.player)
        new_group.add_stone(row, col)
        group_ids_to_merge = set()

        neighbors = list(self.get_neighbors(row, col))

        for nr, nc in neighbors:
            if self.board[nr, nc] == 0:
                new_group.add_liberty(nr, nc)

        for nr, nc in neighbors:
            if self.board[nr, nc] == self.player:
                gid = self.group_id_board[nr, nc]
                if gid == -1 or gid in group_ids_to_merge:
                    continue
                group_ids_to_merge.add(gid)
                group_dict = self.get_group_dict(self.player)
                if gid not in group_dict:
                    print("\n❌ PROBLEMA DETECTADO: gid no encontrado en group_dict")
                    print(f"Jugador actual: {self.player}")
                    print(f"gid que falta: {gid}")
                    print("Estado del tablero:")
                    print(self.board)
                    print("group_id_board:")
                    print(self.group_id_board)
                    print("IDs presentes en group_dict:")
                    print(list(group_dict.keys()))

                    print("\n📜 Historial completo:")
                    for i, entry in enumerate(self.history):
                        print(f"\n--- Movimiento #{i + 1} ---")
                        print(f"  Jugador: {entry['player']}")
                        print(f"  Acción: {entry['action']}")
                        print(f"  Capturados: {[g.id for g in entry['captured']]}")
                        print(f"  Grupos fusionados: {[gid for gid, _ in entry['merged_groups']]}")
                        print(f"  Grupo creado: {entry['created_group_id']}")
                        print(f"  Snapshot group_id_board: {list(entry['group_id_board_snapshot'].items())}")

                    print("\nGrupos actuales (black_groups):", list(self.black_groups.keys()))
                    print("Grupos actuales (white_groups):", list(self.white_groups.keys()))
                    print()
                    continue  # O raise si quieres detener aquí

                other_group = group_dict.pop(gid)
                undo_entry['merged_groups'].append((gid, other_group.clone()))
                new_group.merge_with(other_group)

        for r, c in new_group.stones:
            undo_entry['group_id_board_snapshot'][(r, c)] = self.group_id_board[r, c]
            self.group_id_board[r, c] = new_group.id

        self.get_group_dict(self.player)[new_group.id] = new_group
        self.next_group_id += 1

        enemy_dict = self.get_group_dict(-self.player)
        captured_groups = set()
        self.undo_capture_list = []

        for nr, nc in neighbors:
            if self.board[nr, nc] == -self.player:
                gid = self.group_id_board[nr, nc]
                if gid in captured_groups or gid == -1:
                    continue
                group = enemy_dict.get(gid)
                if not group:
                    continue
                group.remove_liberty(row, col)
                if len(group.liberties) == 0:
                    self.capture_group(group)
                    self.undo_capture_list.append(group.clone())
                    captured_groups.add(gid)

        undo_entry['captured'] = self.undo_capture_list

        self.player *= -1
        self.turns += 1
        self.history.append(undo_entry)

        return True

    def undo_move(self):
        if not self.history:
            return False

        entry = self.history.pop()
        self.player = entry['player']
        self.last_board_hash = entry['last_board_hash']
        self.turns -= 1
        if entry['action'] == self.ACTION_SIZE - 1:
            self.passes = entry['passes']
            return True

        row, col = divmod(entry['action'], self.SIZE)
        self.board[row, col] = 0
        self.group_id_board[row, col] = -1

        # Restaurar grupos capturados
        for group in entry['captured']:
            group_dict = self.get_group_dict(group.color)
            group_dict[group.id] = group
            for r, c in group.stones:
                self.board[r, c] = group.color
                self.group_id_board[r, c] = group.id

        # Eliminar nuevo grupo creado
        created_id = entry['created_group_id']
        self.get_group_dict(self.player).pop(created_id, None)

        # Restaurar grupos fusionados
        for gid, group in entry['merged_groups']:
            self.get_group_dict(self.player)[gid] = group
            for r, c in group.stones:
                self.group_id_board[r, c] = gid

        # Restaurar celdas modificadas de group_id_board
        for (r, c), val in entry['group_id_board_snapshot'].items():
            self.group_id_board[r, c] = val

        self.next_group_id = entry['created_group_id']

        return True


    def is_game_over(self):
        # Condición 1: ambos jugadores han pasado
        if self.passes >= 2:
            self.result = self.get_result_score()
            return True

        # Condición 2: se alcanzó el número máximo de turnos
        if self.turns >= self.MAX_TURNS:
            self.result = self.get_result_score()
            return True

        # Condición 3 (opcional pero eficiente): el rival no tiene fichas en el tablero
        # Esto implica que ha sido completamente capturado
        if self.turns > 1 and not np.any(self.board == -self.player):
            self.result = self.player  # El jugador actual gana

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

    def clone(self):
        nuevo = Go5x5Faster()
        nuevo.board = self.board.copy()
        nuevo.player = self.player
        nuevo.turns = self.turns
        nuevo.passes = self.passes
        nuevo.komi = self.komi
        nuevo.last_board_hash = self.last_board_hash
        nuevo.group_id_board = self.group_id_board.copy()
        nuevo.next_group_id = self.next_group_id

        # Clonamos los grupos con .clone() optimizado
        nuevo.black_groups = {gid: group.clone() for gid, group in self.black_groups.items()}
        nuevo.white_groups = {gid: group.clone() for gid, group in self.white_groups.items()}

        return nuevo

    def print_board(self):
        symbol_map = {0: '.', 1: 'X', -1: 'O'}
        print("  " + " ".join(str(i) for i in range(self.SIZE)))
        for r in range(self.SIZE):
            print(f"{r} " + " ".join(symbol_map[self.board[r][c]] for c in range(self.SIZE)))


    def hash_board(self):
        player_byte = 0 if self.player == -1 else 1
        return hash(self.board.tobytes() + bytes([player_byte]))

    def get_neighbors(self, r, c):
        return Go5x5Faster.neighbor_table[(r, c)]

    def is_sacrifice(self, row, col):

        neighbors = list(self.get_neighbors(row, col))
        opponent = -self.player

        # Libertad directa
        for nr, nc in neighbors:
            if self.board[nr, nc] == 0:
                return False  # Libertad directa: no es suicidio

        # Captura posible
        enemy_groups = self.get_group_dict(opponent)
        for nr, nc in neighbors:
            if self.board[nr, nc] == opponent:
                group_id = self.group_id_board[nr, nc]
                if group_id == -1:
                    continue
                group = enemy_groups.get(group_id)
                if group and len(group.liberties) == 1 and (row, col) in group.liberties:
                    return False  # Captura válida: no es suicidio

        # Se une a grupo propio con libertades
        own_groups = self.get_group_dict(self.player)
        for nr, nc in neighbors:
            if self.board[nr, nc] == self.player:
                group_id = self.group_id_board[nr, nc]
                if group_id == -1:
                    continue
                group = own_groups.get(group_id)
                if group and len(group.liberties) > 1 and (row, col) in group.liberties:
                    return False

        return True  # No hay libertades, ni captura, ni grupo amigo → suicidio

    def ko_rule(self, row, col):

        temp_board = self.board.copy()
        temp_board[row, col] = self.player

        # Simular captura de enemigos
        for nr, nc in self.get_neighbors(row, col):
            if self.board[nr, nc] == -self.player:
                group_id = self.group_id_board[nr, nc]
                if group_id == -1:
                    continue
                group = self.get_group_dict(-self.player).get(group_id)
                if group and len(group.liberties) == 1 and (row, col) in group.liberties:
                    for r, c in group.stones:
                        temp_board[r, c] = 0

        # Calcular hash del tablero resultante + cambio de jugador
        next_player = -self.player
        temp_hash = hash_board_static(temp_board, next_player)

        return temp_hash == self.last_board_hash

    def capture_group(self, group):
        group_dict = self.get_group_dict(group.color)
        del group_dict[group.id]

        for r, c in group.stones:
            self.board[r, c] = 0
            self.group_id_board[r, c] = -1

            # Añadir la casilla como libertad a grupos adyacentes del jugador actual
            for nr, nc in self.get_neighbors(r, c):
                if self.board[nr, nc] == self.player:
                    gid = self.group_id_board[nr, nc]
                    if gid == -1:
                        continue
                    own_group = self.get_group_dict(self.player).get(gid)
                    if own_group:
                        own_group.add_liberty(r, c)

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


def hash_board_static(board, player):
    player_byte = 0 if player == -1 else 1
    return hash(board.tobytes() + bytes([player_byte]))



