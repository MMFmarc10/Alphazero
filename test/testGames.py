
from games.CuatroEnRayaFast import CuatroEnRayaFast
import numpy as np
import random

from test.Go5x5Faster import Go5x5Faster

def test_make_and_undo():
    game = CuatroEnRayaFast()

    # Guardar estado inicial
    initial_board = game.board.copy()
    initial_top = game.top.copy()
    initial_player = game.player

    moves = []

    print("hello")
    # Jugar hasta que el juego termine
    while not game.is_game_over():
        game.print_board()
        legal = game.legal_moves()
        move = random.choice(legal)
        game.make_move(move)
        moves.append(move)

    # Deshacer todos los movimientos
    for _ in range(len(moves)):

        game.undo_move()
        game.print_board()

    # Comprobar que el estado vuelve al original
    assert np.array_equal(game.board, initial_board), "❌ El tablero no coincide tras deshacer."
    assert np.array_equal(game.top, initial_top), "❌ `top` no coincide tras deshacer."
    assert game.player == initial_player, "❌ `player` no coincide tras deshacer."

    print("✅ Test passed: make_move y undo_move funcionan correctamente con movimientos aleatorios.")




def test_make_and_undo_go():
    game = Go5x5Faster()

    # Guardar estado inicial
    initial_board = game.board.copy()
    initial_player = game.player
    initial_turns = game.turns
    initial_passes = game.passes
    initial_next_gid = game.next_group_id
    initial_last_hash = game.last_board_hash
    initial_gid_board = game.group_id_board.copy()
    initial_black_groups = {gid: g.clone() for gid, g in game.black_groups.items()}
    initial_white_groups = {gid: g.clone() for gid, g in game.white_groups.items()}

    moves = []

    print("Hello (Go Test)")

    # Jugar hasta que el juego termine o un número razonable de turnos
    while not game.is_game_over():
        legal = game.legal_moves()
        move = random.choice(legal)
        game.make_move(move)
        moves.append(move)


    # Deshacer todos los movimientos
    for _ in range(len(moves)):
        game.undo_move()

    # Comprobaciones básicas
    assert np.array_equal(game.board, initial_board), "❌ El tablero no coincide tras deshacer."
    assert game.player == initial_player, "❌ `player` no coincide tras deshacer."
    assert game.turns == initial_turns, "❌ `turns` no coincide tras deshacer."
    assert game.passes == initial_passes, "❌ `passes` no coincide tras deshacer."
    assert game.next_group_id == initial_next_gid, "❌ `next_group_id` no coincide tras deshacer."
    assert game.last_board_hash == initial_last_hash, "❌ `last_board_hash` no coincide tras deshacer."
    assert np.array_equal(game.group_id_board, initial_gid_board), "❌ `group_id_board` no coincide tras deshacer."

    # Verificar que los grupos han sido restaurados
    assert len(game.black_groups) == len(initial_black_groups), "❌ `black_groups` no coincide tras deshacer."
    assert len(game.white_groups) == len(initial_white_groups), "❌ `white_groups` no coincide tras deshacer."

    for gid in initial_black_groups:
        g = game.black_groups[gid]
        orig = initial_black_groups[gid]
        assert g.stones == orig.stones, f"❌ Grupo negro {gid} piedras incorrectas"
        assert g.liberties == orig.liberties, f"❌ Grupo negro {gid} libertades incorrectas"

    for gid in initial_white_groups:
        g = game.white_groups[gid]
        orig = initial_white_groups[gid]
        assert g.stones == orig.stones, f"❌ Grupo blanco {gid} piedras incorrectas"
        assert g.liberties == orig.liberties, f"❌ Grupo blanco {gid} libertades incorrectas"

    # Verificar que no hay IDs solapados entre jugadores
    assert not (set(game.black_groups) & set(game.white_groups)), "❌ Conflicto de IDs entre grupos blancos y negros."

    print("✅ Test passed: make_move y undo_move funcionan correctamente con movimientos aleatorios en Go.")

# Ejecutar el test

for _ in range(10000):
    test_make_and_undo_go()

