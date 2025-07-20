import random
from games.Go5x5 import Go5x5


def simulate_random_game():
    game = Go5x5()

    while not game.is_game_over():
        print("**************************")
        print(game.player)
        game.print_board()
        print(game.encode_board())
        legal = game.legal_moves()
        move = random.choice(legal)
        game.make_move(move)


    print("\n=== Partida terminada ===")
    game.print_board()

    winner = game.get_game_result()
    if winner == 1:
        print("Ganador: Negro (X)")
    elif winner == -1:
        print("Ganador: Blanco (O)")
    else:
        print("Empate")


if __name__ == "__main__":
    simulate_random_game()
