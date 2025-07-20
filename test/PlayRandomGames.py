import time
import random
from test.Go9x9Faster import Go9x9Faster


def play_random_game(GameClass):
    game = GameClass()

    while not game.is_game_over():

        legal = game.legal_moves()
        move = random.choice(legal)
        game.make_move(move)
    return game.get_game_result()

def test_random_games(GameClass, num_games=100):
    print(f"Jugando {num_games} partidas aleatorias con {GameClass.__name__}...")
    start_time = time.time()
    results = {1: 0, -1: 0, 0: 0}

    for _ in range(num_games):
        result = play_random_game(GameClass)
        results[result] += 1

    end_time = time.time()
    duration = end_time - start_time

    print(f"Tiempo total: {duration:.2f} segundos")
    print(f"Promedio por partida: {duration / num_games:.4f} segundos")
    print("Resultados:")
    print(f"  Gana jugador 1: {results[1]}")
    print(f"  Gana jugador -1: {results[-1]}")
    print(f"  Empates: {results[0]}")

if __name__ == "__main__":

    test_random_games(Go9x9Faster, num_games=1000)
