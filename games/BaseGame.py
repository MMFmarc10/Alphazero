from abc import ABC, abstractmethod

import numpy as np


class BaseGame(ABC):

    @abstractmethod
    def legal_moves(self) -> list[int]:
        """Devuelve la lista de movimientos legales del estado de la partida."""
        pass

    @abstractmethod
    def legal_moves_mask(self) -> np.ndarray:
        """Devuelve una máscara binaria de movimientos legales."""
        pass

    @abstractmethod
    def make_move(self, action: int) -> bool:
        """Realiza un movimiento si es válido."""
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        """Devuelve True si el juego ha terminado."""
        pass

    @abstractmethod
    def get_game_result(self) -> int:
        """
        Devuelve el jugador ganador:
        - 0 si es empate
        """
        pass

    @abstractmethod
    def get_action_space(self) -> int:
        """Devuelve el número total de acciones posibles."""
        pass

    @abstractmethod
    def encode_board(self) -> np.ndarray:
        """Devuelve una representación en tensor del tablero para la red."""
        pass

    @abstractmethod
    def get_opposite_player(self) -> int:
        """Devuelve el jugador contrario al actual."""
        pass

    @abstractmethod
    def print_board(self):
        """Imprime el tablero en consola de forma legible."""
        pass
