import tkinter as tk
import tkinter.messagebox
import torch
import numpy as np

from games.CuatroEnRayaFast import CuatroEnRayaFast
from AlphaZeroModel import AlphaZeroModel
from gui.MCTSsimpleNew import MCTS

import tkinter as tk
import tkinter.messagebox
import torch
import numpy as np

from games.CuatroEnRayaFast import CuatroEnRayaFast
from AlphaZeroModel import AlphaZeroModel
from gui.MCTSsimpleNew import MCTS

CELL_SIZE = 70
BOARD_WIDTH = CuatroEnRayaFast.COLS * CELL_SIZE
BOARD_HEIGHT = CuatroEnRayaFast.ROWS * CELL_SIZE
STONE_RADIUS = 30
BOARD_COLOR = "#0016d3"  # Azul (fondo del tablero)
RED_COLOR = "#FF4136"    # Rojo (jugador 1)
YELLOW_COLOR = "#FFD700" # Amarillo (jugador -1)


class Connect4VsModelView(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.game = CuatroEnRayaFast()

        # Cambia esto para elegir si el humano juega como 1 (negro) o -1 (blanco)
        self.human_player = -1

        # Cargar modelo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroModel(self.game, 8, 128).to(self.device)
        self.model.load_state_dict(torch.load("../model_versions/CuatroEnRayaTest2_/CuatroEnRayaTest2_Best.pth", map_location=self.device))
        self.model.eval()

        self.canvas = tk.Canvas(self, width=BOARD_WIDTH, height=BOARD_HEIGHT, bg=BOARD_COLOR)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)

        self.draw_board()
        self.draw_stones()

        # Si empieza el modelo, juega automáticamente
        if self.game.player != self.human_player:
            self.after(10, self.play_ai_move)

    def draw_board(self):
        self.canvas.delete("grid")
        for row in range(CuatroEnRayaFast.ROWS):
            for col in range(CuatroEnRayaFast.COLS):
                x0 = col * CELL_SIZE + 5
                y0 = row * CELL_SIZE + 5
                x1 = (col + 1) * CELL_SIZE - 5
                y1 = (row + 1) * CELL_SIZE - 5
                self.canvas.create_oval(x0, y0, x1, y1, fill="white", tags="grid")

    def draw_stones(self):
        self.canvas.delete("stone")
        for row in range(CuatroEnRayaFast.ROWS):
            for col in range(CuatroEnRayaFast.COLS):
                value = self.game.board[row][col]
                if value != 0:
                    x = col * CELL_SIZE + CELL_SIZE // 2
                    y = row * CELL_SIZE + CELL_SIZE // 2
                    color = RED_COLOR if value == 1 else YELLOW_COLOR
                    self.canvas.create_oval(x - STONE_RADIUS, y - STONE_RADIUS,
                                            x + STONE_RADIUS, y + STONE_RADIUS,
                                            fill=color, tags="stone")

    def handle_click(self, event):
        if self.game.player != self.human_player:
            return  # No es el turno del humano

        col = event.x // CELL_SIZE

        if col in self.game.legal_moves():
            if self.game.make_move(col):
                self.draw_board()
                self.draw_stones()
                if self.check_game_over():
                    return
                self.after(500, self.play_ai_move)

    def play_ai_move(self):
        mcts = MCTS(self.game, 300, 1.5, self.model, self.device)
        _, probs, _ = mcts.iniciar()

        move = np.argmax(probs)

        if move in self.game.legal_moves():
            self.game.make_move(move)

        self.draw_board()
        self.draw_stones()
        self.check_game_over()

    def check_game_over(self):
        if self.game.is_game_over():
            result = self.game.get_game_result()
            if result == 0:
                msg = "Empate"
            elif result == self.human_player:
                msg = "Ganaste"
            else:
                msg = "Ganó el modelo"

            tk.messagebox.showinfo("Fin del juego", msg)
            return True
        return False
