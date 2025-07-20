import tkinter as tk
import tkinter.messagebox
import torch
import numpy as np

from games.Go5x5 import Go5x5
from AlphaZeroModel import AlphaZeroModel
from gui.MCTSsimpleNew import MCTS

CELL_SIZE = 60
BOARD_SIZE = Go5x5.SIZE
STONE_RADIUS = 20
LINE_COLOR = "#000"
BOARD_COLOR = "#DEB887"
BLACK_COLOR = "#000000"
WHITE_COLOR = "#FFFFFF"
HIGHLIGHT_COLOR = "#00F"

class GoVsModelView(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.game = Go5x5()

        # Puedes cambiar aquí quién juega como 1: 1 = humano, -1 = modelo
        self.human_player = 1

        # Cargar modelo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroModel(self.game, 8, 128).to(self.device)
        self.model.load_state_dict(torch.load("../model_versions/Go_5x5_shared/Go_5x5_sharedBest.pth", map_location=self.device))
        self.model.eval()

        self.canvas = tk.Canvas(self, width=BOARD_SIZE * CELL_SIZE, height=BOARD_SIZE * CELL_SIZE, bg=BOARD_COLOR)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)

        self.last_move = None

        self.pass_button = tk.Button(self, text="Pasar turno", command=self.pass_turn)
        self.pass_button.pack(pady=10)

        self.draw_board()
        self.draw_stones()

        # Si empieza el modelo, juega automáticamente
        if self.game.player != self.human_player:
            self.after(500, self.play_ai_move)

    def draw_board(self):
        self.canvas.delete("grid")
        for i in range(BOARD_SIZE):
            x = y = CELL_SIZE // 2 + i * CELL_SIZE
            self.canvas.create_line(CELL_SIZE // 2, y, CELL_SIZE * (BOARD_SIZE - 0.5), y, fill=LINE_COLOR, tags="grid")
            self.canvas.create_line(x, CELL_SIZE // 2, x, CELL_SIZE * (BOARD_SIZE - 0.5), fill=LINE_COLOR, tags="grid")

    def draw_stones(self):
        self.canvas.delete("stone")
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                value = self.game.board[r][c]
                if value != 0:
                    x = c * CELL_SIZE + CELL_SIZE // 2
                    y = r * CELL_SIZE + CELL_SIZE // 2
                    color = BLACK_COLOR if value == 1 else WHITE_COLOR
                    self.canvas.create_oval(x - STONE_RADIUS, y - STONE_RADIUS,
                                            x + STONE_RADIUS, y + STONE_RADIUS,
                                            fill=color, tags="stone")

        if self.last_move:
            r, c = self.last_move
            x = c * CELL_SIZE + CELL_SIZE // 2
            y = r * CELL_SIZE + CELL_SIZE // 2
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=HIGHLIGHT_COLOR, tags="stone")

    def handle_click(self, event):
        if self.game.player != self.human_player:
            return  # No es el turno del humano

        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        action = row * BOARD_SIZE + col

        if action in self.game.legal_moves():
            if self.game.make_move(action):
                self.last_move = (row, col)
                self.draw_board()
                self.draw_stones()
                if self.check_game_over():
                    return
                self.after(500, self.play_ai_move)

    def play_ai_move(self):
        mcts = MCTS(self.game, 300, 1.5, self.model, self.device)
        _, probs, _ = mcts.iniciar()
        print(probs)
        move = np.argmax(probs)
        self.game.make_move(move)

        if move != self.game.ACTION_SIZE - 1:
            self.last_move = divmod(move, BOARD_SIZE)
        else:
            self.last_move = None

        self.draw_board()
        self.draw_stones()
        self.check_game_over()

    def pass_turn(self):
        if self.game.player != self.human_player:
            return  # Solo el humano puede pasar
        self.game.make_move(self.game.ACTION_SIZE - 1)
        self.last_move = None
        self.draw_board()
        self.draw_stones()
        if not self.check_game_over():
            self.after(500, self.play_ai_move)

    def check_game_over(self):
        if self.game.is_game_over():
            result = self.game.get_game_result()
            msg = "Empate"
            if result == self.human_player:
                msg = "Ganaste"
            elif result == 0:
                msg = "Empate"
            else:
                msg = "Ganó el modelo"
            tk.messagebox.showinfo("Fin del juego", msg)
            return True
        return False
