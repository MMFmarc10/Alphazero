import tkinter as tk
from tkinter import messagebox
from games.Go9x9 import Go9x9
import colorsys
import numpy as np

CELL_SIZE = 60
BOARD_SIZE = Go9x9.SIZE
STONE_RADIUS = 20
LINE_COLOR = "#000"
BOARD_COLOR = "#DEB887"
BLACK_COLOR = "#000000"
WHITE_COLOR = "#FFFFFF"
HIGHLIGHT_COLOR = "#00F"


def group_color(gid):
    # Genera un color RGB único por grupo usando HSV
    h = (gid * 0.15) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.6, 0.9)
    return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))


class GoView(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.game = Go9x9()
        self.canvas = tk.Canvas(self, width=BOARD_SIZE * CELL_SIZE, height=BOARD_SIZE * CELL_SIZE, bg=BOARD_COLOR)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.handle_click)

        self.last_move = None

        self.draw_board()
        self.draw_stones()

        self.pass_button = tk.Button(self, text="Pasar turno", command=self.pass_turn)
        self.pass_button.pack(pady=5)

        self.undo_button = tk.Button(self, text="Deshacer", command=self.undo_move)
        self.undo_button.pack(pady=5)

    def draw_board(self):
        self.canvas.delete("grid")
        for i in range(BOARD_SIZE):
            x = y = CELL_SIZE // 2 + i * CELL_SIZE
            self.canvas.create_line(CELL_SIZE // 2, y, CELL_SIZE * (BOARD_SIZE - 0.5), y, fill=LINE_COLOR, tags="grid")
            self.canvas.create_line(x, CELL_SIZE // 2, x, CELL_SIZE * (BOARD_SIZE - 0.5), fill=LINE_COLOR, tags="grid")

    def draw_stones(self):
        self.canvas.delete("stone")
        self.canvas.delete("liberty")

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                value = self.game.board[r][c]
                group_id = self.game.group_id_board[r, c]
                color = group_color(group_id) if group_id >= 0 else "gray"

                if value != 0:
                    x = c * CELL_SIZE + CELL_SIZE // 2
                    y = r * CELL_SIZE + CELL_SIZE // 2
                    stone_color = BLACK_COLOR if value == 1 else WHITE_COLOR

                    self.canvas.create_oval(x - STONE_RADIUS, y - STONE_RADIUS,
                                            x + STONE_RADIUS, y + STONE_RADIUS,
                                            fill=stone_color, tags="stone")

                    self.canvas.create_text(
                        x, y,
                        text=str(group_id),
                        fill=color,
                        font=("Arial", 10, "bold"),
                        tags="stone")

        # Libertades solo de los grupos que están en el tablero
        grupos_activos = np.unique(self.game.group_id_board[self.game.group_id_board >= 0])

        # Agrupar las libertades por celda
        liberty_map = {}

        for gid in grupos_activos:
            liberties = self.game.group_liberties[gid]
            liberty_color = group_color(gid)

            for lr, lc in np.argwhere(liberties):
                liberty_map.setdefault((lr, lc), []).append(liberty_color)

        # Pintar cada celda
        for (lr, lc), colors in liberty_map.items():
            x = lc * CELL_SIZE + CELL_SIZE // 2
            y = lr * CELL_SIZE + CELL_SIZE // 2

            # Si hay muchos colores, los desplazamos horizontalmente
            total_colors = len(colors)
            start_offset = - (total_colors - 1) * 4  # Centra los puntos

            for i, color in enumerate(colors):
                offset = start_offset + i * 8  # Espacio entre colores
                self.canvas.create_oval(
                    x + offset - 3, y - 3,
                    x + offset + 3, y + 3,
                    fill=color,
                    outline="",
                    tags="liberty"
                )

        # Última jugada
        if self.last_move:
            r, c = self.last_move
            x = c * CELL_SIZE + CELL_SIZE // 2
            y = r * CELL_SIZE + CELL_SIZE // 2
            # Solo borde, sin relleno
            self.canvas.create_oval(
                x - 8, y - 8, x + 8, y + 8,
                outline=HIGHLIGHT_COLOR,
                width=2,
                tags="stone"
            )

    def handle_click(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        action = row * BOARD_SIZE + col

        if action in self.game.legal_moves():
            success = self.game.make_move(action)
            if success:
                self.last_move = (row, col)
                self.draw_board()
                self.draw_stones()

                if self.game.is_game_over():
                    result = self.game.get_game_result()
                    msg = "Empate"
                    if result == 1:
                        msg = "Ganó negro"
                    elif result == -1:
                        msg = "Ganó blanco"
                    messagebox.showinfo("Fin del juego", msg)

    def pass_turn(self):
        action = self.game.ACTION_SIZE - 1
        self.game.make_move(action)
        self.last_move = None
        self.draw_board()
        self.draw_stones()

        if self.game.is_game_over():
            result = self.game.get_game_result()
            msg = "Empate"
            if result == 1:
                msg = "Ganó negro"
            elif result == -1:
                msg = "Ganó blanco"
            messagebox.showinfo("Fin del juego", msg)

    def undo_move(self):
        if not self.game.history:
            return

        self.game.undo_move()
        self.last_move = None
        self.draw_board()
        self.draw_stones()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Go AlphaZero View")
    app = GoView(master=root)
    app.pack()
    root.mainloop()
