import tkinter as tk
import tkinter.messagebox

from gui.Connect4VsModelView import Connect4VsModelView
from gui.GoVsModelView import GoVsModelView
from gui.goView import GoView


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AlphaZero Games")
        self.geometry("500x580")

        self.current_view = None
        self.show_game_selector()

    def show_game_selector(self):
        frame = tk.Frame(self)
        frame.pack()

        tk.Label(frame, text="Selecciona un juego:").pack()


        tk.Button(frame, text="Go", command=lambda: self.load_game(GoView)).pack(pady=5)
        tk.Button(frame, text="Go vs Modelo", command=lambda: self.load_game(GoVsModelView)).pack(pady=5)
        tk.Button(frame, text="CuatroEnRaya", command=lambda: self.load_game(Connect4VsModelView)).pack(pady=5)

    def load_game(self, view_class):
        if self.current_view:
            self.current_view.destroy()

        self.current_view = view_class(self)
        self.current_view.pack(fill="both", expand=True)

if __name__ == "__main__":
    app = App()
    app.mainloop()
