import tkinter as tk
import math

import numpy as np
import torch

from AlphaZeroModel import AlphaZeroModel
from games.CuatroEnRayaFast import CuatroEnRayaFast

NODE_RADIUS = 20
X_SPACING = 80
Y_SPACING = 80

class MCTSVisualizer(tk.Tk):
    def __init__(self, root_node):
        super().__init__()
        self.title("MCTS Tree Visualization")
        self.canvas = tk.Canvas(self, width=1200, height=800, bg="white")
        self.canvas.pack(fill="both", expand=True)
        self.draw_tree(root_node, 600, 50, 600)

    def draw_tree(self, node, x, y, width, depth=0):
        # Dibuja el nodo actual
        node_text = f"M:{node.move}\nV:{node.value():.2f}\nN:{node.visit_count}"
        self.canvas.create_oval(x - NODE_RADIUS, y - NODE_RADIUS, x + NODE_RADIUS, y + NODE_RADIUS, fill="lightblue")
        self.canvas.create_text(x, y, text=node_text, font=("Arial", 8))

        # Dibuja los hijos
        num_children = len(node.children)
        if num_children == 0:
            return

        step = width / max(num_children, 1)
        start_x = x - width / 2 + step / 2

        for i, child in enumerate(node.children):
            child_x = start_x + i * step
            child_y = y + Y_SPACING

            # Línea hacia el hijo
            self.canvas.create_line(x, y + NODE_RADIUS, child_x, child_y - NODE_RADIUS)

            # Recursivo
            self.draw_tree(child, child_x, child_y, width / 2, depth + 1)


class MCTS:
    class Node:
        def __init__(self, prior, board, move=None, C=1.5):
            self.prior = prior
            self.visit_count = 0
            self.value_sum = 0
            self.children = []
            self.board = board
            self.move = move
            self.C = C

        def value(self):
            if self.visit_count == 0:
                return 0
            return self.value_sum / self.visit_count

        def is_expanded(self):
            return len(self.children) > 0

        def expand(self, distribution):

            for move, prob in enumerate(distribution):

                if prob != 0:

                    node = MCTS.Node(prob, self.board, move, C=self.C)
                    self.children.append(node)


        def print_values(self):
            print(self.move)
            print(self.visit_count)

        def select(self,i):

            mejor_ucb = -np.inf
            mejor_nodo = None
            for node in self.children:
                ucb_score = self.get_ucb_score(node)
                if ucb_score > mejor_ucb:
                    mejor_ucb = ucb_score
                    mejor_nodo = node
            return mejor_nodo

        def get_ucb_score(self, child):
            q_value = -child.value()
            u_value = self.C * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))
            return q_value + u_value

    def __init__(self, game, num_simulations, C, modelo, device):
        self.inicial_board = game
        self.simulations = num_simulations
        self.modelo = modelo
        self.device = device
        self.C = C

    def iniciar(self):
        root = self.Node(1, self.inicial_board.clone(), C=self.C)

        for i in range(self.simulations):
            node = root
            search_path = [node]

            print(f"\n=== Simulación {i + 1} ===")

            # Selection
            while node.is_expanded():
                node = node.select(i)
                print(f"Seleccionado movimiento: {node.move}, visitas: {node.visit_count}, valor: {node.value():.2f}")
                node.board.make_move(node.move)
                search_path.append(node)

            # Expansion o fin de partida
            value = 0
            terminada = node.board.is_game_over()
            if terminada:
                winner = node.board.get_game_result()
                player = node.board.player
                value = 1 if winner == player else -1 if winner != 0 else 0
                print(f"Fin de partida detectado: ganador {winner}, valor propagado: {value}")
            else:
                distribucion, value = self.obtener_distribucion(node.board)
                if i == 0:
                    print("Distribución inicial del modelo:", distribucion)
                node.expand(distribucion)
                print(f"Nodo expandido con movimientos {[m.move for m in node.children]}")

            # Backpropagation
            self.backpropagate(search_path, value)

            # Deshacer jugadas
            for node in reversed(search_path[1:]):
                node.board.undo_move()

            # Mostrar estadísticas tras cada simulación
            print("Estadísticas del root:")
            for child in root.children:
                print(f"   Movimiento {child.move}: visitas {child.visit_count}, valor medio {child.value():.2f}")

        # Política final
        moves = []
        distribution = np.zeros(self.inicial_board.ACTION_SIZE)
        for child in root.children:
            if child is not None:
                moves.append(child.move)
                distribution[child.move] = child.visit_count

        if np.sum(distribution) != 0:
            distribution /= np.sum(distribution)
        else:
            print("⚠️ Problema: root sin hijos explorados")
            root.board.print_board()

        print("Distribución final de visitas:", distribution)
        return root,moves, distribution, root.value()

    def backpropagate(self, search_path, value):
        oponente = 1
        for node in reversed(search_path):
            node.value_sum += value * oponente
            node.visit_count += 1
            oponente = oponente * -1

    def obtener_distribucion(self, board):

        self.modelo.eval()
        with torch.no_grad():
            tensor_board = torch.tensor(board.encode_board(), dtype=torch.float32).unsqueeze(0).to(self.device)

            policy, value = self.modelo(tensor_board)
        value = value.item()
        policy = torch.softmax(policy, dim=1).squeeze().detach().cpu().numpy()

        legal_move_policy = policy * board.legal_moves_mask()

        if np.sum(legal_move_policy) == 0:
            legal_move_policy = board.get_legal_moves_mask()

        legal_move_policy /= np.sum(legal_move_policy)
        return legal_move_policy, value




juego =CuatroEnRayaFast()
juego.make_move(3)
juego.make_move(2)
juego.make_move(3)
juego.make_move(2)
juego.make_move(3)
juego.make_move(2)

juego.print_board()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlphaZeroModel(juego, 8, 128).to(device)
model.load_state_dict(
torch.load("../model_versions/CuatroEnRayaTest/CuatroEnRayaTestBest.pth", map_location=device))
model.eval()
mcts = MCTS(juego, 300, 1.5, model, device)
root,_, probs, _ = mcts.iniciar()

visualizer = MCTSVisualizer(root)
visualizer.mainloop()