import math

import numpy as np
import torch


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
                    board_child = self.board.clone()
                    #board_child = copy.deepcopy(self.board)
                    board_child.make_move(move)

                    node = MCTS.Node(prob, board_child, move, C=self.C)
                    self.children.append(node)

        def print_values(self):
            print(self.move)
            print(self.visit_count)

        def select(self):

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

        root = self.Node(1, self.inicial_board, C=self.C)

        for _ in range(self.simulations):

            node = root
            search_path = [node]

            # selection
            while node.is_expanded():
                node = node.select()
                search_path.append(node)

            value = 0
            terminada = node.board.is_game_over()
            if terminada:

                winner = node.board.get_game_result()
                player = node.board.player

                if winner == 0:
                    value = 0
                elif player == winner:
                    value = 1
                else:
                    value = -1

            if not terminada:
                distribucion, value = self.obtener_distribucion(node.board)
                node.expand(distribucion)

            self.backpropagate(search_path, value)
        moves = []
        distribution = np.zeros(self.inicial_board.ACTION_SIZE)
        for _, child in enumerate(root.children):
            moves.append(child.move)
            distribution[child.move] = child.visit_count
        if np.sum(distribution) != 0:
            distribution /= np.sum(distribution)
        else:
            print("proooooblllleeeeeeem")
            print(root.board.print_board())

        return moves, distribution, root.value()

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

