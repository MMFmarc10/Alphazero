import torch
import torch.nn as nn
import torch.nn.functional as F

# Red neuronal AlphaZero basada en ResNet.
# Recibe un tablero codificado como entrada y produce:
# - Una distribución de probabilidad sobre las acciones posibles (policy head).
# - Una estimación del valor del estado actual de la partida (value head).
class AlphaZeroModel(nn.Module):
    def __init__(self, game, num_residual_blocks, num_filters):
        super().__init__()

        self.rows = game.ROWS
        self.cols = game.COLS
        self.actions = game.ACTION_SIZE
        self.input_channels = game.encode_board().shape[0]

        # Entrada
        self.input_layer = nn.Sequential(
            nn.Conv2d(self.input_channels, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )

        # Bloques residuales
        self.residual_layers = nn.Sequential(*[
            self.residual_block(num_filters) for _ in range(num_residual_blocks)
        ])

        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * self.rows * self.cols, self.actions)

        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * self.rows * self.cols, 64)
        self.value_fc2 = nn.Linear(64, 1)

    # Define un bloque residual
    def residual_block(self, num_filters):
        return nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):

        # Entrada
        x = self.input_layer(x)

        # Bloques residuales
        for block in self.residual_layers:
            residual = x
            out = block(x)
            x = F.relu(out + residual)

        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
