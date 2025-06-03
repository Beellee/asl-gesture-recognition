import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# bloque residual causal dilatado
class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(channels, channels, 1)  # proyección 1×1 si cambia dim

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv(x)               # (B, C, T + padding)
        out = out[..., :x.shape[2]]      # recorta el exceso de padding
        out = self.relu(out)
        out = self.dropout(out)
        # conexión residual
        return self.relu(out + self.residual(x))


# TCN causal completa
class SignTCN(nn.Module):
    def __init__(self,
                 input_dim: int = 126,
                 num_classes: int = 15,
                 num_channels: int = 64,
                 kernel_size: int = 3,
                 dilations: list = [1, 2, 4, 8],
                 dropout: float = 0.1):
        super().__init__()

        # Primera capa para adaptar input_dim→num_channels
        self.init_conv = nn.Conv1d(input_dim, num_channels,
                                   kernel_size=1)

        # Bloques residuales dilatados
        self.res_blocks = nn.ModuleList([
            DilatedResidualBlock(num_channels, kernel_size, d, dropout)
            for d in dilations
        ])

        # Global pooling para colapsar dimensión temporal variable
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Capas fully-connected
        # +3: 1 para duration + 2 para fracción izquierda/derecha
        self.fc1 = nn.Linear(num_channels + 3, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self,
                x: torch.Tensor,         # (B, T, D)
                lengths: torch.Tensor,   # (B, 1)
                hand_fracs: torch.Tensor # (B, 2)
               ) -> torch.Tensor:
        # tranpose → (B, D, T)
        x = x.transpose(1, 2)

        # adaptamos dimensión de canales
        x = self.init_conv(x)  # (B, C, T)

        # aplicamos cada bloque residual dilatado
        for block in self.res_blocks:
            x = block(x)       # (B, C, T)

        # pooling global → (B, C, 1) → (B, C)
        x = self.global_pool(x).squeeze(-1)

        # concatenar duration + hand_fracs (B, 3)
        feats = torch.cat([lengths, hand_fracs], dim=1)

        # fully-connected
        h = F.relu(self.fc1(torch.cat([x, feats], dim=1)))
        return self.fc2(h)  # (B, num_classes)

