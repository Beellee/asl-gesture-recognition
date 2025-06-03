import torch
import torch.nn as nn
import torch.nn.functional as F

class SignLSTM(nn.Module):
    """
    LSTM que recibe:
      - x: (B, T=73, D=126)
      - lengths: (B, 1) duración normalizada [0,1]
    Devuelve logits (B, num_classes).
    """
    def __init__(self,
                 input_dim: int = 126,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 num_classes: int = 15,
                 dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = input_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0
        )
        self.layernorm = nn.LayerNorm(hidden_dim*2) # LayerNorm entre capas para evitar sobreajuste.
        self.fc1 = nn.Linear(hidden_dim +1 + 2, 64)  # +1 para la duracion y +2 para cada mano
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, hand_fracs: torch.Tensor) -> torch.Tensor:
        # x: (B, 73, 126), lengths: (B,1), hand_fracs: (B,1)
        _, (h_n, _) = self.lstm(x)        # h_n: (num_layers, B, hidden_dim)
        h = h_n[-1]                        # (B, hidden_dim), última capa
        feats = torch.cat([lengths, hand_fracs], dim=1)
        h_and_feats = torch.cat([h, feats], dim=1)
        y = F.relu(self.fc1(h_and_feats))     # (B, 64)
        return self.fc2(y)                # (B, num_classes)
