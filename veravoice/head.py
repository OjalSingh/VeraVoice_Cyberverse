"""Small classifier over per-layer WavLM features."""

from __future__ import annotations

import torch
from torch import nn


class LayerWeightedHead(nn.Module):
    """Softmax-weighted sum over layers, then a 2-layer MLP producing one logit (spoof)."""

    def __init__(self, num_layers: int, hidden_size: int, mlp_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1),
        )
        self.config = {"num_layers": num_layers, "hidden_size": hidden_size, "mlp_dim": mlp_dim, "dropout": dropout}

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, L, H)
        x = (feats * self.layer_weights()[None, :, None]).sum(1)
        return self.mlp(self.norm(x)).squeeze(-1)
