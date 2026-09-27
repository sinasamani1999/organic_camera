"""Temporal branch: GRU over the motion history -> query vector for cross-attention.

Per-frame features (MOTION_DIM = 5):
    [intent_yaw, intent_pitch, residual_yaw, residual_pitch, speed]
intent   = slow (player/operator) rotation per frame, degrees
residual = fast organic rotation per frame, degrees (the quantity the model predicts)
speed    = locomotion speed (m/s)
"""
import torch
import torch.nn as nn

MOTION_DIM = 5


class TemporalBackbone(nn.Module):
    def __init__(self, input_dim: int = MOTION_DIM, hidden_size: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_size, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, input_dim] -> [B, hidden_dim]"""
        out, _ = self.gru(x)
        return self.proj(out[:, -1])
