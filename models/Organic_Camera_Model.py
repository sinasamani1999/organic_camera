"""CNN-GRU-Cross-Attention model that predicts the organic residual rotation of the next frame."""
import torch
import torch.nn as nn

from models.Attention_Base import CrossAttention
from models.CNN_Base import VisualBackbone
from models.Temporal_Base import MOTION_DIM, TemporalBackbone


class OrganicCameraModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, pretrained: bool = True,
                 freeze_backbone: bool = True, use_visual: bool = True, use_attention: bool = True):
        super().__init__()
        self.use_visual = use_visual
        self.use_attention = use_attention
        self.temporal = TemporalBackbone(MOTION_DIM, 64, hidden_dim)
        if use_visual:
            self.visual = VisualBackbone(hidden_dim, pretrained=pretrained, freeze=freeze_backbone)
            if use_attention:
                self.cross = CrossAttention(hidden_dim, num_heads)
        in_dim = hidden_dim * 2 if use_visual else hidden_dim
        self.head = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))

    def forward(self, frame: torch.Tensor, motion: torch.Tensor, return_attention: bool = False):
        """frame: [B, 3, 224, 224], motion: [B, T, MOTION_DIM] -> residual [B, 2] (normalized units)."""
        q = self.temporal(motion)
        attn = None
        if not self.use_visual:
            fused = q
        else:
            tokens = self.visual(frame)
            if self.use_attention:
                ctx, attn = self.cross(tokens, q)
            else:  # ablation: plain concatenation of pooled features
                ctx = tokens.mean(dim=1)
            fused = torch.cat([ctx, q], dim=-1)
        out = self.head(fused)
        return (out, attn) if return_attention else out
