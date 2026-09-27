"""Cross-attention: motion query attends over 49 visual tokens."""
import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, feature_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, visual_tokens: torch.Tensor, motion_query: torch.Tensor):
        """visual_tokens: [B, N, D] (Key/Value), motion_query: [B, D] (Query).

        Returns context [B, D] and attention weights [B, N] (averaged over heads).
        """
        q = motion_query.unsqueeze(1)
        ctx, w = self.attn(q, visual_tokens, visual_tokens, need_weights=True, average_attn_weights=True)
        ctx = self.norm(ctx.squeeze(1) + motion_query)  # residual keeps the motion path alive
        return ctx, w.squeeze(1)
