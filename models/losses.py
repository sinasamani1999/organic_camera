"""Hybrid loss on two consecutive predictions per sample (fixes smoothness under shuffling)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrganicCameraLoss(nn.Module):
    """pred, target: [B, 2 steps, 2 axes].

    position   : MSE on both steps
    direction  : 1 - cosine similarity, only where the target is clearly non-zero
    smoothness : MSE between predicted and true frame-to-frame change (penalises jitter,
                 not motion itself, and uses pairs from the SAME sample)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.6, gamma: float = 0.4, min_norm: float = 0.1):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.min_norm = alpha, beta, gamma, min_norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l_pos = F.mse_loss(pred, target)

        p, t = pred.reshape(-1, 2), target.reshape(-1, 2)
        mask = t.norm(dim=-1) > self.min_norm
        if mask.any():
            l_dir = (1 - F.cosine_similarity(p[mask], t[mask], dim=-1, eps=1e-6)).mean()
        else:
            l_dir = pred.new_zeros(())

        l_smooth = F.mse_loss(pred[:, 1] - pred[:, 0], target[:, 1] - target[:, 0])

        total = self.alpha * l_pos + self.beta * l_dir + self.gamma * l_smooth
        return total, {"pos": l_pos.item(), "dir": l_dir.item(), "smooth": l_smooth.item()}
