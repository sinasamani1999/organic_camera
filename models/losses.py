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


class RolloutLoss(nn.Module):
    """Loss over K free-running steps. pred, target: [B, K, 2] (normalized units).

    position, direction : as OrganicCameraLoss, over all K steps
    smoothness          : MSE between predicted and true step-to-step change, over all K-1 pairs
    spectral (delta)    : MSE between the log power spectra of prediction and target over the K-step
                          window, averaged over the batch first. It does not depend on phase, so it
                          rewards the right rhythm and amplitude (e.g. the step frequency) even when the
                          exact timing of each step cannot be predicted. delta = 0 turns it off.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.6, gamma: float = 0.4, delta: float = 0.0,
                 min_norm: float = 0.1, eps: float = 1e-4):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.min_norm, self.eps = min_norm, eps

    def spectrum(self, x):
        """Batch-mean power spectrum of x [B, K, 2] -> [K//2 + 1, 2], mean removed, Hann window."""
        w = torch.hann_window(x.shape[1], periodic=False, device=x.device, dtype=x.dtype).view(1, -1, 1)
        return torch.fft.rfft((x - x.mean(1, keepdim=True)) * w, dim=1).abs().pow(2).mean(0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l_pos = F.mse_loss(pred, target)

        p, t = pred.reshape(-1, 2), target.reshape(-1, 2)
        mask = t.norm(dim=-1) > self.min_norm
        l_dir = (1 - F.cosine_similarity(p[mask], t[mask], dim=-1, eps=1e-6)).mean() if mask.any() else pred.new_zeros(())

        l_smooth = F.mse_loss(pred.diff(dim=1), target.diff(dim=1))

        if self.delta > 0:
            ps, ts = self.spectrum(pred)[1:], self.spectrum(target)[1:]  # skip DC
            l_spec = F.mse_loss(torch.log(ps + self.eps), torch.log(ts + self.eps))
        else:
            l_spec = pred.new_zeros(())

        total = self.alpha * l_pos + self.beta * l_dir + self.gamma * l_smooth + self.delta * l_spec
        return total, {"pos": l_pos.item(), "dir": l_dir.item(), "smooth": l_smooth.item(), "spec": l_spec.item()}
