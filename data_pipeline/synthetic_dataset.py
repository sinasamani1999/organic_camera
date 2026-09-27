"""Synthetic data with the same shapes as the real phone dataset.

Used to test the pipeline before real data exists. The target depends on the motion
history (damped oscillation scaled by speed) AND on the image (a bright square on the
left/right adds a yaw bias), so a correct model must use both branches.

Each item returns two consecutive steps:
    frames [2, 3, 224, 224], motion [2, T, 5], target [2, 2]
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticOrganicDataset(Dataset):
    def __init__(self, num_samples: int = 256, seq_len: int = 10, seed: int = 0):
        self.n, self.T = num_samples, seq_len
        rng = np.random.default_rng(seed)
        self.params = [
            dict(speed=rng.choice([0.0, 1.4, 3.0]), side=rng.choice([-1, 1]),
                 phase=rng.uniform(0, 2 * np.pi), seed=int(rng.integers(1 << 30)))
            for _ in range(num_samples)
        ]

    def __len__(self):
        return self.n

    def _sequence(self, p, length):
        t = np.arange(length)
        amp = 0.05 + 0.1 * p["speed"]
        freq = 0.15 + 0.1 * p["speed"]
        res_yaw = amp * np.sin(freq * t + p["phase"])
        res_pitch = 0.5 * amp * np.cos(2 * freq * t + p["phase"])
        intent = np.stack([0.3 * np.sin(0.02 * t + p["phase"]), np.zeros(length)], 1)
        speed = np.full((length, 1), p["speed"])
        return np.concatenate([intent, np.stack([res_yaw, res_pitch], 1), speed], 1)

    def _frame(self, p):
        img = torch.full((3, 224, 224), -1.0)
        x0 = 20 if p["side"] < 0 else 150
        img[:, 80:140, x0:x0 + 54] = 2.0
        return img

    def __getitem__(self, i):
        p = self.params[i]
        seq = self._sequence(p, self.T + 2)
        motion = np.stack([seq[k:k + self.T] for k in range(2)])            # [2, T, 5]
        target = seq[self.T:self.T + 2, 2:4].copy()                          # next residuals
        target[:, 0] += 0.1 * p["side"]                                      # visual bias
        frame = self._frame(p)
        return (torch.stack([frame, frame]),
                torch.tensor(motion, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32))
