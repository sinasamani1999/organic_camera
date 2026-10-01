"""Training samples from preprocessed phone clips (output of phone_preprocess.py).

Sample for time t (two consecutive steps, t and t+1, so the smoothness loss uses real neighbours):
    frames [2, 3, 224, 224]  frame t and t+1
    motion [2, T, 5]         rows t-T..t-1 (and t-T+1..t) of [intent_yaw, intent_pitch, res_yaw, res_pitch, speed]
    target [2, 2]            residual of rows t and t+1 (rotation frame t -> t+1), degrees

Splits are by clip, and clip names start with the person id (p01_walk_01), so a person is
never in both train and test. For a single clip (test01) the split falls back to time blocks.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

FEATURES = ["intent_yaw", "intent_pitch", "res_yaw", "res_pitch", "speed"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def load_motion(clip_dir):
    import csv
    with open(os.path.join(clip_dir, "motion.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    feats = np.array([[float(r[k]) for k in FEATURES] for r in rows], dtype=np.float32)
    return feats  # [N, 5]; target residual = feats[:, 2:4]


def sample_indices(n_rows, seq_len, lo=0, hi=None):
    """Valid t: needs rows t-T..t-1 as history and rows t, t+1 as targets."""
    hi = n_rows if hi is None else hi
    return np.arange(max(lo, seq_len), hi - 1)


def split_clips(clips, val_people, test_people):
    person = lambda c: c.split("_")[0]
    pick = lambda ps: [c for c in clips if person(c) in ps]
    val, test = pick(val_people), pick(test_people)
    train = [c for c in clips if c not in val and c not in test]
    return train, val, test


class PhoneOrganicDataset(Dataset):
    def __init__(self, root, clips, seq_len=10, time_range=(0.0, 1.0)):
        """time_range: fraction of each clip to use (for the single-clip time split)."""
        self.T = seq_len
        self.items = []   # (clip_idx, t)
        self.clips = []   # (frames_dir, feats)
        for c in clips:
            d = os.path.join(root, c)
            feats = load_motion(d)
            n = len(feats)
            lo, hi = int(time_range[0] * n), int(time_range[1] * n)
            ci = len(self.clips)
            self.clips.append((os.path.join(d, "frames"), feats))
            self.items += [(ci, int(t)) for t in sample_indices(n, seq_len, lo, hi)]

    def __len__(self):
        return len(self.items)

    def _frame(self, frames_dir, t):
        img = np.asarray(Image.open(os.path.join(frames_dir, f"{t:06d}.jpg")).convert("RGB"), dtype=np.float32)
        img = img.transpose(2, 0, 1) / 255.0
        return torch.from_numpy((img - MEAN) / STD)

    def __getitem__(self, i):
        ci, t = self.items[i]
        frames_dir, feats = self.clips[ci]
        frames = torch.stack([self._frame(frames_dir, t + k) for k in range(2)])
        motion = np.stack([feats[t + k - self.T:t + k] for k in range(2)])
        target = feats[t:t + 2, 2:4]
        return frames, torch.from_numpy(motion), torch.from_numpy(np.ascontiguousarray(target))


def build_phone_splits(cfg):
    d = cfg["data"]
    root, T = d["root"], d["seq_len"]
    clips = sorted(c for c in os.listdir(root) if os.path.isfile(os.path.join(root, c, "motion.csv")))
    if len(clips) == 1:  # smoke test on one clip: 70/15/15 by time, 1 s gap to limit leakage
        c = clips
        return (PhoneOrganicDataset(root, c, T, (0.0, 0.69)),
                PhoneOrganicDataset(root, c, T, (0.70, 0.84)),
                PhoneOrganicDataset(root, c, T, (0.85, 1.0)))
    tr, va, te = split_clips(clips, d.get("val_people", []), d.get("test_people", []))
    return PhoneOrganicDataset(root, tr, T), PhoneOrganicDataset(root, va, T), PhoneOrganicDataset(root, te, T)
