"""Train the organic camera model.

    python train.py --config config.yaml
"""
import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from data_pipeline.synthetic_dataset import SyntheticOrganicDataset
from models.losses import OrganicCameraLoss
from models.Organic_Camera_Model import OrganicCameraModel


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_dataset(cfg):
    kind = cfg["data"]["kind"]
    if kind == "synthetic":
        return SyntheticOrganicDataset(cfg["data"]["num_samples"], cfg["data"]["seq_len"], cfg["seed"])
    raise ValueError(f"unknown dataset kind: {kind}")  # 'phone' is added in week 2


class Normalizer:
    """Per-feature mean/std from training data; saved in the checkpoint for inference."""

    def __init__(self, m_mean, m_std, t_mean, t_std):
        self.m_mean, self.m_std, self.t_mean, self.t_std = m_mean, m_std, t_mean, t_std

    @classmethod
    def fit(cls, dataset, max_items=2000):
        ms, ts = [], []
        for i in range(min(len(dataset), max_items)):
            _, m, t = dataset[i]
            ms.append(m.reshape(-1, m.shape[-1])); ts.append(t.reshape(-1, 2))
        m, t = torch.cat(ms), torch.cat(ts)
        return cls(m.mean(0), m.std(0).clamp_min(1e-6), t.mean(0), t.std(0).clamp_min(1e-6))

    def to(self, device):
        for k in ("m_mean", "m_std", "t_mean", "t_std"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def motion(self, m): return (m - self.m_mean) / self.m_std
    def target(self, t): return (t - self.t_mean) / self.t_std
    def state(self): return {k: getattr(self, k).cpu() for k in ("m_mean", "m_std", "t_mean", "t_std")}


def run_epoch(model, loader, criterion, norm, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for frames, motion, target in loader:
            B = frames.shape[0]
            frames = frames.to(device).flatten(0, 1)                  # [B*2, 3, H, W]
            motion = norm.motion(motion.to(device)).flatten(0, 1)     # [B*2, T, 5]
            target = norm.target(target.to(device))                   # [B, 2, 2]
            pred = model(frames, motion).view(B, 2, 2)
            loss, _ = criterion(pred, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * B; n += B
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8"))
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    ds = build_dataset(cfg)
    n_val = max(1, int(len(ds) * cfg["data"]["val_fraction"]))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(cfg["seed"]))
    # NOTE: for real data the split must be by person (done in the dataset), not random frames.
    tl = DataLoader(train_ds, cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["workers"], drop_last=True)
    vl = DataLoader(val_ds, cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["workers"])
    norm = Normalizer.fit(train_ds).to(device)

    m = cfg["model"]
    model = OrganicCameraModel(m["hidden_dim"], m["num_heads"], m["pretrained"], m["freeze_backbone"],
                               m["use_visual"], m["use_attention"]).to(device)
    lw = cfg["loss"]
    criterion = OrganicCameraLoss(lw["alpha"], lw["beta"], lw["gamma"], lw["min_norm"])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["train"]["lr"])

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(cfg["out_dir"])
    except Exception:
        writer = None

    os.makedirs(cfg["out_dir"], exist_ok=True)
    best = float("inf")
    for epoch in range(cfg["train"]["epochs"]):
        tr = run_epoch(model, tl, criterion, norm, device, opt)
        va = run_epoch(model, vl, criterion, norm, device)
        print(f"epoch {epoch + 1:3d}  train {tr:.4f}  val {va:.4f}")
        if writer:
            writer.add_scalars("loss", {"train": tr, "val": va}, epoch)
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "norm": norm.state(), "config": cfg},
                       os.path.join(cfg["out_dir"], "best.pth"))
    print(f"best val loss {best:.4f} -> {cfg['out_dir']}/best.pth")


if __name__ == "__main__":
    main()
