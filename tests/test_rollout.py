"""Checks for rollout training (no real data or GPU needed).   python -m tests.test_rollout

1. loss       : RolloutLoss shapes; spectral term is ~0 for identical signals and much smaller than
                MSE for the same rhythm with a different phase
2. dataset    : PhoneRolloutDataset windows have the right shapes and line up with the clip
3. same as eval: train.rollout gives the same predictions as evaluate.py --mode free
4. gradient   : the loss reaches the GRU through the fed-back outputs
"""
import math
import os
import tempfile

import numpy as np
import torch
from PIL import Image

from data_pipeline.phone_dataset import PhoneOrganicDataset, PhoneRolloutDataset
from evaluate import predict_clip
from models.losses import RolloutLoss
from models.Organic_Camera_Model import OrganicCameraModel
from train import Normalizer, rollout

T, K = 10, 30


def make_clip(root, name="p01_office_walk_01", n=80, frames=True):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "frames"), exist_ok=True)
    rng = np.random.default_rng(0)
    t = np.arange(n) / 30.0
    feats = np.stack([0.3 * np.sin(t), 0.1 * np.cos(t), 0.2 * np.sin(2 * math.pi * 2 * t),
                      0.15 * np.sin(2 * math.pi * 2 * t + 1), np.full(n, 1.4)], 1) + 0.01 * rng.standard_normal((n, 5))
    with open(os.path.join(d, "motion.csv"), "w") as f:
        f.write("intent_yaw,intent_pitch,res_yaw,res_pitch,speed\n")
        for r in feats:
            f.write(",".join(f"{v:.6f}" for v in r) + "\n")
    if frames:
        img = Image.fromarray(np.full((224, 224, 3), 128, np.uint8))
        for i in range(n):
            img.save(os.path.join(d, "frames", f"{i:06d}.jpg"))
    return feats


def test_loss():
    crit = RolloutLoss(delta=1.0)
    t = torch.arange(K) / 30.0
    a = torch.sin(2 * math.pi * 2 * t).view(1, K, 1).repeat(8, 1, 2)
    b = torch.sin(2 * math.pi * 2 * t + math.pi).view(1, K, 1).repeat(8, 1, 2)  # same rhythm, opposite phase
    _, same = crit(a, a)
    _, shifted = crit(b, a)
    assert same["spec"] < 1e-6, same
    assert shifted["spec"] < 0.05 * shifted["pos"], shifted
    _, flat = crit(torch.zeros_like(a), a)
    assert flat["spec"] > 1.0, flat  # predicting no motion is punished by the spectral term
    print(f"OK loss (phase-shifted: pos {shifted['pos']:.3f}, spec {shifted['spec']:.4f}; zero output: spec {flat['spec']:.2f})")


def test_dataset():
    with tempfile.TemporaryDirectory() as root:
        feats = make_clip(root, frames=True)
        ds = PhoneRolloutDataset(root, ["p01_office_walk_01"], T, K, stride=5, load_frames=True)
        assert len(ds) == len(range(T, len(feats) - K + 1, 5)), len(ds)
        frames, motion, target = ds[1]
        t0 = ds.items[1][1]
        assert frames.shape == (K, 3, 224, 224), frames.shape
        assert motion.shape == (T + K, 5) and target.shape == (K, 2)
        assert np.allclose(motion.numpy(), feats[t0 - T:t0 + K], atol=1e-5)
        assert np.allclose(target.numpy(), feats[t0:t0 + K, 2:4], atol=1e-5)
        ds2 = PhoneRolloutDataset(root, ["p01_office_walk_01"], T, K, stride=5, load_frames=False)
        assert ds2[0][0].shape == (K, 1)
    print(f"OK dataset ({len(ds)} windows)")


def test_same_as_free_eval():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as root:
        make_clip(root, frames=True)
        model = OrganicCameraModel(pretrained=False, use_visual=False).eval()
        rd = PhoneRolloutDataset(root, ["p01_office_walk_01"], T, K, stride=5, load_frames=False)
        norm = Normalizer.fit(rd)
        _, motion, _ = rd[0]  # window starting at t0 = T, like the start of evaluate's free mode
        with torch.no_grad():
            pred = rollout(model, torch.zeros(1, K, 1), motion.unsqueeze(0), norm, T, K, False)[0]
        pred = (pred * norm.t_std + norm.t_mean).numpy()
        sd = PhoneOrganicDataset(root, ["p01_office_walk_01"], T)
        nd = {k: v for k, v in norm.state().items()}
        ts, free, _ = predict_clip(model, nd, sd, 0, "free", torch.device("cpu"))
        assert ts[0] == T
        err = np.abs(pred - free[:K]).max()
        assert err < 1e-4, err
    print(f"OK rollout == evaluate free mode (max diff {err:.2e})")


def test_gradient_through_feedback():
    torch.manual_seed(0)
    model = OrganicCameraModel(pretrained=False, use_visual=False)
    norm = Normalizer(torch.zeros(5), torch.ones(5), torch.zeros(2), torch.ones(2))
    motion = torch.randn(4, T + K, 5)
    pred = rollout(model, torch.zeros(4, K, 1), motion, norm, T, K, False)
    pred[:, -1].sum().backward()
    gru = sum(p.grad.abs().sum().item() for p in model.temporal.gru.parameters())
    assert gru > 0
    print(f"OK gradient (GRU grad {gru:.3e})")


def test_with_images():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as root:
        make_clip(root, frames=True)
        model = OrganicCameraModel(pretrained=False, use_visual=True)
        rd = PhoneRolloutDataset(root, ["p01_office_walk_01"], T, K, stride=20, load_frames=True)
        norm = Normalizer.fit(rd)
        frames, motion, target = (x.unsqueeze(0) for x in rd[0])
        pred = rollout(model, frames, motion, norm, T, K, True)
        loss, parts = RolloutLoss(delta=1.0)(pred, norm.target(target))
        loss.backward()
        assert pred.shape == (1, K, 2)
    print(f"OK with images (loss {loss.item():.3f})")


if __name__ == "__main__":
    test_loss()
    test_dataset()
    test_same_as_free_eval()
    test_gradient_through_feedback()
    test_with_images()
    print("all rollout checks passed")
