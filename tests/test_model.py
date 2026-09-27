"""Week-1 gate tests.   python -m tests.test_model

1. shapes           : output [B, 2], attention [B, 49] summing to 1
2. motion matters   : changing the motion history changes the output, GRU gets gradient
3. overfit          : loss on a small synthetic set drops by > 90 %
"""
import torch
from torch.utils.data import DataLoader

from data_pipeline.synthetic_dataset import SyntheticOrganicDataset
from models.losses import OrganicCameraLoss
from models.Organic_Camera_Model import OrganicCameraModel

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_shapes():
    m = OrganicCameraModel(pretrained=False).to(DEV).eval()
    out, attn = m(torch.randn(4, 3, 224, 224, device=DEV), torch.randn(4, 10, 5, device=DEV), return_attention=True)
    assert out.shape == (4, 2), out.shape
    assert attn.shape == (4, 49), attn.shape
    assert torch.allclose(attn.sum(-1), torch.ones(4, device=DEV), atol=1e-4)
    print("OK shapes")


def test_motion_matters():
    torch.manual_seed(0)
    m = OrganicCameraModel(pretrained=False).to(DEV)
    f = torch.randn(2, 3, 224, 224, device=DEV)
    a = m.eval()(f, torch.zeros(2, 10, 5, device=DEV))
    b = m(f, torch.ones(2, 10, 5, device=DEV))
    assert (a - b).abs().max() > 1e-4, "output ignores motion history"
    m.train()
    m(f, torch.randn(2, 10, 5, device=DEV)).sum().backward()
    g = sum(p.grad.abs().sum().item() for p in m.temporal.gru.parameters())
    assert g > 0, "GRU receives no gradient"
    print(f"OK motion matters (GRU grad {g:.3e})")


def test_overfit(steps=300):
    torch.manual_seed(0)
    ds = SyntheticOrganicDataset(32)
    dl = DataLoader(ds, batch_size=32)
    frames, motion, target = next(iter(dl))
    frames, motion, target = frames.to(DEV).flatten(0, 1), motion.to(DEV).flatten(0, 1), target.to(DEV)
    target = (target - target.mean((0, 1))) / target.std((0, 1))
    m = OrganicCameraModel(pretrained=False).to(DEV)
    crit = OrganicCameraLoss()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    first = None
    for s in range(steps):
        loss, _ = crit(m(frames, motion).view(32, 2, 2), target)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.1 * first, f"did not overfit: {first:.3f} -> {loss.item():.3f}"
    print(f"OK overfit {first:.3f} -> {loss.item():.4f}")


if __name__ == "__main__":
    print("device:", DEV)
    test_shapes()
    test_motion_matters()
    test_overfit()
    print("ALL WEEK-1 MODEL TESTS PASSED")
