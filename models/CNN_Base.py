"""Visual branch: ResNet-18 feature map as 49 spatial tokens (Key/Value for cross-attention)."""
import os

import torch
import torch.nn as nn
import torchvision.models as tvm

WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resnet18-f37072fd.pth")


class VisualBackbone(nn.Module):
    """[B, 3, 224, 224] -> [B, 49, hidden_dim] spatial tokens.

    The 7x7 map before global pooling is kept so attention can weight regions of the frame.
    """

    def __init__(self, hidden_dim: int = 128, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        resnet = tvm.resnet18(weights=None)
        if pretrained:
            if os.path.exists(WEIGHTS_FILE):
                resnet.load_state_dict(torch.load(WEIGHTS_FILE, map_location="cpu"))
            else:  # downloads once into the torch cache
                resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # -> [B, 512, 7, 7]
        self.freeze = freeze
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(512, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, 49, hidden_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.norm = nn.LayerNorm(hidden_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:  # keep frozen BatchNorm statistics fixed
            self.features.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)                      # [B, 512, 7, 7]
        f = f.flatten(2).transpose(1, 2)          # [B, 49, 512]
        return self.norm(self.proj(f) + self.pos)  # [B, 49, hidden]
