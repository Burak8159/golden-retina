"""Full-frame multi-class face/eye segmentation network for v3."""
from __future__ import annotations

import torch
from torch import nn


def block(cin: int, cout: int) -> nn.Sequential:
    groups = min(8, cout)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
    )


class FaceEyeUNet(nn.Module):
    """RGB input and 12 semantic classes from the v3 Blender renderer."""

    def __init__(self, base: int = 16, classes: int = 12):
        super().__init__()
        self.enc1 = block(3, base)
        self.enc2 = block(base, base * 2)
        self.enc3 = block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.mid = block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = block(base * 2, base)
        self.head = nn.Conv2d(base, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        z = self.mid(self.pool(c))
        z = self.dec3(torch.cat((self.up3(z), c), 1))
        z = self.dec2(torch.cat((self.up2(z), b), 1))
        z = self.dec1(torch.cat((self.up1(z), a), 1))
        return self.head(z)
