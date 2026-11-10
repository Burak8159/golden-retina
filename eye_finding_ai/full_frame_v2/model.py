"""Small full-frame U-Net for semantic eye/pupil localization."""
import torch
from torch import nn


def conv_block(a: int, b: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.SiLU(inplace=True),
        nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.SiLU(inplace=True),
    )


class FullFrameUNet(nn.Module):
    def __init__(self, base: int = 16, classes: int = 7):
        super().__init__()
        self.enc1=conv_block(1,base); self.enc2=conv_block(base,base*2)
        self.enc3=conv_block(base*2,base*4); self.pool=nn.MaxPool2d(2)
        self.mid=conv_block(base*4,base*8)
        self.up3=nn.ConvTranspose2d(base*8,base*4,2,2); self.dec3=conv_block(base*8,base*4)
        self.up2=nn.ConvTranspose2d(base*4,base*2,2,2); self.dec2=conv_block(base*4,base*2)
        self.up1=nn.ConvTranspose2d(base*2,base,2,2); self.dec1=conv_block(base*2,base)
        self.head=nn.Conv2d(base,classes,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a=self.enc1(x); b=self.enc2(self.pool(a)); c=self.enc3(self.pool(b))
        z=self.mid(self.pool(c)); z=self.dec3(torch.cat((self.up3(z),c),1))
        z=self.dec2(torch.cat((self.up2(z),b),1)); z=self.dec1(torch.cat((self.up1(z),a),1))
        return self.head(z)
