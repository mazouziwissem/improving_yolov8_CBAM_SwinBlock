import torch
import torch.nn as nn
from ultralytics.nn.modules import GhostConv, Concat

class GhostC2f(nn.Module):
    """GhostC2f module like C2f but using GhostConv"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c1, c_, 1, 1)
        self.cv3 = GhostConv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[GhostConv(c_, c_, 3, 1) for _ in range(n)])
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y1 = self.m(y1)
        y = torch.cat((y1, y2), dim=1)
        out = self.cv3(y)
        if self.shortcut:
            out = out + x
        return out
