import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.ghost_channels = out_channels // 2
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.ghost_channels, k, s, autopad(k, p), groups=g, bias=False),
            nn.BatchNorm2d(self.ghost_channels),
            nn.SiLU() if act else nn.Identity()
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.ghost_channels, self.ghost_channels, 5, 1, 2, groups=self.ghost_channels, bias=False),
            nn.BatchNorm2d(self.ghost_channels),
            nn.SiLU() if act else nn.Identity()
        )

    def forward(self, x):
        y = self.primary_conv(x)
        return torch.cat([y, self.cheap_operation(y)], 1)


def autopad(k, p=None):  # auto-padding
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class GhostSPPF(nn.Module):
    """ Ghost SPPF: Spatial Pyramid Pooling Fast with Ghost Convs """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # Intermediate channels
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c_ * 4, c2, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))
