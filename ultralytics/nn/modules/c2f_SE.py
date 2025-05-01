import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class C2F_SE(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # intermediate channels
        self.cv1 = Conv(c1, 2 * c_, 1, 1)  # pointwise convolution
        self.cv2 = Conv((2 + n) * c_, c2, 1, 1)  # corrected to use int kernel/stride
        self.m = nn.ModuleList([
            Conv(c_, c_, 3, 1, g=g) for _ in range(n)
        ])
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 16, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(c2 // 16, c2, 1, 1, 0),
            nn.Sigmoid()
        )
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)
        y1, y2 = y.chunk(2, 1)
        y_ = [y1]
        for m in self.m:
            y1 = m(y1)
            y_.append(y1)
        y = self.cv2(torch.cat(y_ + [y2], 1))
        s = self.se(y)
        y = y * s
        if self.shortcut:
            y = x + y
        return y
