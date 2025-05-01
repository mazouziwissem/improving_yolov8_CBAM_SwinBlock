import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv  # Ultralytics custom Conv
from ultralytics.nn.modules.block import Bottleneck  # Import Bottleneck

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, c, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // reduction, c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class C2f_SE(nn.Module):
    """C2f block with Squeeze-and-Excitation (SE) attention."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        #self.cv2 = Conv((2 + n) * c_, c2, 1, 1)
        self.cv2 = Conv(int((2 + n) * c_), c2, 1, 1)

        self.m = nn.ModuleList([
            Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)
        ])
        self.se = SEBlock(c2)

    def forward(self, x):
        y1, y2 = self.cv1(x).chunk(2, 1)
        outs = [y1, y2]
        for i in self.m:
            y2 = i(y2)
            outs.append(y2)
        return self.se(self.cv2(torch.cat(outs, 1)))
