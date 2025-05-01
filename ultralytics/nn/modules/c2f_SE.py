import torch
import torch.nn as nn

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
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
        w = self.pool(x)
        w = self.fc(w)
        return x * w


# C2f Block with SE integrated
class C2f_SE(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        c1: input channels
        c2: output channels
        n: number of bottlenecks
        shortcut: use shortcut connections
        g: number of groups in convolutions
        e: expansion ratio
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, 2 * c_, 1, 1, bias=False)
        self.cv2 = nn.Conv2d((2 + n) * c_, c2, 1, 1, bias=False)
        self.m = nn.Sequential(*[
            Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)
        ])
        self.se = SEBlock(c2)

    def forward(self, x):
        y1, y2 = self.cv1(x).chunk(2, 1)
        y = [y1, y2] + [self.m(y2)]
        out = self.cv2(torch.cat(y, 1))
        return self.se(out)


# Supporting bottleneck block used inside C2f
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c_, c2, 3, 1, 1, groups=g, bias=False)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
