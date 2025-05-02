import torch
import torch.nn as nn

class C3K2(nn.Module):
    """Lightweight C3 block with kernel factorization for medical imaging"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=4):
        super().__init__()
        self.c = c2 // 2  # Split channels
        self.cv1 = nn.Conv2d(c1, 2*self.c, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c1, 2*self.c, 1, 1, bias=False)
        
        # Depthwise separable convolutions with kernel factorization
        self.m = nn.Sequential(*[
            nn.Sequential(
                # Grouped spatial convolution (K=3)
                nn.Conv2d(self.c, self.c, 3, 1, 1, groups=g, bias=False),
                # Pointwise convolution (K=1)
                nn.Conv2d(self.c, self.c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.c),
                nn.SiLU()
            ) for _ in range(n)])
        
        self.cv3 = nn.Conv2d(2*self.c, c2, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = self.cv1(x).chunk(2, 1)
        return self.cv3(torch.cat((
            self.m(x1),
            self.cv2(x2)
        ), 1))