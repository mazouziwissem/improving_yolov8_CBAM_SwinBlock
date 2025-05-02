import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class C3K2(nn.Module):
    def __init__(self, c1, c2, n=1, g=4):  # Signature simplifiée
        super().__init__()
        self.c_ = (c2 // (2 * g)) * g  # Garantit la divisibilité
        
        self.cv1 = Conv(c1, self.c_, 1)
        self.cv2 = Conv(c1, self.c_, 1)
        
        self.m = nn.Sequential(*[
            nn.Sequential(
                Conv(self.c_, self.c_, 3, 1, g=g),
                Conv(self.c_, self.c_, 1)
            ) for _ in range(n)])
        
        self.cv3 = Conv(2 * self.c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat([
            self.m(self.cv1(x)),
            self.cv2(x)
        ], 1))