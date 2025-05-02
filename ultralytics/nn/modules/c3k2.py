import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

class C3K2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=4):
        super().__init__()
        self.g = g
        
        # Calcul sécurisé des canaux
        self.c_ = max((c2 // (2 * g)) * g, g)  # Garantit c_ >= g et divisible par g
        
        self.cv1 = Conv(c1, self.c_, 1, 1)
        self.cv2 = Conv(c1, self.c_, 1, 1)
        
        self.m = nn.Sequential(*[
            nn.Sequential(
                Conv(self.c_, self.c_, 3, 1, g=self.g),
                Conv(self.c_, self.c_, 1, 1)
            ) for _ in range(n)])
        
        self.cv3 = Conv(2 * self.c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], 1))