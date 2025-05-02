import torch
import torch.nn as nn



class C3K2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=4):
        super().__init__()
        c_ = int(c2 // 2)  # Conversion explicite
        
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[
            nn.Sequential(
                Conv(c_, c_, 3, 1, g=g),
                Conv(c_, c_, 1, 1)
            ) for _ in range(n)])
        
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((
            self.m(self.cv1(x)),
            self.cv2(x)
        ), 1))