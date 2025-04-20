import torch
import torch.nn as nn

class C2PSS(nn.Module):
    def __init__(self, c1, c2=None):
        super(C2PSS, self).__init__()
        c2 = c2 or c1
        self.cv1 = nn.Conv2d(c1, c2 // 2, 1, 1)
        self.cv2 = nn.Conv2d(c1, c2 // 2, 3, 1, 1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        out = torch.cat((x1, x2), dim=1)
        attn = self.attn(out)
        return out * attn
