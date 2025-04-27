import torch
import torch.nn as nn
from torch.nn import functional as F

class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, ratio=2):
        super().__init__()
        c_ = max(c2 // ratio, 1)  # Garantit au moins 1 canal
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c_, k, s, k//2, groups=g, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(),
            nn.Conv2d(c_, c_, 5, 1, 2, groups=max(c_, 1), bias=False),  # Groups >=1
            nn.BatchNorm2d(c_),
            nn.SiLU(),
        )
        self.shortcut = nn.Conv2d(c1, c2, 1, 1, 0) if c1 != c2 else nn.Identity()
        
    def forward(self, x):
        return torch.cat([self.conv(x), self.shortcut(x)], 1)

class BoTBlock(nn.Module):
    def __init__(self, c1, c2, num_heads=4, expansion=4):
        super().__init__()
        c_ = int(c2 * expansion)
        self.conv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.conv2 = nn.Conv2d(c_, c2, 1, 1, 0)
        self.attn = nn.MultiheadAttention(c2, num_heads)
        self.norm = nn.LayerNorm(c2)
        
    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(2, 0, 1)  # (H*W, B, C)
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = attn_out.permute(1, 2, 0).view(b, c, h, w)
        return self.conv2(x) + shortcut

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, dim*dim_scale**2, bias=False)
        self.norm = nn.LayerNorm(dim // dim_scale)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.expand(x)  # (B, H*W, C*4)
        x = x.view(B, H, W, 4*C).permute(0, 3, 1, 2)  # (B, 4C, H, W)
        x = F.pixel_shuffle(x, 2)  # (B, C, 2H, 2W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class C2f_Faster(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = max(int(c2 * e), 1)  # Canal minimal = 1
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv((2 + n) * c_, c2, 1)
        self.m = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c_, c_, 1, groups=max(c_, 1)),  # Groups dynamique
                nn.BatchNorm2d(c_),
                nn.SiLU()
            ) for _ in range(n))
        
    def forward(self, x):
        y = list(self.cv1(x).split((self.cv1.out_channels // (2 + len(self.m)),)* (2 + len(self.m))))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# Version Ghost de SPPF
class SPPF_Ghost(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))