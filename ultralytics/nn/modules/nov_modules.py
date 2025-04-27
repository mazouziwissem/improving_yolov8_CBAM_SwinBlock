# botnet.py
import torch
import torch.nn as nn

class MHSA(nn.Module):
    """ Multi-Head Self Attention for BoTNet """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, HW, C//heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out)

        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return out


class BoTBlock(nn.Module):
    """ BoTNet Block: Bottleneck + MHSA """
    def __init__(self, c1, c2, stride=1, heads=4):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2 // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2 // 4)

        self.mhsa = MHSA(c2 // 4, num_heads=heads) if stride == 1 else nn.Conv2d(c2 // 4, c2 // 4, 3, stride, 1)

        self.conv2 = nn.Conv2d(c2 // 4, c2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

        self.downsample = None
        if stride != 1 or c1 != c2:
            self.downsample = nn.Sequential(
                nn.Conv2d(c1, c2, 1, stride, bias=False),
                nn.BatchNorm2d(c2),
            )
        
        self.act = nn.SiLU()

    def forward(self, x):
        identity = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.mhsa(out)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out
