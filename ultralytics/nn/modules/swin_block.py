# # ultralytics/nn/modules/swin_block.py
# import torch
# import torch.nn as nn
# from einops import rearrange
# import torch.nn.functional as F


# def window_partition(x, window_size):
#     # x: [B, H, W, C]
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)
#     return windows  # shape: [num_windows*B, window_size*window_size, C]

# def window_reverse(windows, window_size, H, W):
#     # windows: [num_windows*B, window_size*window_size, C]
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
#     return x  # shape: [B, H, W, C]


# class SwinBlock(nn.Module):
#     def __init__(self, dim, num_heads=2, window_size=7):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.GELU(),
#             nn.Linear(dim * 4, dim)
#         )

#     def forward(self, x):
#         B, C, H, W = x.shape

#         # padding if not divisible by window_size
#         pad_h = (self.window_size - H % self.window_size) % self.window_size
#         pad_w = (self.window_size - W % self.window_size) % self.window_size
#         x = F.pad(x, (0, pad_w, 0, pad_h))  # pad last two dims

#         Hp, Wp = x.shape[2], x.shape[3]

#         x = rearrange(x, 'b c h w -> b h w c')
#         x_windows = window_partition(x, self.window_size)  # [num_windows*B, ws*ws, C]

#         x_windows = self.norm1(x_windows)
#         attn_windows, _ = self.attn(x_windows, x_windows, x_windows)  # [num_windows*B, ws*ws, C]
#         x_windows = x_windows + attn_windows
#         x_windows = x_windows + self.mlp(self.norm2(x_windows))

#         x = window_reverse(x_windows, self.window_size, Hp, Wp)  # [B, H', W', C]
#         x = rearrange(x, 'b h w c -> b c h w')

#         return x[:, :, :H, :W]  # remove padding


import torch
import torch.nn as nn
from einops import rearrange

def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return x

def window_reverse(windows, window_size, H, W, C):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape  # after permute

        x_windows = x.view(B, H // self.window_size, self.window_size,
                           W // self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size ** 2, C)

        qkv = self.qkv(x_windows).reshape(-1, self.window_size**2, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(-1, self.window_size**2, C)
        out = self.proj(out)

        # Reverse windows
        out = out.view(B, H // self.window_size, W // self.window_size,
                       self.window_size, self.window_size, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        return out


class SwinBlock(nn.Module):
    def __init__(self, channels, window_size=8, num_heads=4, shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm1 = nn.BatchNorm2d(channels)
        self.attn = WindowAttention(channels, window_size, num_heads)
        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        if self.shift:
            x = torch.roll(x, shifts=(-self.window_size // 2,) * 2, dims=(2, 3))

        shortcut = x
        x = self.norm1(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C) for Linear
        x = self.attn(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W) back
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))
        return x
