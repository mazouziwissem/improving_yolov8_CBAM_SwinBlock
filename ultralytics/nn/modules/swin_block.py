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
# ultralytics/nn/modules/swin_block.py

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


def window_partition(x, window_size):
    # x: [B, H, W, C]
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)
    return windows  # [num_windows*B, window_size*window_size, C]


def window_reverse(windows, window_size, H, W):
    # windows: [num_windows*B, window_size*window_size, C]
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x  # [B, H, W, C]


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=2, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        assert 0 <= shift_size < window_size, "shift_size must be in [0, window_size)"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Padding
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h))  # pad last two dims
        Hp, Wp = x.shape[2], x.shape[3]

        x = rearrange(x, 'b c h w -> b h w c')  # to BHWC

        # Shift if shift_size > 0
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Partition windows
        x_windows = window_partition(x, self.window_size)  # [num_windows*B, ws*ws, C]

        # Attention + MLP
        x_windows = self.norm1(x_windows)
        attn_windows, _ = self.attn(x_windows, x_windows, x_windows)
        x_windows = x_windows + attn_windows
        x_windows = x_windows + self.mlp(self.norm2(x_windows))

        # Merge windows
        x = window_reverse(x_windows, self.window_size, Hp, Wp)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = rearrange(x, 'b h w c -> b c h w')  # back to BCHW

        return x[:, :, :H, :W]  # remove padding


