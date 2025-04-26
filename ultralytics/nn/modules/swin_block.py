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
def window_partition(x, window_size):
    """
    Partition feature map x into non-overlapping windows with size window_size x window_size
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse windows back to the original feature map
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True):
        super(SwinBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias

        # define attention and MLP
        self.attn = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

        # placeholders for LayerNorms (initialized on first forward pass)
        self.norm1 = None
        self.norm2 = None

    def forward(self, x):
        B, C, H, W = x.shape
        # initialize LayerNorms if not already
        if self.norm1 is None or self.norm2 is None:
            self.norm1 = nn.LayerNorm(C).to(x.device)
            self.norm2 = nn.LayerNorm(C).to(x.device)

        shortcut = x
        # flatten to sequence (B, HW, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)

        # reshape to windows
        x_windows = x.view(B, H, W, C)
        if self.shift_size > 0:
            x_windows = torch.roll(x_windows, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = window_partition(x_windows.permute(0, 3, 1, 2), self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, C)

        attn_windows, _ = self.attn(windows, windows, windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        attn_windows = attn_windows.permute(0, 3, 1, 2)
        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        # sequence back to (B, HW, C)
        x = x.flatten(2).transpose(1, 2)
        x = shortcut.flatten(2).transpose(1, 2) + x
        x = self.norm2(x)
        x = x + self.mlp(x)

        # back to (B, C, H, W)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

# Notes:
# - The updated SwinBlock now dynamically creates LayerNorm layers matching the channel dimension at runtime,
#   avoiding mismatches between the passed 'dim' and actual input channels.
# - Ensure the 'dim' argument in SwinBlock matches the in_channels of the preceding layer in your YAML.
# - Typical usage: SwinBlock, [64, 4, 7, 3] => dim=64, heads=4, window=7, shift=3
