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
import torch.nn.functional as F
from einops import rearrange

class WindowAttention(nn.Module):
    """Attention avec décalage de fenêtre optimisée pour YOLO"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Masque d'attention dynamique
        self.register_buffer("relative_index", self.create_relative_index(window_size))
        self.relative_bias = nn.Parameter(torch.randn(num_heads, (2 * window_size - 1) ** 2))

    def create_relative_index(self, size):
        coords = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size)))
        return (coords[0] - coords[1]) + size - 1

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn += self.relative_bias[:, self.relative_index.flatten()].view(
            self.num_heads, self.window_size**2, self.window_size**2)
        
        if mask is not None:
            attn += mask.unsqueeze(1)
            
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        return self.proj(x)

class SwinBlock(nn.Module):
    """Version optimisée pour l'intégration dans YOLO"""
    def __init__(self, channels, window_size=7, num_heads=4, shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.num_heads = num_heads
        
        self.norm1 = nn.LayerNorm(channels)
        self.attn = WindowAttention(channels, window_size, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        
        if self.shift:
            self.register_buffer('attention_mask', self.create_mask(window_size))

    def create_mask(self, window_size):
        mask = torch.zeros(window_size**2, window_size**2)
        shift = window_size // 2
        for i in range(window_size):
            for j in range(window_size):
                if (i < shift and j < shift) or (i >= shift and j >= shift):
                    mask[i*window_size+j, :] = 0
                else:
                    mask[i*window_size+j, :] = -100
        return mask.unsqueeze(0)

    def window_partition(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H//self.window_size, self.window_size, W//self.window_size, self.window_size)
        return rearrange(x, 'b c h w1 h w2 -> (b h w) (w1 w2) c')

    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size**2))
        return rearrange(windows, '(b h w) (w1 w2) c -> b c (h w1) (w w2)', 
                         b=B, h=H//self.window_size, w1=self.window_size)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Shift des fenêtres
        if self.shift:
            x = torch.roll(x, shifts=(self.window_size//2, self.window_size//2), dims=(2,3))
        
        # Partitionnement des fenêtres
        shortcut = x
        x = self.norm1(x.permute(0,2,3,1))
        windows = self.window_partition(x)
        
        # Attention
        if self.shift:
            attn = self.attn(windows, self.attention_mask)
        else:
            attn = self.attn(windows)
        
        # Fusion des fenêtres
        x = self.window_reverse(attn, H, W)
        x = shortcut + x.permute(0,3,1,2)
        
        # MLP
        x = x + self.mlp(self.norm2(x.permute(0,2,3,1))).permute(0,3,1,2)
        
        return x
