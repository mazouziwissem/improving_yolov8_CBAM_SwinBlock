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

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        qkv = self.qkv(x).chunk(3, dim=1)
        
        q, k, v = map(lambda t: rearrange(t, 
            'b (h d) (ws_h hh) (ws_w ww) -> b h (hh ww) (ws_h ws_w) d',
            h=self.num_heads, 
            ws_h=self.ws, 
            ws_w=self.ws), qkv)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn += mask
            
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        
        x = rearrange(x, 
            'b h (hh ww) (ws_h ws_w) d -> b (h d) (hh ws_h) (ww ws_w)',
            hh=H//self.ws, 
            ws_h=self.ws,
            h=self.num_heads)
            
        return self.proj(x)

class SwinBlock(nn.Module):
    def __init__(self, base_channels, window_size=8, num_heads=4, shift=False):
        super().__init__()
        self.channels = base_channels  # Pas de scale supplémentaire
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift = shift
        
        self.norm1 = nn.BatchNorm2d(self.channels)
        self.attn = WindowAttention(self.channels, window_size, num_heads)
        self.norm2 = nn.BatchNorm2d(self.channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(self.channels * 4, self.channels, 1)
        )
        
        if self.shift:
            self.register_buffer('mask', self.create_mask(window_size))

    def create_mask(self, window_size):
        mask = torch.zeros(window_size**2, window_size**2)
        shift = window_size // 2
        for i in range(window_size):
            for j in range(window_size):
                if (i < shift and j < shift) or (i >= shift and j >= shift):
                    mask[i*window_size+j, :] = 0
                else:
                    mask[i*window_size+j, :] = -100
        return mask.unsqueeze(0).unsqueeze(0)

    def window_partition(self, x):
        B, C, H, W = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, \
            f"Dimensions ({H}x{W}) must be divisible by window_size ({self.window_size})"
        x = x.view(B, C, H//self.window_size, self.window_size, W//self.window_size, self.window_size)
        return rearrange(x, 'b c h wh w ww -> b (h w) (wh ww) c')

    def window_reverse(self, windows, H, W):
        B = windows.shape[0] // (H * W // self.window_size**2)
        return rearrange(windows, 
            'b (h w) (wh ww) c -> b c (h wh) (w ww)',
            h=H//self.window_size,
            w=W//self.window_size,
            wh=self.window_size)

    def forward(self, x):
        B, C, H, W = x.shape
        
        if self.shift:
            x = torch.roll(x, shifts=(self.window_size//2,)*2, dims=(2,3))
        
        # Attention
        shortcut = x
        x = self.norm1(x)
        windows = self.window_partition(x)
        attn = self.attn(windows, self.mask if self.shift else None)
        x = self.window_reverse(attn, H, W)
        x = shortcut + x
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x