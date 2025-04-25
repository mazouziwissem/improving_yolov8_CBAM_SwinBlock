# ultralytics/nn/modules/swin_block.py
import torch
import torch.nn as nn
from einops import rearrange

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
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
        x = rearrange(x, 'b c h w -> b (h w) c')
        x1 = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + x1
        x = x + self.mlp(self.norm2(x))
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x
