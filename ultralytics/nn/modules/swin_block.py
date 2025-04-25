import torch
import torch.nn as nn
from einops import rearrange

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.dim = dim
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

        assert C == self.dim, f"Expected input with {self.dim} channels, got {C}"

        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, HW, C]

        x1, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))  # Self-attention
        x = x + x1  # Residual connection

        x = x + self.mlp(self.norm2(x))  # Feed-forward + residual
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x
