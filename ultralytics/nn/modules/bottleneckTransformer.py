import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, head_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads

        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.proj = nn.Linear(self.inner_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (b, hw, c)

        # Apply LayerNorm
        x_ln = self.norm1(x_flat)
        qkv = self.qkv(x_ln).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn_scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, -1, self.inner_dim)
        out = self.proj(out)

        # Add & Norm
        x = x_flat + out
        x = x + self.ffn(self.norm2(x))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x
