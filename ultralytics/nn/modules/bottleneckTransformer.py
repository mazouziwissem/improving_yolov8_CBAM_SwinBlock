import torch
import torch.nn as nn
import torch.nn.functional as F



class BottleneckTransformer(nn.Module):
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, self.inner_dim * 3)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(self.inner_dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.shape

        assert C == self.norm1.normalized_shape[0], f"Expected input channels {self.norm1.normalized_shape[0]}, got {C}"

        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, -1, self.inner_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out + x  # Residual

        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out



