import torch
import torch.nn as nn
import torch.nn.functional as F



# class BottleneckTransformer(nn.Module):
#     def __init__(self, dim, num_heads, head_dim):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = head_dim
#         self.inner_dim = num_heads * head_dim

#         self.norm1 = nn.LayerNorm(dim)
#         self.qkv = nn.Linear(dim, self.inner_dim * 3)
#         self.attn_drop = nn.Dropout(0.1)
#         self.proj = nn.Linear(self.inner_dim, dim)
#         self.proj_drop = nn.Dropout(0.1)

#     def forward(self, x):
#         B, C, H, W = x.shape

#         assert C == self.norm1.normalized_shape[0], f"Expected input channels {self.norm1.normalized_shape[0]}, got {C}"

#         x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

#         x_norm = self.norm1(x)
#         qkv = self.qkv(x_norm).chunk(3, dim=-1)
#         q, k, v = map(lambda t: t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2), qkv)

#         attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         out = (attn @ v).transpose(1, 2).reshape(B, -1, self.inner_dim)
#         out = self.proj(out)
#         out = self.proj_drop(out)
#         out = out + x  # Residual

#         out = out.permute(0, 2, 1).view(B, C, H, W)
#         return out

class BottleneckTransformer(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        # Input projection - this needs to match the input channel size
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(0.1)
        )
        
        # Layer normalization for input
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1)
        )
        
        # Optional: spatial positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, dim))
        
    def forward(self, x):
        # Input shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Reshape and transpose to [B, HW, C]
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        
        # Add positional embedding
        x_flat = x_flat + self.pos_embed
        
        # Apply layer norm
        x_norm = self.norm1(x_flat)
        
        # Self-attention
        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, H * W, self.heads, -1).transpose(1, 2), qkv)
        
        # Attention computation
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v).transpose(1, 2)
        out = out.reshape(B, H * W, self.inner_dim)
        out = self.to_out(out)
        
        # First residual connection
        x_res = x_flat + out
        
        # Second norm and MLP
        y = self.norm2(x_res)
        y = self.mlp(y)
        
        # Second residual connection
        out = x_res + y
        
        # Reshape back to [B, C, H, W]
        out = out.permute(0, 2, 1).view(B, C, H, W)
        
        return out