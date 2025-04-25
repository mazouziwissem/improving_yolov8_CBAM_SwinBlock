# ultralytics/nn/modules/swin_block.py
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


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
# Fichier: /kaggle/working/yolo_MD/ultralytics/nn/modules/custom_swin_block.py



class SwinBlock(nn.Module):
    """
    SwinBlock adapté qui utilise correctement les dimensions d'entrée.
    """
    def __init__(self, dim, window_size=7):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        # Utiliser la dimension correcte passée en paramètre
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Second bloc de normalisation et MLP
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)  # Supposer une image carrée pour simplifier
        
        # Transformer de (B, N, C) à (B, H, W, C)
        x = x.reshape(B, H, W, C)
        
        # Créer les fenêtres
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        Hp = H + pad_h
        Wp = W + pad_w
        
        # Partitionner en fenêtres non chevauchantes
        x = x.reshape(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size * self.window_size, C)
        
        # Self-attention simplifiée dans chaque fenêtre
        shortcut = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(-1, self.window_size * self.window_size, 3, C).permute(0, 2, 1, 3)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Calcul d'attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (C ** 0.5))
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v)
        
        # Projection et résidu
        x = x @ self.proj.weight.t() + self.proj.bias
        x = shortcut + x
        
        # MLP et second résidu
        x = x + self.mlp(self.norm2(x))
        
        # Reconstruire la forme spatiale
        x = x.reshape(-1, Hp // self.window_size, Wp // self.window_size, 
                      self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        
        # Supprimer le padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        # Retourner à la forme (B, N, C)
        x = x.reshape(B, H * W, C)
        
        return x