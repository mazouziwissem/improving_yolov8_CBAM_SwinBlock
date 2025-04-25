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
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=7, shift_size=0, 
                 mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.dim = dim  # Nous utilisons cette dimension pour la normalisation
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Correction: Utiliser la dimension d'entrée pour norm1 et norm2
        self.norm1 = norm_layer(dim)  # Utiliser dim au lieu d'une valeur codée en dur
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads, qkv_bias=True,
            attn_drop=drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)  # Utiliser dim au lieu d'une valeur codée en dur
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        # x est de forme (B, H*W, C)
        B, L, C = x.shape
        H = W = int(L**0.5)  # Supposons que l'image soit carrée pour simplifier
        
        # Redimensionner x pour obtenir la forme spatiale (B, H, W, C)
        x = x.view(B, H, W, C)
        
        # Partition de fenêtre et traitement
        x_windows = window_partition(x, self.window_size)  # [num_windows*B, ws*ws, C]
        
        # Utiliser notre normalisation avec la dimension correcte
        x_windows = self.norm1(x_windows)
        attn_windows, _ = self.attn(x_windows, x_windows, x_windows)
        x_windows = x_windows + self.drop_path(attn_windows)
        
        # Appliquer MLP
        x_windows = x_windows + self.drop_path(self.mlp(self.norm2(x_windows)))
        
        # Restaurer la forme spatiale
        x = window_reverse(x_windows, self.window_size, H, W)  # (B, H, W, C)
        
        # Retour à la forme (B, H*W, C)
        x = x.view(B, -1, C)
        
        return x

# Fonction auxiliaire pour la partition de fenêtre
def window_partition(x, window_size):
    """
    Partitionne une image en fenêtres locales non-chevauchantes.
    
    Args:
        x: (B, H, W, C)
        window_size (int): Taille de la fenêtre
    
    Returns:
        windows: (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

# Fonction auxiliaire pour reconstruire l'image à partir des fenêtres
def window_reverse(windows, window_size, H, W):
    """
    Reconstitue une image à partir de fenêtres locales.
    
    Args:
        windows: (num_windows*B, window_size*window_size, C)
        window_size (int): Taille de la fenêtre
        H (int): Hauteur de l'image
        W (int): Largeur de l'image
    
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x