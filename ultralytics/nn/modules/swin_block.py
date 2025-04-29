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
# import torch
# import torch.nn as nn

# class SwinBlock(nn.Module):
#     def __init__(self, c1=None, window_size=8):
#         super().__init__()
#         self.c1 = c1
#         self.window_size = window_size
        
#         # Les couches seront créées lors du premier forward pass si c1 n'est pas spécifié
#         self.initialized = c1 is not None
        
#         if self.initialized:
#             self._initialize_layers(c1)
    
#     def _initialize_layers(self, c1):
#         # Version simplifiée du bloc SwinTransformer adaptée pour YOLOv8
#         self.conv = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
#         self.bn = nn.BatchNorm2d(c1)
#         self.act = nn.SiLU()
        
#         # Attention simplifiée
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c1, c1, kernel_size=1),
#             nn.Sigmoid()
#         )
        
#         self.initialized = True
        
#     def forward(self, x):
#         # Si les couches ne sont pas encore initialisées, on les crée
#         if not self.initialized:
#             _, c, _, _ = x.shape
#             self._initialize_layers(c)
        
#         shortcut = x
        
#         # Convolution standard
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.act(x)
        
#         # Mécanisme d'attention simplifié
#         att = self.attention(x)
#         x = x * att
        
#         return x + shortcut


#==========================================

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SwinBlock(nn.Module):
#     # def __init__(self, c1, c2=None, num_heads=8, window_size=7, shift_size=0):
#     #     """
#     #     Swin Transformer Block.
        
#     #     Args:
#     #     - c1 (int): The number of input channels (e.g., 512)
#     #     - c2 (int, optional): This can be used to modify the output channels, or None if we use c1 as the output channels as well.
#     #     - num_heads (int): Number of attention heads for multi-head self-attention.
#     #     - window_size (int): Size of the local window for attention.
#     #     - shift_size (int): Size of the shift to apply for the windowed attention.
#     #     """
#     #     super(SwinBlock, self).__init__()

#     #     # Set default values for c2 if not provided
#     #     if c2 is None:
#     #         c2 = c1

#     #     self.c1 = c1
#     #     self.c2 = c2

#     #     # Attention Layer - this can be a window-based self-attention mechanism
#     #     self.attn = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)

#     #     # Feed-forward network (position-wise)
#     #     self.ffn = nn.Sequential(
#     #         nn.Linear(c1, c1 * 4),  # First projection to higher dimension
#     #         nn.ReLU(),
#     #         nn.Linear(c1 * 4, c1)   # Project back to c1
#     #     )

#     #     # Layer normalization and residual connections
#     #     self.norm1 = nn.LayerNorm(c1)
#     #     self.norm2 = nn.LayerNorm(c1)

#     #     self.window_size = window_size
#     #     self.shift_size = shift_size
#     def __init__(self, c):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(c)
#         self.attn = nn.MultiheadAttention(embed_dim=c, num_heads=4, batch_first=True)
#         self.norm2 = nn.LayerNorm(c)
#         self.ffn = nn.Sequential(
#             nn.Linear(c, c * 4),
#             nn.GELU(),
#             nn.Linear(c * 4, c)
#         )
#     # def forward(self, x):
#     #     """
#     #     Forward pass of the Swin Transformer Block.
        
#     #     Args:
#     #     - x (Tensor): Input tensor with shape [batch_size, seq_len, c1]
        
#     #     Returns:
#     #     - out (Tensor): Output tensor after applying Swin Transformer Block
#     #     """
#     #     # Input x shape [batch_size, seq_len, c1]
#     #     residual = x
        
#     #     # Apply the first layer normalization (for attention)
#     #     x = self.norm1(x)
        
#     #     # Attention mechanism - compute self-attention
#     #     attn_output, _ = self.attn(x, x, x)
        
#     #     # Add the residual connection
#     #     x = residual + attn_output
        
#     #     # Apply second normalization (for Feed Forward)
#     #     residual = x
#     #     x = self.norm2(x)
        
#     #     # Feed Forward Network (Position-wise)
#     #     x = self.ffn(x)
        
#     #     # Add the residual connection
#     #     out = residual + x
        
#     #     return out

#     def forward(self, x):
#         B, C, H, W = x.shape

#         # Flatten spatial dimensions and permute to shape (B, N, C)
#         x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]

#         # LayerNorm and Multi-head Attention
#         residual = x_flat
#         x_norm = self.norm1(x_flat)
#         attn_output, _ = self.attn(x_norm, x_norm, x_norm)
#         x_attn = residual + attn_output

#         # Feed-forward
#         residual = x_attn
#         x_norm = self.norm2(x_attn)
#         x_ffn = self.ffn(x_norm)
#         out = residual + x_ffn

#         # Reshape back to [B, C, H, W]
#         out = out.permute(0, 2, 1).view(B, C, H, W)

#         return out

# ==============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        """
        x: Tensor [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Réorganise en [B, H*W, C] pour le transformer
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Normalisation + attention
        x_res = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x_res + attn_output

        # MLP
        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)

        # Revenir au format [B, C, H, W]
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x
