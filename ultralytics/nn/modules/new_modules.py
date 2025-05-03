import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from ultralytics.nn.modules.block import Bottleneck  # For SwinBlock

class CBAM(nn.Module):
    """Fixed CBAM with dimension validation"""
    def __init__(self, c1, reduction=16):
        super().__init__()
        if c1 // reduction == 0:
            raise ValueError(f"Channel reduction too aggressive for {c1} channels (reduction={reduction})")
        
        self.channels = c1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Channel attention with dimension validation
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // reduction, c1)
        )
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Channel attention
        b, c, h, w = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        
        return x * channel * spatial

class CoordAttention(nn.Module):
    """Coordinate Attention (from CA-Net)"""
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1)
        self.conv2 = nn.Conv2d(channels // reduction, channels, 1)

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        
        # X-direction
        x_h = self.pool_h(x)
        # Y-direction 
        x_w = self.pool_w(x).permute(0,1,3,2)
        
        # Concatenate and encode
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = F.relu(y)
        
        # Split and transform
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0,1,3,2)
        
        # Attention maps
        att = torch.sigmoid(self.conv2(x_h + x_w))
        return identity * att

# --------------------- 2. Transformer Block ---------------------
class SwinBlock(nn.Module):
    """Simplified Swin Transformer Block"""
    def __init__(self, embed_dim, num_heads=4, window_size=7):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.window_size = window_size

    def window_partition(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, H // self.window_size, self.window_size, 
                   W // self.window_size, self.window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return windows.view(-1, C, self.window_size, self.window_size)

    def forward(self, x):
        B, C, H, W = x.size()
        residual = x
        
        # Window-based attention
        x = self.window_partition(x)
        x = x.flatten(2).permute(2, 0, 1)  # [N, B, C]
        x = self.attn(x, x, x)[0]
        x = x.permute(1, 2, 0).view(B, C, H, W)
        
        # Residual connection
        x = residual + x
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm1(x)
        
        # MLP
        residual = x
        x = self.mlp(x)
        x = residual + x
        x = self.norm2(x)
        return x.permute(0, 3, 1, 2)

# --------------------- 3. ASPP Module ---------------------
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, 
                          dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

# --------------------- 4. C2f Modification ---------------------
class C2f(nn.Module):
    """YOLOv8-compatible C2f with attention support"""
    def __init__(self, c1, c2, n=1, shortcut=False, attention=False):
        super().__init__()
        self.c = int(c2 * 0.5)  # hidden channels
        self.conv = nn.Conv2d(c1, 2 * self.c, 1)
        self.bottlenecks = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g=1, k=(3, 3), e=1.0) for _ in range(n)
        )
        self.attention = CBAM(2 * self.c) if attention else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x1, x2 = x.chunk(2, 1)
        for bottleneck in self.bottlenecks:
            x2 = bottleneck(x2)
        x = torch.cat([x1, x2], 1)
        return self.attention(x)