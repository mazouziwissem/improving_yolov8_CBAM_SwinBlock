import torch
import torch.nn as nn

class ECBAM(nn.Module):
    """Efficient Channel and Spatial Attention Module"""
    def __init__(self, c1, r=16):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Make sure c1 is used as input and output channels
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1 // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // r, c1, kernel_size=1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        
        # Spatial Attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = self.sigmoid_channel(avg_out + max_out)
        x = x * ca
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.sigmoid_spatial(self.conv(sa))
        
        return x * sa
