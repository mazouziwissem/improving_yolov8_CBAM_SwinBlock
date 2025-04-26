import torch
import torch.nn as nn
import torch.nn.functional as F

class MSDA(nn.Module):
    """Multi-Scale Dilated Attention Block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 2, 4], reduction_ratio=16):
        super().__init__()
        self.dilations = dilations
        
        # Dilated Convolution Branches
        self.conv_branches = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=d * (kernel_size-1)//2, dilation=d)
            for d in dilations
        ])
        
        # Channel Attention (Squeeze-and-Excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // reduction_ratio, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Multi-Scale Feature Fusion
        branch_outputs = [conv(x) for conv in self.conv_branches]
        fused = torch.stack(branch_outputs, dim=0).sum(dim=0)
        
        # Channel Attention
        channel_weights = self.channel_attention(fused)
        channel_refined = fused * channel_weights
        
        # Spatial Attention
        spatial_weights = self.spatial_attention(channel_refined.mean(dim=1, keepdim=True))
        refined = channel_refined * spatial_weights
        
        return refined