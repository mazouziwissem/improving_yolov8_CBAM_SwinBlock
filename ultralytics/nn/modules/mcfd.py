import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class MCFD(nn.Module):
    """Simplified Multi-Context Feature Distillation module for tumor detection"""
    def __init__(self, c1, c2):
        super().__init__()
        # Make sure c1 and c2 match to avoid dimension issues
        c2 = c1 if c2 != c1 else c2
        
        # Context extraction with simpler structure
        self.conv1 = Conv(c1, c1, k=3, p=1)
        self.conv2 = Conv(c1, c1, k=3, p=2, d=2)  # Dilated for larger context
        
        # Channel attention for feature emphasis
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c1, c1 // 4, k=1),
            nn.SiLU(),
            Conv(c1 // 4, c1, k=1),
            nn.Sigmoid()
        )
        
        # Output projection to maintain channel dimensions
        self.proj = Conv(c1, c2, k=1)
        
    def forward(self, x):
        # Apply standard convolution
        conv1_out = self.conv1(x)
        
        # Apply dilated convolution for context
        conv2_out = self.conv2(x)
        
        # Combine features
        combined = conv1_out + conv2_out
        
        # Apply channel attention
        attention = self.channel_attention(combined)
        enhanced = combined * attention
        
        # Final projection to maintain output dimensions
        return self.proj(enhanced)