import torch
import torch.nn as nn
import torch.nn.functional as F

class SAM(nn.Module):
    def __init__(self, in_channels):
        super(SAM, self).__init__()
        
        # Channel attention: average pooling and max pooling along spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        
        # 1x1 convolution for generating the attention map (after pooling)
        self.conv = nn.Conv2d(in_channels * 2, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Apply average and max pooling along spatial dimensions
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        
        # Concatenate both pooled features along the channel axis
        combined = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid to generate attention map
        attention_map = self.conv(combined)
        attention_map = self.sigmoid(attention_map)
        
        # Scale the input feature map with the attention map
        out = x * attention_map
        return out


