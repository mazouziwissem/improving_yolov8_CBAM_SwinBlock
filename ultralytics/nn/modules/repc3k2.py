import torch
import torch.nn as nn
import torch.nn.functional as F

class RepC3K2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RepC3K2, self).__init__()
        # Adjusted the layer to ensure channels are flexible
        self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2x2 = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=1, padding=0)
        
        # Batch Normalization and Activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        # Apply 3x3 conv -> 2x2 conv -> BatchNorm -> ReLU
        x = self.act(self.bn(self.conv_3x3(x)))
        x = self.act(self.bn(self.conv_2x2(x)))
        return x
