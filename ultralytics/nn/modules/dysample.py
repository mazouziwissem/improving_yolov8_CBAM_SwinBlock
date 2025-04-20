import torch
import torch.nn as nn
import torch.nn.functional as F

class Dysample(nn.Module):
    def __init__(self, in_channels, out_channels=None, scale_factor=2):
        super(Dysample, self).__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.attn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.scale = scale_factor

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        attn = self.attn(x)
        return x * attn
