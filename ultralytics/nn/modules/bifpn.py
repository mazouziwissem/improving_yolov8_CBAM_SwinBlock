# ultralytics/nn/modules/bifpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiFPN(nn.Module):
    def __init__(self, channels=[256, 512, 1024], out_channels=256, num_repeats=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels[0], out_channels, 1)
        self.conv2 = nn.Conv2d(channels[1], out_channels, 1)
        self.conv3 = nn.Conv2d(channels[2], out_channels, 1)

        self.out_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # Suppose x = [P3, P4, P5]
        p3, p4, p5 = x

        p4_up = F.interpolate(p5, scale_factor=2, mode='nearest') + self.conv2(p4)
        p3_up = F.interpolate(p4_up, scale_factor=2, mode='nearest') + self.conv1(p3)

        p4_down = F.max_pool2d(p3_up, 2) + p4_up
        p5_down = F.max_pool2d(p4_down, 2) + self.conv3(p5)

        return [self.out_conv(p3_up), self.out_conv(p4_down), self.out_conv(p5_down)]
