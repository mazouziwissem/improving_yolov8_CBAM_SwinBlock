import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv 


class BiFPNBlock(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super(BiFPNBlock, self).__init__()

        self.eps = eps
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))

        self.conv3_up = Conv(channels, channels, 3, 1)
        self.conv4_up = Conv(channels, channels, 3, 1)
        self.conv4_down = Conv(channels, channels, 3, 1)
        self.conv5_down = Conv(channels, channels, 3, 1)

    def forward(self, inputs):
        p3, p4, p5 = inputs

        # Upsample pathway
        w = F.relu(self.w1)
        w = w / (torch.sum(w, dim=0) + self.eps)
        p4_td = self.conv4_up(w[0] * p4 + w[1] * F.interpolate(p5, scale_factor=2, mode="nearest"))
        p3_td = self.conv3_up(w[0] * p3 + w[1] * F.interpolate(p4_td, scale_factor=2, mode="nearest"))

        # Downsample pathway
        w = F.relu(self.w2)
        w = w / (torch.sum(w, dim=0) + self.eps)
        p4_out = self.conv4_down(w[0] * p4 + w[1] * p4_td + w[2] * F.max_pool2d(p3_td, 2))
        p5_out = self.conv5_down(w[0] * p5 + w[1] * p5 + w[2] * F.max_pool2d(p4_out, 2))

        return [p3_td, p4_out, p5_out]

class BiFPN(nn.Module):
    def __init__(self, channels=256, num_blocks=2):
        super(BiFPN, self).__init__()
        self.blocks = nn.Sequential(*[BiFPNBlock(channels) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x