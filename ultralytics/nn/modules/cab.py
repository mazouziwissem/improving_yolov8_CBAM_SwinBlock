import torch
import torch.nn as nn

class CAB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.avg_pool(x)
        ch_att = self.fc(avg_out)
        x = x * ch_att

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_pool, max_pool], dim=1)
        sp_att = self.sigmoid(self.spatial(pool))
        x = x * sp_att

        return x
