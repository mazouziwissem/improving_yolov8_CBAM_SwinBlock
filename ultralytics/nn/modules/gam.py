import torch
import torch.nn as nn
import torch.nn.functional as F


class GAM(nn.Module):
    """
    Global Attention Mechanism (GAM) module.
    Applies both channel and spatial attention to enhance feature maps.
    """

    def __init__(self, c=None):
        super().__init__()
        self.c = c
        if c:
            self._build(c)

    def _build(self, c):
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 8, c, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if not hasattr(self, 'channel_attn'):
            self._build(x.shape[1])

        # Channel attention
        ca = self.channel_attn(x)
        x_ca = x * ca

        # Spatial attention
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa_input = torch.cat((avg_pool, max_pool), dim=1)
        sa = self.spatial_attn(sa_input)
        x_out = x_ca * sa

        return x_out
