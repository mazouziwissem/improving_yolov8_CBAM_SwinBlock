import torch
import torch.nn as nn

class ECBAM(nn.Module):
    """Efficient Channel and Spatial Attention Module with dynamic channels"""
    def __init__(self, c1, r=16):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Adaptive input channels
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
        # Ensure the number of input channels matches
        c_in = x.shape[1]  # Get input channels dynamically

        if c_in != 128:
            print(f"Warning: ECBAM expects 128 channels, but got {c_in} channels!")

        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))  # Average pooling along channels
        max_out = self.fc(self.max_pool(x))  # Max pooling along channels
        ca = self.sigmoid_channel(avg_out + max_out)
        x = x * ca  # Apply channel attention

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Mean across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max across channels
        sa = torch.cat([avg_out, max_out], dim=1)  # Concatenate the average and max
        sa = self.sigmoid_spatial(self.conv(sa))  # Apply spatial attention

        return x * sa  # Apply spatial attention to the input
