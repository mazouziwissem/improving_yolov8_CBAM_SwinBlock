import torch
import torch.nn as nn

class ECBAM(nn.Module):
    """Efficient Channel and Spatial Attention Module with truly dynamic channels"""
    def __init__(self, c1, r=16):
        super().__init__()
        # Store the expected channel count at init time
        self.expected_channels = c1
        self.r = r
        
        # Spatial Attention (fixed components)
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
        self.sigmoid_channel = nn.Sigmoid()
        
        # Pooling operations (fixed components)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Initialize the channel attention layers with expected channels
        self._create_channel_layers(c1)

    def _create_channel_layers(self, channels):
        """Dynamically create channel attention layers for the given channel count"""
        # Ensure reduction doesn't go below 1
        reduction_channels = max(channels // self.r, 1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # Get actual input channels
        c_in = x.shape[1]
        
        # If channel count doesn't match the expected, recreate the layers
        if c_in != self.expected_channels:
            print(f"Adjusting ECBAM for {c_in} channels (was expecting {self.expected_channels})")
            self._create_channel_layers(c_in)
            # Update expected channels for future forwards
            self.expected_channels = c_in
            
            # Move to the same device as input
            self.fc = self.fc.to(x.device)

        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = self.sigmoid_channel(avg_out + max_out)
        x = x * ca  # Apply channel attention

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.sigmoid_spatial(self.conv(sa))

        return x * sa  # Apply spatial attention to the input