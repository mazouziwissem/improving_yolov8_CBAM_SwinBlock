# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes=None, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.in_planes = in_planes
#         self.ratio = ratio
        
#         # Si in_planes est spécifié, on crée le MLP immédiatement
#         if in_planes is not None:
#             self.create_mlp(in_planes)
#         else:
#             # Sinon, le MLP sera créé au premier forward pass
#             self.shared_MLP = None
            
#     def create_mlp(self, in_planes):
#         # Assurez-vous que le ratio ne réduit pas les canaux à zéro
#         reduced_planes = max(1, in_planes // self.ratio)
#         self.shared_MLP = nn.Sequential(
#             nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
#         )
            
#     def forward(self, x):
#         # Si le MLP n'a pas été créé, on le crée maintenant avec les bonnes dimensions
#         if self.shared_MLP is None:
#             _, c, _, _ = x.shape
#             self.create_mlp(c)
            
#         avg_out = self.shared_MLP(self.avg_pool(x))
#         max_out = self.shared_MLP(self.max_pool(x))
#         out = avg_out + max_out
#         return torch.sigmoid(out)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
        
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv(x)
#         return torch.sigmoid(x)

# class CBAM(nn.Module):
#     def __init__(self, channels=None):
#         super(CBAM, self).__init__()
#         # On laisse le module déterminer automatiquement le nombre de canaux
#         self.ca = ChannelAttention(channels, ratio=8 if channels and channels < 128 else 16)
#         self.sa = SpatialAttention(kernel_size=7)
        
#     def forward(self, x):
#         # Appliquez l'attention des canaux
#         ca_output = self.ca(x)
#         x = x * ca_output
        
#         # Appliquez l'attention spatiale
#         sa_output = self.sa(x)
#         x = x * sa_output
        
#         return x



import torch.nn as nn
import torch
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, in_channels=None, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.initialized = False

    def _init(self, channels):
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // self.reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // self.reduction_ratio, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        self.spatial = nn.Conv2d(2, 1, kernel_size=self.kernel_size, padding=self.kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._init(x.size(1))  # Dynamically initialize

        # Channel attention
        max_out = self.shared_mlp(F.adaptive_max_pool2d(x, (1, 1)))
        avg_out = self.shared_mlp(F.adaptive_avg_pool2d(x, (1, 1)))
        channel_att = self.sigmoid_channel(max_out + avg_out)
        x = x * channel_att

        # Spatial attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.spatial(torch.cat([max_out, avg_out], dim=1)))
        x = x * spatial_att

        return x
