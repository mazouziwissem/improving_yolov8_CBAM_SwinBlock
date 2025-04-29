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



import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

