import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ESEAttn(nn.Module):
    """Efficient Squeeze-and-Excitation Attention Module"""
    def __init__(self, channels, reduction=4):
        super(ESEAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class GlobalContext(nn.Module):
    """Global Context Block"""
    def __init__(self, channels, reduction=16):
        super(GlobalContext, self).__init__()
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.LayerNorm([channels // reduction, 1, 1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1)
        )
        
    def forward(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height*width)
        context_mask = self.softmax(context_mask)
        # [N, 1, H, W]
        context_mask = context_mask.view(batch, 1, height, width)
        
        # [N, C, 1, 1]
        context = torch.sum(x * context_mask, dim=(2, 3), keepdim=True)
        
        # Transform the context
        channel_add_term = self.channel_add_conv(context)
        
        return input_x + channel_add_term

class C2GC(nn.Module):
    """C2f block with Global Context"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C2GC, self).__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(nn.Sequential(
            nn.Conv2d(self.c, self.c, 3, 1, 1, groups=g),
            GlobalContext(self.c),
            nn.SiLU()) for _ in range(n))
        self.shortcut = shortcut
        
    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures"""
    def __init__(self, c, kernel_size=3, up_factor=1):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.channels = c
        
        self.comp = nn.Conv2d(c, c // 4, 1)
        self.enc = nn.Conv2d(c // 4, 
                           self.kernel_size * self.kernel_size * self.up_factor * self.up_factor, 
                           kernel_size=3, 
                           padding=1)
        self.pix_shuffle = nn.PixelShuffle(self.up_factor)
        
        self.relu = nn.ReLU(inplace=True)
        self.mask_softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Feature compression
        compressed = self.comp(x)
        compressed = self.relu(compressed)
        
        # Generate kernel weights
        kernel_weights = self.enc(compressed)
        kernel_weights = self.relu(kernel_weights)
        
        # Reshape for efficient content-aware reassembly
        kernel_weights = kernel_weights.view(b, self.kernel_size * self.kernel_size, 
                                           self.up_factor * h, self.up_factor * w)
        kernel_weights = self.mask_softmax(kernel_weights)
        
        # Apply reassembly
        if self.up_factor > 1:
            x = self.pix_shuffle(x)
        
        # Output feature map
        return kernel_weights * x

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, c, dilations=(1, 6, 12, 18)):
        super(ASPP, self).__init__()
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            self.aspp.append(nn.Sequential(
                nn.Conv2d(c, c//len(dilations), 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(c//len(dilations)),
                nn.SiLU(inplace=True)
            ))
        self.conv = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        res = []
        for module in self.aspp:
            res.append(module(x))
        res = torch.cat(res, dim=1)
        res = self.conv(res)
        res = self.bn(res)
        return self.act(res)

class BiCAU(nn.Module):
    """Bidirectional Content-Aware Upsampling"""
    def __init__(self, c, scale_factor):
        super(BiCAU, self).__init__()
        self.scale_factor = scale_factor
        
        self.conv_mask = nn.Conv2d(c, scale_factor**2 * 9, kernel_size=3, padding=1)
        self.up_scale = nn.PixelShuffle(scale_factor)
        
        self.mask_act = nn.Sigmoid()
        
    def forward(self, x):
        # Generate upsampling mask
        mask = self.conv_mask(x)
        mask = self.mask_act(mask)
        
        # Simple bilinear upsampling
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        # Apply content-aware refinement
        n, c, h, w = x_up.shape
        mask = mask.view(n, 9, h, w)
        
        # Apply mask to feature (simplified implementation)
        refined = x_up * mask.mean(dim=1, keepdim=True)
        
        return refined

class DyReLU(nn.Module):
    """Dynamic ReLU"""
    def __init__(self, c, reduction=4):
        super(DyReLU, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c // reduction),
            nn.SiLU(inplace=True),
            nn.Linear(c // reduction, 2 * c)
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 2, c, 1, 1)
        
        a, b = y[:, 0], y[:, 1]
        
        # Dynamic ReLU: max(a*x + b, 0)
        out = torch.max(x * a + b, torch.zeros_like(x))
        return out

class GSConv(nn.Module):
    """Ghost-Shuffle Convolution"""
    def __init__(self, c1, c2, k=1, s=1, g=1):
        super(GSConv, self).__init__()
        c_ = c2 // 2  # Half the number of filters for Ghost Conv
        self.cv1 = nn.Conv2d(c1, c_, k, s, k//2, groups=g, bias=False)
        self.cv2 = nn.Conv2d(c_, c_, 5, 1, 2, groups=c_, bias=False)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(y1)
        
        # Channel shuffle
        bs, c, h, w = y1.shape
        y = torch.cat([y1, y2], dim=1)
        y = y.view(bs, 2, c, h, w).permute(0, 2, 1, 3, 4).contiguous().view(bs, 2*c, h, w)
        
        return self.act(self.bn(y))