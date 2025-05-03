import torch

import torch.nn as nn

class SPD(nn.Module):
  """
  SPD (Spatial Pyramid Downsampling) module for feature extraction.
  This module is commonly used in object detection architectures to extract
  multi-scale features by applying pooling operations at different scales.
  """
  def __init__(self, in_channels, out_channels, pool_sizes=(5, 9, 13)):
    super(SPD, self).__init__()
    self.pool_layers = nn.ModuleList([
      nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2)
      for size in pool_sizes
    ])
    self.conv = nn.Conv2d(in_channels * (len(pool_sizes) + 1), out_channels, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    pooled_features = [x] + [pool(x) for pool in self.pool_layers]
    concatenated = torch.cat(pooled_features, dim=1)
    output = self.conv(concatenated)
    return output