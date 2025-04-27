import torch
import torch.nn as nn

class RepConv2x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RepConv2x2, self).__init__()
        self.conv_2x2 = nn.Conv2d(in_channels, out_channels, 2, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv_2x2(x)))

class RepC3K2(nn.Module):
    def __init__(self, in_channels, out_channels, n=3):
        super(RepC3K2, self).__init__()
        hidden_channels = int(out_channels * 0.5)

        self.cv1 = RepConv2x2(in_channels, hidden_channels)
        self.cv2 = RepConv2x2(in_channels, hidden_channels)
        self.m = nn.Sequential(*[RepConv2x2(hidden_channels, hidden_channels) for _ in range(n)])
        self.cv3 = RepConv2x2(2 * hidden_channels, out_channels)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y1 = self.m(y1)
        return self.cv3(torch.cat((y1, y2), dim=1))
