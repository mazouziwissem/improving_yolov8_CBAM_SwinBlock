import torch
import torch.nn as nn

from ultralytics.nn.modules import GhostConv  # attention à bien importer GhostConv
from ultralytics.nn.modules import Concat

class GhostC2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        c1: nombre de channels en entrée
        c2: nombre de channels en sortie
        n: nombre de GhostConv à répéter
        shortcut: utiliser ou non un residual
        g: nombre de groupes pour Grouped Convolution
        e: expansion ratio
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = GhostConv(c1, c_, k=1, g=g)
        self.cv2 = GhostConv(c1, c_, k=1, g=g)
        self.cv3 = GhostConv(2 * c_, c2, k=1)
        self.m = nn.Sequential(*[GhostConv(c_, c_, k=3, s=1, g=g) for _ in range(n)])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y = torch.cat((self.m(y1), y2), dim=1)
        return self.cv3(y)
