import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class MCFD(nn.Module):
    """Multi-Context Feature Distillation module for tumor detection"""
    def __init__(self, c1, c2):
        super().__init__()
        self.mid_c = c1 // 2
        
        # Context pathway with varying dilation rates
        self.context_path = nn.ModuleList([
            nn.Sequential(
                Conv(c1, self.mid_c // 4, k=3, d=1, p=1),  # Standard context
                Conv(self.mid_c // 4, self.mid_c // 4, k=3, d=1, p=1)
            ),
            nn.Sequential(
                Conv(c1, self.mid_c // 4, k=3, d=2, p=2),  # Medium context
                Conv(self.mid_c // 4, self.mid_c // 4, k=3, d=2, p=2)
            ),
            nn.Sequential(
                Conv(c1, self.mid_c // 4, k=3, d=4, p=4),  # Large context
                Conv(self.mid_c // 4, self.mid_c // 4, k=3, d=4, p=4)
            ),
            nn.Sequential(
                Conv(c1, self.mid_c // 4, k=1),  # Point-wise context
                Conv(self.mid_c // 4, self.mid_c // 4, k=1)
            )
        ])
        
        # Dynamic input channel handling for density gate
        self.density_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c1, max(c1 // 16, 8), k=1),  # Ensure minimum channel count
            nn.SiLU(),
            Conv(max(c1 // 16, 8), 4, k=1),
            nn.Sigmoid()
        )
        
        # Feature distillation pathway
        self.distill_path = nn.Sequential(
            Conv(c1, self.mid_c, k=1),
            nn.Dropout(0.1),  # Reduced dropout for stability
            Conv(self.mid_c, self.mid_c, k=3, p=1, g=self.mid_c),  # Depthwise
            Conv(self.mid_c, c2, k=1)  # Pointwise projection
        )
        
        # Refined channel attention - fixed input channels calculation
        self.channel_refine = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(self.mid_c, max(self.mid_c // 4, 8), k=1),
            nn.SiLU(),
            Conv(max(self.mid_c // 4, 8), c2, k=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Apply each context pathway
        density_weights = self.density_gate(x)
        context_outputs = []
        
        for i, path in enumerate(self.context_path):
            # Apply context extraction with density-aware weighting
            context_feat = path(x) * density_weights[:, i:i+1, :, :]
            context_outputs.append(context_feat)
            
        # Concatenate all context features
        context_combined = torch.cat(context_outputs, dim=1)
        
        # Distill features through specialized pathway
        distilled = self.distill_path(x)
        
        # Apply refined channel attention (fixed to prevent channel mismatch)
        channel_weights = self.channel_refine(context_combined)
        
        # Apply weighted feature fusion
        output = distilled * channel_weights
        
        return output