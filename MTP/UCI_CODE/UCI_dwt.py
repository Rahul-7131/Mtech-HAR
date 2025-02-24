import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from UCI_model import *

# ------------------------- Added DWT Components -------------------------
class DualWaveletAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        # Channel attention components
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
        # Spatial attention components
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def dwt_decompose(self, x):
        """2D Haar Wavelet Decomposition (Level-1)"""
        B, C, H, W = x.shape
        x = x.view(B*C, 1, H, W)
        x = F.unfold(x, kernel_size=2, stride=2)
        x = x.view(-1, 2, 2)
        
        # Compute subbands
        LL = x[:, 0, 0]
        LH = x[:, 0, 1] - x[:, 0, 0]
        HL = x[:, 1, 0] - x[:, 0, 0]
        HH = x[:, 1, 1] - x[:, 0, 0]
        
        # Reshape back
        LL = LL.view(B, C, H//2, W//2)
        LH = LH.view(B, C, H//2, W//2)
        HL = HL.view(B, C, H//2, W//2)
        HH = HH.view(B, C, H//2, W//2)
        
        return LL, LH, HL, HH

    def forward(self, x):
        # Channel attention
        LL, LH, HL, HH = self.dwt_decompose(x)
        Wc = torch.mean(LL + LH + HL + HH, dim=(2,3))
        Mc = self.fc(Wc).unsqueeze(-1).unsqueeze(-1)
        x = x * Mc
        
        # Spatial attention
        Ws_low = LL
        Ws_high = torch.cat([LH, HL, HH], dim=1)
        avg_pool = torch.mean(Ws_high, dim=1, keepdim=True)
        max_pool = torch.max(Ws_high, dim=1, keepdim=True)[0]
        Ms = self.conv(torch.cat([avg_pool, max_pool], dim=1))
        Ms = F.interpolate(Ms, scale_factor=2, mode='bilinear')
        
        return x * Ms

# ------------------------- Modified PyramidAttentionModel -------------------------
class PyramidAttentionModel(nn.Module):
    def __init__(self, input_channels, n_classes, num_splits):
        super(PyramidAttentionModel, self).__init__()
        assert num_splits >= 2 and num_splits % 2 == 0, "num_splits must be even and at least 2"
        self.num_splits = num_splits
        self.dropout = nn.Dropout(p=0.3)
        self.n_classes = n_classes
        
        # Create PyramidMultiScaleCNN groups with DWA
        self.groups = nn.ModuleList([
            nn.Sequential(
                PyramidMultiScaleCNN(input_channels),
                DualWaveletAttention(1792)  # 1792 comes from original channel count
            ) for _ in range(num_splits)
        ])
        
        # GAP branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Wavelet attention branch
        feature_size = int((num_splits/2)*1792)
        self.classifier_gap = nn.Linear(feature_size, n_classes)
        self.classifier_wavelet = nn.Linear(feature_size, n_classes)
        self.batch_norm = nn.BatchNorm1d(feature_size)

    def forward(self, x):
        # Channel handling remains same
        batch_size, channels, height, width = x.size()
        if channels % self.num_splits != 0:
            padding = (0, 0, 0, 0, 0, self.num_splits - (channels % self.num_splits))
            x = F.pad(x, padding)
        
        split_channels = channels // self.num_splits
        split_inputs = torch.split(x, split_channels, dim=1)
        
        # Process through groups with DWA
        outputs = [group(split_input) for group, split_input in zip(self.groups, split_inputs)]
        concat_out = torch.cat(outputs, dim=1)
        
        # GAP Branch
        gap_out = self.gap(concat_out).view(batch_size, -1)
        gap_pred = self.classifier_gap(gap_out)
        
        # Wavelet Branch
        wavelet_out = concat_out.mean(dim=(2,3))  # Spatial pooling
        wavelet_out = self.batch_norm(wavelet_out)
        wavelet_out = self.dropout(wavelet_out)
        wavelet_pred = self.classifier_wavelet(wavelet_out)
        
        return gap_pred, wavelet_pred