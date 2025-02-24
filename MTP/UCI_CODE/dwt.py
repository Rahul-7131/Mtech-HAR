import torch
import torch.nn as nn
import torch.nn.functional as F

def dwt_haar_2d(x):
    """
    Perform 2D Haar Wavelet Transform on input tensor.
    Args:
        x (Tensor): Input tensor of shape (C, H, W).
    Returns:
        LL, LH, HL, HH (Tensor): Subbands each of shape (C, H//2, W//2).
    """
    C, H, W = x.shape
    x = x.reshape(C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    
    # Split into subbands (approximation and details)
    LL = x[..., 0, 0]
    LH = x[..., 0, 1] - x[..., 0, 0]
    HL = x[..., 1, 0] - x[..., 0, 0]
    HH = x[..., 1, 1] - x[..., 0, 0]
    
    return LL, LH, HL, HH

def wavelet_channel_attention(F):
    """
    Wavelet Channel Attention Module.
    Args:
        F (Tensor): Input features of shape (C, H, W).
    Returns:
        Tensor: Channel attention weights of shape (C,).
    """
    LL, LH, HL, HH = dwt_haar_2d(F)
    # Sum coefficients across spatial dimensions
    Wc = LL.sum(dim=(1,2)) + LH.sum(dim=(1,2)) + HL.sum(dim=(1,2)) + HH.sum(dim=(1,2))
    # FC layers with ReLU and Sigmoid
    fc = nn.Sequential(
        nn.Linear(Wc.size(0), Wc.size(0) // 16),
        nn.ReLU(),
        nn.Linear(Wc.size(0) // 16, Wc.size(0)),
        nn.Sigmoid()
    )
    return fc(Wc)

def wavelet_spatial_attention(F_prime):
    """
    Wavelet Spatial Attention Module.
    Args:
        F_prime (Tensor): Input features of shape (C, H, W).
    Returns:
        Tensor: Spatial attention map of shape (C, H, W).
    """
    C, H, W = F_prime.shape
    LL, LH, HL, HH = dwt_haar_2d(F_prime)
    # Concatenate LL and (LL + LH + HH) along channel dim
    LL_plus = LL + LH + HH
    Ws = torch.cat([LL.unsqueeze(1), LL_plus.unsqueeze(1)], dim=1)  # (C, 2, H//2, W//2)
    # Process with Conv layers and upsample
    conv = nn.Sequential(
        nn.Conv2d(2, 1, kernel_size=1),
        nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)
    )
    Ms = torch.sigmoid(conv(Ws)).squeeze(1)  # (C, H, W)
    return Ms

def dual_wavelet_attention(F):
    """
    Dual Wavelet Attention Block: Combines channel and spatial attention.
    Args:
        F (Tensor): Input features of shape (C, H, W).
    Returns:
        Tensor: Refined features of shape (C, H, W).
    """
    # Channel attention
    Mc = wavelet_channel_attention(F)
    F_prime = F * Mc.unsqueeze(-1).unsqueeze(-1)
    # Spatial attention
    Ms = wavelet_spatial_attention(F_prime)
    F_double_prime = F_prime * Ms
    return F_double_prime