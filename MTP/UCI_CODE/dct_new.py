import torch
import numpy as np

def precompute_dct_basis(H, W, max_u, max_v, device=None):
    h_indices = (2 * torch.arange(H, device=device) + 1).view(-1, 1)  # Shape (H, 1)
    w_indices = (2 * torch.arange(W, device=device) + 1).view(-1, 1)  # Shape (W, 1)
    
    cos_u = torch.cos(torch.pi * torch.arange(max_u + 1, device=device).view(1, -1) * h_indices / (2 * H))
    cos_v = torch.cos(torch.pi * torch.arange(max_v + 1, device=device).view(1, -1) * w_indices / (2 * W))
    return cos_u, cos_v


def dct_2d_torch(X, cos_u, cos_v):
    # Ensure dimensions are correct
    assert X.shape[1] == cos_u.shape[1], "Mismatch: X height must match cos_u height."
    assert X.shape[2] == cos_v.shape[1], "Mismatch: X width must match cos_v width."
    
    # Perform the DCT using batch operations
    dct_matrix = cos_u @ X @ cos_v.T  # Correct batch matrix multiplication
    Part = dct_matrix[:, :cos_u.shape[0], :cos_v.shape[0]].mean(dim=(1, 2))
    return Part


def multi_frequency_compression(X, k, max_u, max_v):
    C, H, W = X.shape
    assert C % k == 0, "The number of channels C should be divisible by k."
    C_k = C // k  # Channels per part

    # Precompute cosine matrices with the actual dimensions of X
    # Recompute cos_u and cos_v for full dimensions
    cos_u, cos_v = precompute_dct_basis(129, 8, max_u=129 - 1, max_v=8 - 1, device=X.device)

    # Split channels into k parts and compute DCT in batch
    parts = X.view(k, C_k, H, W)
    dct_parts = torch.stack([dct_2d_torch(part, cos_u, cos_v) for part in parts], dim=0)

    # Flatten and normalize
    Freq = dct_parts.flatten()
    Freq = Freq / torch.norm(Freq) if torch.norm(Freq) != 0 else Freq
    return Freq


def channel_attention(Freq):
    """
    Apply a fully connected layer followed by a sigmoid function for channel attention.
    
    Args:
        Freq: numpy array of shape (C,), the multi-frequency vector.
        
    Returns:
        Freq_att: numpy array of shape (C,), the channel attention.
    """
    # Fully connected layer can be represented as a dot product with a weight vector
    # For simplicity, we assume the fully connected layer has weights of ones and no bias
    Freq_att = 1 / (1 + np.exp(-Freq))  # Sigmoid activation
    return Freq_att