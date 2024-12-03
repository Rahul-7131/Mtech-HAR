import torch
import numpy as np

def precompute_dct_basis(H, W, max_u, max_v, device=None):
    """
    Precompute the DCT basis matrices for given dimensions using efficient batch operations.
    """
    h_indices = (2 * torch.arange(H, device=device) + 1).view(1, -1)  # Shape (1, H)
    w_indices = (2 * torch.arange(W, device=device) + 1).view(1, -1)  # Shape (1, W)

    u_vals = torch.arange(max_u + 1, device=device).view(-1, 1)  # Shape (max_u+1, 1)
    v_vals = torch.arange(max_v + 1, device=device).view(-1, 1)  # Shape (max_v+1, 1)

    cos_u = torch.cos(torch.pi * u_vals * h_indices / (2 * H))  # Shape (max_u+1, H)
    cos_v = torch.cos(torch.pi * v_vals * w_indices / (2 * W))  # Shape (max_v+1, W)

    return cos_u, cos_v

def dct_2d_torch(X, cos_u, cos_v):
    """
    Perform DCT compression using precomputed cosine matrices with batch processing.
    """
    C_k, H, W = X.shape
    assert cos_u.shape[1] == H, f"Mismatch in height: cos_u={cos_u.shape[1]}, X={H}"
    assert cos_v.shape[1] == W, f"Mismatch in width: cos_v={cos_v.shape[1]}, X={W}"

    # Adjust cosine matrices if dimensions mismatch
    cos_v = cos_v[:, :W]

    # Efficient matrix multiplication
    dct_matrix = torch.einsum('uh,chv,vw->cuv', cos_u, X, cos_v)

    #print(f"dct_matrix shape: {dct_matrix.shape}")
    return dct_matrix.mean(dim=(1, 2))


def multi_frequency_compression(X, k, max_u, max_v):
    """
    Apply multi-frequency compression using optimized DCT on the input tensor X.
    """
    C, H, W = X.shape
    assert C % k == 0, "The number of channels C should be divisible by k."
    C_k = C // k  # Channels per part

    # Precompute cosine matrices once
    cos_u, cos_v = precompute_dct_basis(129, 8, max_u=129 - 1, max_v=8 - 1, device=X.device)

    # Split the input tensor into k parts and process them in batch
    X_parts = X.view(k, C_k, H, W)  # Shape (k, C_k, H, W)
    Freq_parts = torch.stack([dct_2d_torch(X_part, cos_u, cos_v) for X_part in X_parts], dim=0)

    # Concatenate and normalize
    Freq = Freq_parts.flatten()  # Shape (C,)
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
