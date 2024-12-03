import numpy as np
import torch 

def precompute_dct_basis(H, W, max_u, max_v, device=None):
    """
    Precompute the DCT basis matrices for given dimensions.
    """
    u_vals = torch.arange(max_u + 1, device=device)
    v_vals = torch.arange(max_v + 1, device=device)
    h_indices = 2 * torch.arange(H, device=device) + 1
    w_indices = 2 * torch.arange(W, device=device) + 1

    cos_u = torch.cos(torch.pi * u_vals[:, None] * h_indices / (2 * H))
    cos_v = torch.cos(torch.pi * v_vals[:, None] * w_indices / (2 * W))

    return cos_u, cos_v

def dct_2d_torch(X_i, cos_u, cos_v):
    """
    Perform DCT compression using precomputed cosine matrices.
    """
    C_k, H, W = X_i.shape
    max_u, max_v = cos_u.shape[0] - 1, cos_v.shape[0] - 1
    Part_i = torch.zeros(C_k, device=X_i.device)

    for c in range(C_k):
        # Compute the DCT for all frequencies using matrix multiplication
        dct_matrix = cos_u @ X_i[c, :, :] @ cos_v.T
        Part_i[c] = dct_matrix[:max_u + 1, :max_v + 1].mean()

    return Part_i
    
def multi_frequency_compression(X, k, max_u, max_v):
    """
    Apply multi-frequency compression using optimized DCT on the input tensor X.
    """
    C, H, W = X.shape
    assert C % k == 0, "The number of channels C should be divisible by k."
    C_k = C // k  # Channels per part

    # Precompute cosine matrices
    cos_u, cos_v = precompute_dct_basis(H, W, max_u, max_v, device=X.device)

    Freq_parts = []
    for i in range(k):
        X_i = X[i * C_k:(i + 1) * C_k, :, :]
        Part_i = dct_2d_torch(X_i, cos_u, cos_v)
        Freq_parts.append(Part_i)

    # Concatenate and normalize
    Freq = torch.cat(Freq_parts)
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
