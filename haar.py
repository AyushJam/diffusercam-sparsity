import torch
import pywt
import numpy as np

def soft_threshold(coeffs, tau):
    """
    Applies soft thresholding: sgn(x) * max(0, |x| - tau).
    """
    if tau == 0:
        return coeffs
    # Check if coeffs is a numpy array or torch tensor and use appropriate operations
    if isinstance(coeffs, np.ndarray):
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - tau, 0)
    elif isinstance(coeffs, torch.Tensor):
        return torch.sign(coeffs) * torch.maximum(torch.zeros_like(coeffs), torch.abs(coeffs) - tau)
    else:
        raise TypeError("Input must be a NumPy array or PyTorch Tensor.")

