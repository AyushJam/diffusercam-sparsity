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


def estimate_sigma(detail_coeffs_level1):
    """
    Estimates the noise standard deviation (sigma_n) using the median absolute
    deviation (MAD) method on the finest-scale detail coefficients (Level 1).
    
    Args:
        detail_coeffs_level1 (tuple): Detail coefficient tuple (cH, cV, cD) 
                                      for the finest scale.
                              
    Returns:
        float: Estimated noise standard deviation.
    """
    # Concatenate all three subbands (Horizontal, Vertical, Diagonal) at Level 1
    finest_scale_coeffs = np.concatenate([c.flatten() for c in detail_coeffs_level1])
    
    # MAD (Median Absolute Deviation) is a robust estimator for Gaussian noise variance
    # The coefficients are zero-mean, so we use median absolute deviation
    mad = np.median(np.abs(finest_scale_coeffs))
    
    # Relationship between MAD and standard deviation (sigma) for Gaussian distribution
    # 0.6745 is the quantile corresponding to 0.5 in the standard normal distribution
    sigma_n = mad / 0.6745
    
    # Ensure sigma_n is not zero
    return max(sigma_n, 1e-6)


def wavelet_denoising_bayes(x: torch.Tensor, tau: float, level: int) -> torch.Tensor:
    """
    Performs multi-level 3D Haar Wavelet Denoising (Wavelet Shrinkage)
    on a (T, H, W, C) tensor, processing each channel separately.

    This version uses a level-dependent threshold (Universal Thresholding). 
    The input `tau` is now interpreted as a scaling factor for the estimated 
    Universal Threshold, providing fine control.

    Args:
        x (torch.Tensor): Input tensor of shape (T, H, W, C).
        tau (float): Scaling factor for the Universal Threshold (e.g., tau=1.0 for default).
        level (int): The number of decomposition levels (N).

    Returns:
        torch.Tensor: The denoised tensor of the same shape.
    """

    if x.ndim != 4 or x.size(-1) not in [1, 3]:
        raise ValueError("Expected input shape (T, H, W, C) for single-channel or RGB data.")

    out = torch.zeros_like(x)
    x_np = x[0].cpu().detach().numpy() # shape becomes H, W, C
    wavelet = 'haar'

    # 2. Iterate over each channel (C dimension)
    for c in range(x.size(-1)):
        x_c = x_np[..., c]  # shape: (T, H, W)
        
        # --- Decomposition ---
        # coeffs structure: [cA_n, {cD_n}, {cD_{n-1}}, ..., {cD_1}]
        coeffs = pywt.wavedec2(x_c, wavelet, level=level, mode='symmetric')
        approx_coeffs = coeffs[0]
        detail_levels = list(coeffs[1:])

        # --- Noise Estimation ---
        # Estimate noise from the highest level of detail (cD_1, which is detail_levels[-1])
        # Note: In pywt.wavedecn structure, coeffs[1] is the coarsest detail (level N), 
        # and coeffs[level] is the finest detail (level 1).
        
        # The finest detail coefficients are in detail_levels[-1]
        sigma_n = estimate_sigma(detail_levels[-1]) 
        sigma_n_sq = (sigma_n ** 2)
        
        # --- Thresholding ---
        new_detail_levels = []
        for j in range(len(detail_levels)):
            detail_tuple = detail_levels[j]
            new_detail_tuple = [] # actually a list

            # Apply soft thresholding to each of the three detail subbands (H, V, D)
            for detail_subband in detail_tuple:

                sigma_band_sq = np.sum(np.power(detail_subband,2)) / detail_subband.size
                sigma_coef_est = np.maximum(sigma_band_sq - sigma_n_sq , 0)
                if sigma_coef_est == 0:
                    sigma_coef_est = np.max(np.abs(detail_subband))
                else:
                    sigma_coef_est = pow(sigma_coef_est, 1/2)
                
                tau_est = sigma_n_sq/sigma_coef_est
                new_detail_tuple.append(soft_threshold(detail_subband, tau_est))
            
            new_detail_levels.append(tuple(new_detail_tuple))
        
        # --- Reconstruction ---
        denoised_coeffs = [approx_coeffs] + new_detail_levels
        y_c_np = pywt.waverec2(denoised_coeffs, wavelet, mode='symmetric')
        
        # --- Store Result ---
        out[0, :, :, c] = torch.from_numpy(y_c_np).to(x.device)
    
    return out


