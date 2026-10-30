"""
Adaptive Lee Speckle Filter for SAR imagery using NumPy and SciPy.

Implements the Minimum Mean Square Error (MMSE) Lee filter (Lee, 1980):
    I_hat = I_bar + W * (I - I_bar)
where:
    I_bar     = local window mean
    Var_local = local window variance
    Var_noise = (I_bar^2) / ENL
    W         = clamp((Var_local - Var_noise) / (Var_local + eps), 0, 1)

Supports configurable 3x3, 5x5, and 7x7 windows, preserves high-contrast edges
(where Var_local >> Var_noise -> W -> 1), and safely handles NaN / nodata pixels
via normalized convolution.
"""

from typing import Optional
import numpy as np
from scipy.ndimage import uniform_filter

from core.preprocessing import sanitize_sar_array
from utils.config import CONFIG


def _nan_uniform_mean_and_var(data: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local window mean and variance ignoring NaNs using normalized uniform_filter.
    Returns (local_mean, local_var, valid_mask).
    """
    valid_mask = np.isfinite(data)
    weights = valid_mask.astype(np.float64)
    zeroed = np.where(valid_mask, data, 0.0)

    # Normalized local mean
    weight_sum = uniform_filter(weights, size=window_size, mode="nearest")
    data_sum = uniform_filter(zeroed, size=window_size, mode="nearest")
    data_sq_sum = uniform_filter(zeroed * zeroed, size=window_size, mode="nearest")

    has_support = weight_sum > 1e-8
    safe_weight = np.where(has_support, weight_sum, 1.0)

    local_mean = np.where(has_support, data_sum / safe_weight, np.nan)
    local_mean_sq = np.where(has_support, data_sq_sum / safe_weight, np.nan)
    local_var = np.maximum(local_mean_sq - local_mean * local_mean, 0.0)

    return local_mean, local_var, valid_mask


def lee_speckle_filter(
    image: np.ndarray,
    window_size: int = 5,
    enl: float = 4.4,
    is_db: bool = True,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """
    Apply the adaptive Lee speckle filter to a 2D SAR image.

    Parameters
    ----------
    image : np.ndarray
        2D SAR backscatter array (in dB if `is_db=True`, or linear intensity if `is_db=False`).
    window_size : int
        Filter kernel size; must be one of (3, 5, 7) corresponding to 3x3, 5x5, or 7x7.
    enl : float
        Equivalent Number of Looks (ENL) characterizing multiplicative speckle noise variance
        (default 4.4 for Sentinel-1 IW GRD multi-look imagery).
    is_db : bool
        If True, converts dB to linear intensity before applying the multiplicative Lee MMSE
        estimator and converts back to dB afterward.
    nodata : Optional[float]
        Optional nodata value to mask as NaN.

    Returns
    -------
    np.ndarray
        Filtered 2D SAR array in the same scale as the input, preserving NaNs at nodata locations.
    """
    if image is None or image.ndim != 2:
        raise ValueError("Lee speckle filter requires a 2D numpy array.")

    if window_size not in CONFIG.ALLOWED_LEE_WINDOWS:
        raise ValueError(
            f"Invalid window_size={window_size}. Must be one of {CONFIG.ALLOWED_LEE_WINDOWS} (3x3, 5x5, or 7x7)."
        )

    if enl <= 0:
        raise ValueError(f"Equivalent Number of Looks (enl) must be positive, got {enl}.")

    clean = sanitize_sar_array(image, nodata=nodata)
    valid_mask = np.isfinite(clean)
    if np.count_nonzero(valid_mask) == 0:
        return clean.copy()

    # Multiplicative speckle statistics are physically defined in linear intensity space
    if is_db:
        # Clamp extreme dB values to avoid overflow/underflow during 10^(dB/10)
        clamped_db = np.clip(clean, -80.0, 40.0)
        work_img = np.where(valid_mask, np.power(10.0, clamped_db / 10.0), np.nan)
    else:
        work_img = np.where(valid_mask & (clean >= 0.0), clean, np.nan)

    local_mean, local_var, work_valid = _nan_uniform_mean_and_var(work_img, window_size=window_size)

    # Multiplicative noise coefficient of variation squared: Cu^2 = 1 / ENL
    # Maximum heterogeneity threshold (Lopes et al., 1990): Cmax^2 = 2 / ENL + 1 / ENL^2
    cu_sq = 1.0 / float(enl)
    cmax_sq = (2.0 / float(enl)) + (cu_sq * cu_sq)

    mean_sq = np.maximum(local_mean * local_mean, 1e-18)
    ci_sq = local_var / mean_sq

    # Adaptive MMSE weight W in [0, 1]:
    # 1. Homogeneous region (ci_sq <= cu_sq): W = 0 -> maximum speckle smoothing (local_mean)
    # 2. Moderate heterogeneity (cu_sq < ci_sq < cmax_sq): MMSE transition weight
    # 3. Strong edge / point target (ci_sq >= cmax_sq): W = 1 -> exact edge preservation
    denom = np.maximum(cmax_sq - cu_sq, 1e-12)
    weight = np.clip((ci_sq - cu_sq) / denom, 0.0, 1.0)

    filtered_linear = local_mean + weight * (work_img - local_mean)
    filtered_linear = np.where(work_valid, filtered_linear, np.nan)

    if is_db:
        safe_linear = np.maximum(filtered_linear, CONFIG.EPSILON_LINEAR)
        filtered_out = np.where(work_valid, 10.0 * np.log10(safe_linear), np.nan)
    else:
        filtered_out = filtered_linear

    return filtered_out

