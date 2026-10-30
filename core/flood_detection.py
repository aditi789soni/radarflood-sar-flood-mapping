"""
SAR flood/water classification using configurable dB thresholding, Otsu's method,
dual-polarization (VV + VH) criteria, and morphological post-processing.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import cv2
import numpy as np

from utils.config import CONFIG
from utils.validation import RasterValidationError


@dataclass
class FloodClassificationResult:
    """Result of single-date water/flood classification."""

    water_mask: np.ndarray
    raw_mask: np.ndarray
    valid_mask: np.ndarray
    effective_vv_threshold_db: float
    effective_vh_threshold_db: Optional[float]
    method_used: str
    notes: list[str] = field(default_factory=list)


def compute_otsu_threshold_db(
    sigma0_db: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    num_bins: int = 256,
) -> float:
    """
    Compute Otsu's optimal global threshold (in dB) separating bimodal water/land
    backscatter distributions over valid finite pixels.
    """
    if valid_mask is None:
        valid_mask = np.isfinite(sigma0_db)
    else:
        valid_mask = valid_mask & np.isfinite(sigma0_db)

    valid_vals = sigma0_db[valid_mask]
    if valid_vals.size < 10:
        return CONFIG.DEFAULT_VV_THRESHOLD_DB

    # Clip extreme outliers before computing histogram for Otsu stability
    p01, p99 = np.percentile(valid_vals, [0.5, 99.5])
    if np.isclose(p01, p99):
        return float(p01)

    clipped = np.clip(valid_vals, p01, p99)
    hist, bin_edges = np.histogram(clipped, bins=num_bins, range=(float(p01), float(p99)))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    total = float(hist.sum())
    if total <= 0:
        return CONFIG.DEFAULT_VV_THRESHOLD_DB

    prob = hist.astype(np.float64) / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]

    # Between-class variance: sigma_b^2 = (mu_t * omega - mu)^2 / (omega * (1 - omega))
    denom = omega * (1.0 - omega)
    valid_bins = denom > 1e-10
    if not np.any(valid_bins):
        return CONFIG.DEFAULT_VV_THRESHOLD_DB

    sigma_b_sq = np.zeros_like(omega)
    sigma_b_sq[valid_bins] = ((mu_t * omega[valid_bins] - mu[valid_bins]) ** 2) / denom[valid_bins]

    best_idx = int(np.argmax(sigma_b_sq))
    otsu_val = float(bin_centers[best_idx])

    # Clamp to physically sensible water/land separation bounds
    return float(np.clip(otsu_val, CONFIG.MIN_VV_THRESHOLD_DB, CONFIG.MAX_VV_THRESHOLD_DB))


def apply_morphological_postprocessing(
    binary_mask: np.ndarray,
    valid_mask: np.ndarray,
    apply_opening: bool = True,
    apply_closing: bool = True,
    kernel_size: int = 3,
    min_region_pixels: int = 16,
) -> np.ndarray:
    """
    Refine a binary water/flood mask using morphological opening, morphological closing,
    connected-component small region removal, and strict invalid-pixel masking.
    """
    refined = (binary_mask.astype(bool) & valid_mask.astype(bool)).astype(np.uint8)
    if np.count_nonzero(refined) == 0:
        return refined.astype(bool)

    k_size = max(1, int(kernel_size))
    if k_size % 2 == 0:
        k_size += 1

    if k_size > 1 and (apply_opening or apply_closing):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        if apply_opening:
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        if apply_closing:
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)

    # Ensure morphological dilation/closing never bleeds into invalid/nodata regions
    refined = refined & valid_mask.astype(np.uint8)

    if min_region_pixels > 1 and np.count_nonzero(refined) > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
        if num_labels > 1:
            sizes = stats[:, cv2.CC_STAT_AREA]
            keep_labels = sizes >= int(min_region_pixels)
            keep_labels[0] = False  # background label 0
            refined = keep_labels[labels].astype(np.uint8)

    return (refined.astype(bool)) & valid_mask


def classify_water_mask(
    vv_db: np.ndarray,
    vh_db: Optional[np.ndarray] = None,
    vv_threshold_db: float = CONFIG.DEFAULT_VV_THRESHOLD_DB,
    use_otsu: bool = False,
    include_vh: bool = False,
    vh_threshold_db: float = CONFIG.DEFAULT_VH_THRESHOLD_DB,
    vh_combine_mode: Literal["AND", "OR"] = "AND",
    apply_opening: bool = True,
    apply_closing: bool = True,
    morph_kernel_size: int = CONFIG.DEFAULT_MORPH_KERNEL_SIZE,
    min_region_pixels: int = CONFIG.DEFAULT_MIN_FLOOD_PIXELS,
) -> FloodClassificationResult:
    """
    Classify smooth open water / flood surfaces from calibrated SAR backscatter in dB.
    """
    if vv_db is None or vv_db.ndim != 2:
        raise RasterValidationError("classify_water_mask requires a 2D VV backscatter array in dB.")

    valid_mask = np.isfinite(vv_db)
    if include_vh and vh_db is not None:
        if vh_db.shape != vv_db.shape:
            raise RasterValidationError(
                f"VH band shape {vh_db.shape} does not match VV band shape {vv_db.shape}."
            )
        valid_mask &= np.isfinite(vh_db)

    if np.count_nonzero(valid_mask) == 0:
        raise RasterValidationError("Cannot classify water: 0 valid finite pixels in input backscatter.")

    notes: list[str] = []
    if use_otsu:
        eff_vv_thresh = compute_otsu_threshold_db(vv_db, valid_mask=valid_mask)
        method_used = f"Otsu Automatic Threshold ({eff_vv_thresh:.2f} dB)"
        notes.append(f"Computed Otsu VV threshold: {eff_vv_thresh:.2f} dB.")
    else:
        eff_vv_thresh = float(vv_threshold_db)
        method_used = f"Manual VV Threshold ({eff_vv_thresh:.2f} dB)"

    raw_vv_water = valid_mask & (vv_db <= eff_vv_thresh)
    eff_vh_thresh: Optional[float] = None

    if include_vh and vh_db is not None:
        eff_vh_thresh = float(vh_threshold_db)
        raw_vh_water = valid_mask & (vh_db <= eff_vh_thresh)
        if vh_combine_mode == "OR":
            raw_mask = raw_vv_water | raw_vh_water
            method_used += f" OR VH <= {eff_vh_thresh:.2f} dB"
        else:
            raw_mask = raw_vv_water & raw_vh_water
            method_used += f" AND VH <= {eff_vh_thresh:.2f} dB"
        notes.append(
            f"Combined VV (<= {eff_vv_thresh:.2f} dB) {vh_combine_mode} VH (<= {eff_vh_thresh:.2f} dB)."
        )
    else:
        raw_mask = raw_vv_water
        if include_vh and vh_db is None:
            notes.append("VH classification requested, but no VH band is present; using VV band only.")

    water_mask = apply_morphological_postprocessing(
        binary_mask=raw_mask,
        valid_mask=valid_mask,
        apply_opening=apply_opening,
        apply_closing=apply_closing,
        kernel_size=morph_kernel_size,
        min_region_pixels=min_region_pixels,
    )

    return FloodClassificationResult(
        water_mask=water_mask,
        raw_mask=raw_mask,
        valid_mask=valid_mask,
        effective_vv_threshold_db=eff_vv_thresh,
        effective_vh_threshold_db=eff_vh_thresh,
        method_used=method_used,
        notes=notes,
    )

