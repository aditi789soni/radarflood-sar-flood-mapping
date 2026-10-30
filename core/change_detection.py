"""
Bitemporal SAR Change Detection for separating Permanent Water from Newly Inundated Floodwater.

Computes:
    Delta_sigma0 (dB) = Post_flood_sigma0 (dB) - Pre_flood_sigma0 (dB)

Class definitions:
    0 : Non-water land (valid pixel)
    1 : Permanent water body (water in Pre-flood and Post-flood)
    2 : Newly inundated floodwater (land in Pre-flood, water in Post-flood with significant backscatter drop)
   -1 : Invalid / Nodata pixel
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

from core.flood_detection import (
    apply_morphological_postprocessing,
    classify_water_mask,
)
from utils.config import CONFIG
from utils.validation import RasterValidationError


@dataclass
class ChangeDetectionResult:
    """Outputs of bitemporal SAR flood change detection."""

    delta_vv_db: np.ndarray
    delta_vh_db: Optional[np.ndarray]
    new_flood_mask: np.ndarray
    permanent_water_mask: np.ndarray
    total_post_water_mask: np.ndarray
    classification_map: np.ndarray  # int8: -1=nodata, 0=land, 1=permanent water, 2=new flood
    valid_mask: np.ndarray
    effective_post_vv_thresh_db: float
    effective_pre_water_thresh_db: float
    delta_drop_thresh_db: float
    notes: list[str] = field(default_factory=list)


def run_bitemporal_change_detection(
    pre_vv_db: np.ndarray,
    post_vv_db: np.ndarray,
    pre_vh_db: Optional[np.ndarray] = None,
    post_vh_db: Optional[np.ndarray] = None,
    post_vv_threshold_db: float = CONFIG.DEFAULT_VV_THRESHOLD_DB,
    pre_perm_water_threshold_db: float = CONFIG.DEFAULT_PERMANENT_WATER_DB,
    delta_db_threshold: float = CONFIG.DEFAULT_CHANGE_DROP_DB,
    use_otsu_post: bool = False,
    include_vh: bool = False,
    post_vh_threshold_db: float = CONFIG.DEFAULT_VH_THRESHOLD_DB,
    vh_combine_mode: Literal["AND", "OR"] = "AND",
    apply_opening: bool = True,
    apply_closing: bool = True,
    morph_kernel_size: int = CONFIG.DEFAULT_MORPH_KERNEL_SIZE,
    min_region_pixels: int = CONFIG.DEFAULT_MIN_FLOOD_PIXELS,
) -> ChangeDetectionResult:
    """
    Compare pre-flood and post-flood calibrated dB SAR arrays on a co-registered grid.
    """
    if pre_vv_db.shape != post_vv_db.shape:
        raise RasterValidationError(
            f"Pre-flood shape {pre_vv_db.shape} and Post-flood shape {post_vv_db.shape} must be aligned before change detection."
        )

    valid_mask = np.isfinite(pre_vv_db) & np.isfinite(post_vv_db)
    if include_vh and pre_vh_db is not None and post_vh_db is not None:
        if pre_vh_db.shape != pre_vv_db.shape or post_vh_db.shape != post_vv_db.shape:
            raise RasterValidationError("VH band dimensions do not match VV band dimensions.")
        valid_mask &= np.isfinite(pre_vh_db) & np.isfinite(post_vh_db)

    if np.count_nonzero(valid_mask) == 0:
        raise RasterValidationError("Bitemporal change detection failed: 0 mutually valid pixels between Pre and Post rasters.")

    # Compute Delta sigma0 = Post - Pre (dB)
    delta_vv_db = np.where(valid_mask, post_vv_db - pre_vv_db, np.nan)
    delta_vh_db: Optional[np.ndarray] = None
    if pre_vh_db is not None and post_vh_db is not None:
        vh_valid = np.isfinite(pre_vh_db) & np.isfinite(post_vh_db)
        delta_vh_db = np.where(vh_valid, post_vh_db - pre_vh_db, np.nan)

    # 1. Classify Post-flood water surface
    post_cls = classify_water_mask(
        vv_db=np.where(valid_mask, post_vv_db, np.nan),
        vh_db=np.where(valid_mask, post_vh_db, np.nan) if (include_vh and post_vh_db is not None) else None,
        vv_threshold_db=post_vv_threshold_db,
        use_otsu=use_otsu_post,
        include_vh=include_vh,
        vh_threshold_db=post_vh_threshold_db,
        vh_combine_mode=vh_combine_mode,
        apply_opening=apply_opening,
        apply_closing=apply_closing,
        morph_kernel_size=morph_kernel_size,
        min_region_pixels=min_region_pixels,
    )
    eff_post_thresh = post_cls.effective_vv_threshold_db

    # 2. Classify Pre-flood baseline permanent water
    raw_pre_water = valid_mask & (pre_vv_db <= pre_perm_water_threshold_db)
    pre_water_mask = apply_morphological_postprocessing(
        binary_mask=raw_pre_water,
        valid_mask=valid_mask,
        apply_opening=apply_opening,
        apply_closing=apply_closing,
        kernel_size=morph_kernel_size,
        min_region_pixels=min_region_pixels,
    )

    # Permanent water: water in pre-flood AND still water (or low backscatter) in post-flood
    permanent_water_mask = pre_water_mask & (
        post_cls.water_mask | (valid_mask & (post_vv_db <= eff_post_thresh + 1.5))
    )

    # 3. Newly inundated floodwater:
    #    - Not permanent water in Pre-flood
    #    - Classified as water in Post-flood
    #    - Backscatter decreased by at least delta_db_threshold (e.g., Delta sigma0 <= -3.0 dB)
    raw_new_flood = (
        valid_mask
        & (~permanent_water_mask)
        & post_cls.raw_mask
        & (delta_vv_db <= float(delta_db_threshold))
    )

    new_flood_mask = apply_morphological_postprocessing(
        binary_mask=raw_new_flood,
        valid_mask=valid_mask,
        apply_opening=apply_opening,
        apply_closing=apply_closing,
        kernel_size=morph_kernel_size,
        min_region_pixels=min_region_pixels,
    )
    # Ensure strict mutual exclusivity with permanent water
    new_flood_mask = new_flood_mask & (~permanent_water_mask) & valid_mask
    total_post_water_mask = (permanent_water_mask | new_flood_mask) & valid_mask

    # Multi-class map: -1 = nodata, 0 = non-water land, 1 = permanent water, 2 = new floodwater
    classification_map = np.full(pre_vv_db.shape, -1, dtype=np.int8)
    classification_map[valid_mask] = 0
    classification_map[permanent_water_mask] = 1
    classification_map[new_flood_mask] = 2

    notes = list(post_cls.notes)
    notes.append(
        f"Bitemporal criteria: Post VV <= {eff_post_thresh:.2f} dB, "
        f"Delta sigma0 <= {delta_db_threshold:.2f} dB, "
        f"Pre permanent water <= {pre_perm_water_threshold_db:.2f} dB."
    )

    return ChangeDetectionResult(
        delta_vv_db=delta_vv_db,
        delta_vh_db=delta_vh_db,
        new_flood_mask=new_flood_mask,
        permanent_water_mask=permanent_water_mask,
        total_post_water_mask=total_post_water_mask,
        classification_map=classification_map,
        valid_mask=valid_mask,
        effective_post_vv_thresh_db=eff_post_thresh,
        effective_pre_water_thresh_db=float(pre_perm_water_threshold_db),
        delta_drop_thresh_db=float(delta_db_threshold),
        notes=notes,
    )

