"""
SAR radiometric scale identification and conversion to Sigma Nought (sigma0) in decibels (dB).

Scientific notes:
- Sentinel-1 Level-1 GRD products contain uncalibrated digital numbers (DN) unless
  calibrated using product annotation look-up tables (LUTs) into sigma0, beta0, or gamma0.
- Never claim arbitrary uncalibrated DN values are calibrated sigma0.
- Positive linear sigma0 power values are converted to dB via:
      sigma0_dB = 10 * log10(sigma0_linear)
- Already-dB inputs must NOT undergo logarithmic conversion a second time.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

from core.preprocessing import sanitize_sar_array
from utils.config import CONFIG
from utils.validation import RasterValidationError

ScaleType = Literal["dB", "linear_sigma0", "uncalibrated_dn"]
InputScaleOption = Literal["auto", "dB", "linear_sigma0"]


@dataclass
class CalibrationResult:
    """Output of backscatter scale detection and dB conversion."""

    sigma0_db: np.ndarray
    detected_scale: ScaleType
    conversion_applied: bool
    valid_mask: np.ndarray
    invalid_nonpositive_count: int = 0
    notes: list[str] = field(default_factory=list)


def detect_backscatter_scale(
    arr: np.ndarray,
    nodata: Optional[float] = None,
    metadata_tags: Optional[dict[str, str]] = None,
) -> tuple[ScaleType, list[str]]:
    """
    Inspect pixel distribution and optional metadata tags to determine whether a SAR
    raster is already in decibels (dB), linear calibrated sigma0 intensity, or
    uncalibrated digital numbers (DN).
    """
    notes: list[str] = []
    clean = sanitize_sar_array(arr, nodata=nodata)
    valid = clean[np.isfinite(clean)]

    if valid.size == 0:
        raise RasterValidationError("Cannot determine backscatter scale: raster contains no valid finite pixels.")

    # Check metadata hints first if unambiguous
    if metadata_tags:
        cal_tag = str(metadata_tags.get("CALIBRATION", "") + " " + metadata_tags.get("UNITS", "")).lower()
        if "db" in cal_tag or "decibel" in cal_tag:
            notes.append("Metadata tag indicates backscatter is already in decibels (dB).")
            return "dB", notes
        if "linear" in cal_tag:
            notes.append("Metadata tag indicates calibrated linear sigma0 intensity.")
            return "linear_sigma0", notes

    min_val = float(np.min(valid))
    max_val = float(np.max(valid))
    p05 = float(np.percentile(valid, 5))
    p50 = float(np.percentile(valid, 50))
    p95 = float(np.percentile(valid, 95))
    neg_fraction = float(np.mean(valid < -1e-6))

    # Case 1: Contains substantial negative values within plausible SAR dB range [-60, +15] dB
    if neg_fraction > 0.15 and min_val >= CONFIG.DB_PLAUSIBLE_MIN and p95 <= CONFIG.DB_PLAUSIBLE_MAX:
        notes.append(
            f"Detected decibel (dB) scale: {neg_fraction * 100:.1f}% of pixels are negative "
            f"(median = {p50:.2f} dB, 5th-95th percentile = [{p05:.2f}, {p95:.2f}] dB). "
            "Logarithmic conversion skipped to prevent double conversion."
        )
        return "dB", notes

    # Case 2: Non-negative values with large magnitudes -> likely uncalibrated DN / amplitude
    if neg_fraction <= 0.01 and (p95 > 10.0 or max_val > 50.0):
        notes.append(
            f"WARNING: Pixel values appear to be uncalibrated Digital Numbers (DN) or amplitude "
            f"(median = {p50:.2f}, 95th percentile = {p95:.2f}, max = {max_val:.2f}). "
            "Arbitrary DN values are NOT calibrated sigma0. For scientific accuracy, calibrate Sentinel-1 GRD "
            "products using annotation LUTs (e.g., ESA SNAP) prior to analysis."
        )
        return "uncalibrated_dn", notes

    # Case 3: Non-negative values within linear power ratio range [0, ~10]
    if neg_fraction <= 0.05 and p95 <= 10.0 and max_val > 0.0:
        notes.append(
            f"Detected linear backscatter intensity scale (median = {p50:.4f}, 95th percentile = {p95:.4f}). "
            "Applying sigma0_dB = 10 * log10(sigma0_linear)."
        )
        return "linear_sigma0", notes

    # Fallback if negative values exist outside typical range
    if neg_fraction > 0.05:
        notes.append(
            f"Input treated as decibel (dB) scale based on negative pixel distribution (median = {p50:.2f})."
        )
        return "dB", notes

    notes.append("Input treated as linear intensity scale; converting positive values to dB.")
    return "linear_sigma0", notes


def to_sigma0_db(
    arr: np.ndarray,
    input_scale: InputScaleOption = "auto",
    nodata: Optional[float] = None,
    metadata_tags: Optional[dict[str, str]] = None,
    allow_uncalibrated_dn_log: bool = False,
) -> CalibrationResult:
    """
    Convert SAR backscatter array to decibels (dB) safely without double-converting.

    Rules:
    - If already in dB, returns a sanitized float64 copy with conversion_applied=False.
    - If linear_sigma0, computes 10 * log10(sigma0_linear) for strictly positive pixels (> 0),
      while masking <= 0, NaN, Inf, and nodata as np.nan.
    - If auto-detected as uncalibrated_dn, raises RasterValidationError unless
      allow_uncalibrated_dn_log=True is explicitly requested (in which case it logs a prominent
      scientific warning that 10*log10(DN) is relative decibel power, not calibrated sigma0).
    """
    clean = sanitize_sar_array(arr, nodata=nodata)
    finite_mask = np.isfinite(clean)

    if np.count_nonzero(finite_mask) == 0:
        raise RasterValidationError("Cannot process backscatter: all pixels are NaN or nodata.")

    notes: list[str] = []
    if input_scale == "auto":
        detected_scale, detect_notes = detect_backscatter_scale(clean, nodata=None, metadata_tags=metadata_tags)
        notes.extend(detect_notes)
    elif input_scale == "dB":
        detected_scale = "dB"
        notes.append("User specified input is already in decibels (dB). Skipping 10*log10 conversion.")
    elif input_scale == "linear_sigma0":
        detected_scale = "linear_sigma0"
        # Guard against obvious double-conversion if user mistakenly forced linear_sigma0 on an all-negative dB image
        neg_frac = float(np.mean(clean[finite_mask] < 0))
        if neg_frac > 0.50:
            raise RasterValidationError(
                f"Input scale was set to 'linear_sigma0', but {neg_frac * 100:.1f}% of valid pixels are negative. "
                "This raster appears to be in dB already; refuses to apply log10 to negative dB values."
            )
        notes.append("User specified calibrated linear sigma0 intensity. Applying 10 * log10(sigma0_linear).")
    else:
        raise ValueError(f"Unsupported input_scale option: {input_scale}")

    if detected_scale == "dB":
        return CalibrationResult(
            sigma0_db=clean,
            detected_scale="dB",
            conversion_applied=False,
            valid_mask=finite_mask,
            invalid_nonpositive_count=0,
            notes=notes,
        )

    if detected_scale == "uncalibrated_dn" and not allow_uncalibrated_dn_log:
        raise RasterValidationError(
            "Uncalibrated Digital Numbers (DN) detected. Raw DN values cannot be treated as calibrated sigma0 without "
            "sensor LUT calibration. Enable 'Allow relative dB scaling for uncalibrated DN' only for exploratory visualization."
        )

    # Linear to dB conversion: sigma0_dB = 10 * log10(sigma0_linear)
    positive_mask = finite_mask & (clean > 0.0)
    nonpositive_count = int(np.count_nonzero(finite_mask & (clean <= 0.0)))

    if np.count_nonzero(positive_mask) == 0:
        raise RasterValidationError(
            "Linear backscatter conversion failed: no strictly positive (> 0) pixels found."
        )

    if nonpositive_count > 0:
        notes.append(
            f"Masked {nonpositive_count} non-positive (<= 0) pixel(s) as NaN prior to logarithmic conversion."
        )

    sigma0_db = np.full_like(clean, np.nan, dtype=np.float64)
    sigma0_db[positive_mask] = 10.0 * np.log10(clean[positive_mask])

    return CalibrationResult(
        sigma0_db=sigma0_db,
        detected_scale=detected_scale,
        conversion_applied=True,
        valid_mask=positive_mask,
        invalid_nonpositive_count=nonpositive_count,
        notes=notes,
    )

