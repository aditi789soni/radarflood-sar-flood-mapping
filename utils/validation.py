"""Input validation and compatibility checks for SAR GeoTIFF rasters."""

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.coords import BoundingBox
from rasterio.warp import transform_bounds


class RasterValidationError(ValueError):
    """Raised when an input SAR raster fails scientific or structural validation."""


@dataclass
class ValidationResult:
    """Structured result of raster validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_sar_array(
    arr: np.ndarray,
    nodata: Optional[float] = None,
    name: str = "SAR Raster",
) -> ValidationResult:
    """Validate a 2D or 3D numpy array representing SAR backscatter."""
    errors: list[str] = []
    warnings: list[str] = []

    if arr is None:
        return ValidationResult(False, [f"{name}: Array is None."])

    if not isinstance(arr, np.ndarray):
        return ValidationResult(False, [f"{name}: Expected numpy.ndarray, got {type(arr).__name__}."])

    if arr.ndim not in (2, 3):
        return ValidationResult(
            False,
            [f"{name}: Expected 2D (H, W) or 3D (B, H, W) array, got {arr.ndim}D shape {arr.shape}."],
        )

    height, width = arr.shape[-2], arr.shape[-1]
    if height < 4 or width < 4:
        errors.append(
            f"{name}: Raster dimensions ({height}x{width}) are too small for spatial filtering (minimum 4x4)."
        )

    valid_mask = np.isfinite(arr)
    if nodata is not None and not np.isnan(nodata):
        valid_mask &= ~np.isclose(arr, nodata, rtol=1e-5, atol=1e-8)

    valid_count = int(np.count_nonzero(valid_mask))
    total_count = int(arr.size)

    if valid_count == 0:
        errors.append(f"{name}: Raster contains 0 valid pixels (all pixels are NaN, Inf, or nodata).")
    elif valid_count / total_count < 0.05:
        warnings.append(
            f"{name}: Less than 5% of pixels ({valid_count}/{total_count}) contain valid data."
        )

    if valid_count > 0:
        valid_vals = arr[valid_mask]
        if np.all(valid_vals == 0):
            errors.append(f"{name}: All valid pixels are identically zero (0.0).")
        elif np.nanstd(valid_vals) < 1e-9:
            warnings.append(f"{name}: Valid pixels have near-zero variance (constant raster).")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata={
            "shape": arr.shape,
            "valid_pixels": valid_count,
            "total_pixels": total_count,
            "valid_fraction": float(valid_count / total_count) if total_count > 0 else 0.0,
        },
    )


def validate_raster_dataset(
    dataset: rasterio.DatasetReader,
    name: str = "Input GeoTIFF",
    require_crs: bool = True,
) -> ValidationResult:
    """Validate an open rasterio dataset for CRS, dimensions, bands, and pixel validity."""
    errors: list[str] = []
    warnings: list[str] = []

    if dataset is None:
        return ValidationResult(False, [f"{name}: Dataset is None."])

    if dataset.width < 4 or dataset.height < 4:
        errors.append(
            f"{name}: Dimensions ({dataset.height}x{dataset.width}) are too small for SAR analysis."
        )

    if dataset.count < 1:
        errors.append(f"{name}: Dataset contains no raster bands.")

    crs: Optional[CRS] = dataset.crs
    if crs is None:
        if require_crs:
            errors.append(
                f"{name}: Missing Coordinate Reference System (CRS). Georeferenced GeoTIFF is required for area calculation and spatial alignment."
            )
        else:
            warnings.append(f"{name}: Missing CRS; defaulting to pixel coordinates.")

    # Check transform validity
    transform = dataset.transform
    if transform is None or transform.is_identity:
        if require_crs:
            warnings.append(
                f"{name}: Affine transform is identity; raster may lack proper georeferencing."
            )

    # Read first band to verify non-empty data
    try:
        band1 = dataset.read(1, masked=False).astype(np.float64)
        arr_res = validate_sar_array(band1, nodata=dataset.nodata, name=name)
        errors.extend(arr_res.errors)
        warnings.extend(arr_res.warnings)
    except Exception as exc:
        errors.append(f"{name}: Failed to read raster band data: {exc}")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata={
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "crs": crs.to_string() if crs else None,
            "nodata": dataset.nodata,
            "bounds": tuple(dataset.bounds) if dataset.bounds else None,
        },
    )


def check_bitemporal_compatibility(
    pre_crs: Optional[CRS],
    pre_bounds: BoundingBox,
    pre_shape: tuple[int, int],
    post_crs: Optional[CRS],
    post_bounds: BoundingBox,
    post_shape: tuple[int, int],
) -> ValidationResult:
    """Check whether pre-flood and post-flood rasters overlap and whether alignment is required."""
    errors: list[str] = []
    warnings: list[str] = []
    needs_reprojection = False

    if pre_crs is None or post_crs is None:
        errors.append(
            "Both pre-flood and post-flood rasters must have a valid Coordinate Reference System (CRS) for bitemporal analysis."
        )
        return ValidationResult(False, errors, warnings, {"needs_reprojection": False})

    if pre_crs != post_crs:
        needs_reprojection = True
        warnings.append(
            f"Pre-flood CRS ({pre_crs.to_string()}) differs from Post-flood CRS ({post_crs.to_string()}). "
            "Post-flood raster will be automatically reprojected and aligned to the Pre-flood grid."
        )

    if pre_shape != post_shape or pre_bounds != post_bounds:
        needs_reprojection = True
        if pre_shape != post_shape:
            warnings.append(
                f"Raster dimensions differ (Pre: {pre_shape}, Post: {post_shape}). "
                "Automatic spatial co-registration to the Pre-flood reference grid will be applied."
            )

    # Verify geographic overlap in EPSG:4326
    try:
        pre_wgs84 = transform_bounds(pre_crs, "EPSG:4326", *pre_bounds, densify_pts=21)
        post_wgs84 = transform_bounds(post_crs, "EPSG:4326", *post_bounds, densify_pts=21)

        overlap_west = max(pre_wgs84[0], post_wgs84[0])
        overlap_south = max(pre_wgs84[1], post_wgs84[1])
        overlap_east = min(pre_wgs84[2], post_wgs84[2])
        overlap_north = min(pre_wgs84[3], post_wgs84[3])

        if overlap_west >= overlap_east or overlap_south >= overlap_north:
            errors.append(
                "Pre-flood and post-flood rasters do not overlap geographically! "
                f"Pre bounds (WGS84): {tuple(round(x, 4) for x in pre_wgs84)}, "
                f"Post bounds (WGS84): {tuple(round(x, 4) for x in post_wgs84)}."
            )
    except Exception as exc:
        warnings.append(f"Could not verify WGS84 bounding box overlap: {exc}")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metadata={"needs_reprojection": needs_reprojection},
    )

