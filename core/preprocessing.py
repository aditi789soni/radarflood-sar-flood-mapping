"""SAR raster preprocessing, nodata masking, and spatial co-registration."""

from typing import Optional
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

from utils.raster_utils import SARDataset, estimate_resolution_meters
from utils.validation import RasterValidationError, check_bitemporal_compatibility


def create_valid_mask(arr: np.ndarray, nodata: Optional[float] = None) -> np.ndarray:
    """Return a boolean mask where True indicates finite, non-nodata pixels."""
    mask = np.isfinite(arr)
    if nodata is not None and not np.isnan(nodata):
        mask &= ~np.isclose(arr, nodata, rtol=1e-5, atol=1e-8)
    return mask


def sanitize_sar_array(arr: np.ndarray, nodata: Optional[float] = None) -> np.ndarray:
    """Replace nodata and non-finite entries with np.nan in a float64 copy."""
    clean = np.array(arr, dtype=np.float64, copy=True)
    valid = create_valid_mask(clean, nodata=nodata)
    clean[~valid] = np.nan
    return clean


def reproject_array_to_grid(
    src_array: np.ndarray,
    src_crs: CRS,
    src_transform: Affine,
    dst_crs: CRS,
    dst_transform: Affine,
    dst_shape: tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Reproject and resample a 2D float array onto a target spatial grid, preserving NaNs."""
    if src_crs is None or dst_crs is None:
        raise RasterValidationError("Cannot reproject raster with missing CRS.")

    src_clean = np.array(src_array, dtype=np.float64, copy=True)
    dst_array = np.full(dst_shape, np.nan, dtype=np.float64)

    reproject(
        source=src_clean,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst_array


def align_bitemporal_datasets(
    pre_ds: SARDataset,
    post_ds: SARDataset,
) -> tuple[SARDataset, SARDataset, list[str]]:
    """
    Ensure pre-flood and post-flood SARDatasets share the exact same CRS,
    affine transform, and pixel grid dimensions. Reprojects post_ds onto
    the pre_ds reference grid if needed.
    """
    compat = check_bitemporal_compatibility(
        pre_crs=pre_ds.crs,
        pre_bounds=pre_ds.bounds,
        pre_shape=pre_ds.shape,
        post_crs=post_ds.crs,
        post_bounds=post_ds.bounds,
        post_shape=post_ds.shape,
    )
    if not compat.is_valid:
        raise RasterValidationError(" | ".join(compat.errors))

    notes = list(compat.warnings)
    if not compat.metadata.get("needs_reprojection", False):
        return pre_ds, post_ds, notes

    aligned_vv = reproject_array_to_grid(
        src_array=post_ds.vv_band,
        src_crs=post_ds.crs,
        src_transform=post_ds.transform,
        dst_crs=pre_ds.crs,
        dst_transform=pre_ds.transform,
        dst_shape=pre_ds.shape,
        resampling=Resampling.bilinear,
    )

    if np.count_nonzero(np.isfinite(aligned_vv)) == 0:
        raise RasterValidationError(
            "After reprojecting Post-flood raster to Pre-flood grid, 0 valid overlapping pixels remained."
        )

    aligned_vh: Optional[np.ndarray] = None
    if post_ds.vh_band is not None:
        aligned_vh = reproject_array_to_grid(
            src_array=post_ds.vh_band,
            src_crs=post_ds.crs,
            src_transform=post_ds.transform,
            dst_crs=pre_ds.crs,
            dst_transform=pre_ds.transform,
            dst_shape=pre_ds.shape,
            resampling=Resampling.bilinear,
        )

    res_xy = estimate_resolution_meters(pre_ds.crs, pre_ds.transform, pre_ds.bounds)
    aligned_post = SARDataset(
        name=post_ds.name,
        vv_band=aligned_vv,
        vh_band=aligned_vh,
        crs=pre_ds.crs,
        transform=pre_ds.transform,
        bounds=pre_ds.bounds,
        nodata=pre_ds.nodata,
        width=pre_ds.width,
        height=pre_ds.height,
        count=post_ds.count,
        band_descriptions=post_ds.band_descriptions,
        resolution_xy_m=res_xy,
        wgs84_bounds=pre_ds.wgs84_bounds,
        is_demo=post_ds.is_demo,
        tags={**post_ds.tags, "COREGISTERED_TO": pre_ds.name},
        warnings=post_ds.warnings + notes,
    )

    return pre_ds, aligned_post, notes

