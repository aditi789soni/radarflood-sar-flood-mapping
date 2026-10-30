"""Tests for SAR radiometric calibration, scale detection, spatial alignment, and input validation."""

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from core.calibration import detect_backscatter_scale, to_sigma0_db
from core.preprocessing import align_bitemporal_datasets, reproject_array_to_grid
from utils.raster_utils import (
    SARDataset,
    array_to_geotiff_bytes,
    estimate_resolution_meters,
    generate_synthetic_sar_pair,
    load_sar_geotiff,
)
from utils.validation import RasterValidationError, validate_sar_array


def test_linear_to_db_conversion():
    """Verify exact 10 * log10(sigma0_linear) conversion for positive linear intensity."""
    linear_arr = np.array(
        [
            [0.01, 0.1, 1.0],
            [0.001, 0.5, 0.025],
            [0.01, 0.1, 1.0],
            [0.001, 0.5, 0.025],
        ],
        dtype=np.float64,
    )
    res = to_sigma0_db(linear_arr, input_scale="auto")
    assert res.detected_scale == "linear_sigma0"
    assert res.conversion_applied is True
    np.testing.assert_allclose(res.sigma0_db[0, 0], -20.0, atol=1e-6)
    np.testing.assert_allclose(res.sigma0_db[0, 1], -10.0, atol=1e-6)
    np.testing.assert_allclose(res.sigma0_db[0, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(res.sigma0_db[1, 0], -30.0, atol=1e-6)


def test_already_db_input_not_converted_twice():
    """Ensure inputs already in dB are identified and never undergo log10 conversion twice."""
    rng = np.random.default_rng(12)
    db_arr = rng.uniform(-24.0, -5.0, size=(32, 32))
    res = to_sigma0_db(db_arr, input_scale="auto")
    assert res.detected_scale == "dB"
    assert res.conversion_applied is False
    np.testing.assert_allclose(res.sigma0_db, db_arr, atol=1e-12)


def test_forced_linear_on_negative_db_raises_error():
    """Prevent accidental double log conversion if user forces linear_sigma0 on a negative dB array."""
    db_arr = np.full((16, 16), -15.0, dtype=np.float64)
    with pytest.raises(RasterValidationError, match="negative"):
        to_sigma0_db(db_arr, input_scale="linear_sigma0")


def test_zeros_negatives_and_nodata_handled_safely():
    """Ensure zeros, negative noise spikes, NaNs, and nodata values become NaN in dB output."""
    linear_arr = np.full((10, 10), 0.1, dtype=np.float64)
    linear_arr[0, 0] = 0.0
    linear_arr[0, 1] = -0.05
    linear_arr[0, 2] = -9999.0
    linear_arr[0, 3] = np.nan

    res = to_sigma0_db(linear_arr, input_scale="linear_sigma0", nodata=-9999.0)
    assert res.invalid_nonpositive_count == 2
    assert np.isnan(res.sigma0_db[0, 0])
    assert np.isnan(res.sigma0_db[0, 1])
    assert np.isnan(res.sigma0_db[0, 2])
    assert np.isnan(res.sigma0_db[0, 3])
    np.testing.assert_allclose(res.sigma0_db[5, 5], -10.0, atol=1e-6)


def test_uncalibrated_dn_detection_and_guard():
    """Verify arbitrary large Digital Numbers (DN) are flagged and blocked unless explicitly allowed."""
    rng = np.random.default_rng(7)
    dn_arr = rng.uniform(200.0, 4500.0, size=(20, 20))
    scale, _ = detect_backscatter_scale(dn_arr)
    assert scale == "uncalibrated_dn"

    with pytest.raises(RasterValidationError, match="Uncalibrated Digital Numbers"):
        to_sigma0_db(dn_arr, input_scale="auto", allow_uncalibrated_dn_log=False)

    res_allowed = to_sigma0_db(dn_arr, input_scale="auto", allow_uncalibrated_dn_log=True)
    assert res_allowed.conversion_applied is True


def test_raster_alignment_different_shapes_and_resolutions():
    """Verify bitemporal co-registration aligns a coarser Post-flood raster onto the Pre-flood grid."""
    pre_ds, post_ds = generate_synthetic_sar_pair(height=60, width=60, pixel_size_m=20.0)

    # Create a Post-flood dataset with different dimensions (40x40) covering the same extent
    transform_40 = from_bounds(*pre_ds.bounds, 40, 40)
    res_40 = estimate_resolution_meters(pre_ds.crs, transform_40, pre_ds.bounds)
    post_coarse_vv = reproject_array_to_grid(
        post_ds.vv_band,
        src_crs=post_ds.crs,
        src_transform=post_ds.transform,
        dst_crs=post_ds.crs,
        dst_transform=transform_40,
        dst_shape=(40, 40),
    )

    post_coarse_ds = SARDataset(
        name="Post_Coarse_40x40.tif",
        vv_band=post_coarse_vv,
        vh_band=None,
        crs=post_ds.crs,
        transform=transform_40,
        bounds=post_ds.bounds,
        nodata=-9999.0,
        width=40,
        height=40,
        count=1,
        band_descriptions=("VV",),
        resolution_xy_m=res_40,
        wgs84_bounds=post_ds.wgs84_bounds,
    )

    aligned_pre, aligned_post, notes = align_bitemporal_datasets(pre_ds, post_coarse_ds)
    assert aligned_pre.shape == (60, 60)
    assert aligned_post.shape == (60, 60)
    assert aligned_post.transform == aligned_pre.transform
    assert len(notes) > 0


def test_invalid_inputs_and_missing_crs():
    """Ensure all-nodata arrays, all-zero arrays, and missing CRS GeoTIFFs raise clean validation errors."""
    all_nan = np.full((10, 10), np.nan)
    res_nan = validate_sar_array(all_nan)
    assert not res_nan.is_valid

    all_zero = np.zeros((10, 10))
    res_zero = validate_sar_array(all_zero)
    assert not res_zero.is_valid

    # Create a GeoTIFF without CRS and verify load_sar_geotiff rejects it
    raw_bytes = array_to_geotiff_bytes(
        np.ones((16, 16), dtype=np.float32),
        crs=None,
        transform=from_bounds(0, 0, 160, 160, 16, 16),
    )
    with pytest.raises(RasterValidationError, match="Missing Coordinate Reference System"):
        load_sar_geotiff(raw_bytes, name="no_crs.tif")

