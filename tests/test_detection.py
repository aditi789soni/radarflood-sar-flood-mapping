"""Tests for flood threshold classification, Otsu's method, morphological operations, and bitemporal change."""

import numpy as np

from core.change_detection import run_bitemporal_change_detection
from core.flood_detection import classify_water_mask, compute_otsu_threshold_db


def test_vv_threshold_and_morphological_cleanup():
    """Verify VV threshold classification detects flood patches and removes isolated single-pixel speckle."""
    vv_db = np.full((60, 60), -9.0, dtype=np.float64)
    # True flood block (20x20 = 400 pixels)
    vv_db[20:40, 20:40] = -20.5
    # Isolated single-pixel speckle noise spike
    vv_db[5, 5] = -21.0
    # Internal single-pixel pinhole inside flood block
    vv_db[30, 30] = -10.0

    res = classify_water_mask(
        vv_db=vv_db,
        vv_threshold_db=-16.0,
        use_otsu=False,
        apply_opening=True,
        apply_closing=True,
        morph_kernel_size=3,
        min_region_pixels=16,
    )

    # Single-pixel false positive at (5, 5) must be removed
    assert not res.water_mask[5, 5]
    # Internal pinhole at (30, 30) must be closed/filled
    assert res.water_mask[30, 30]
    # Core of the flood block must be classified as water
    assert np.all(res.water_mask[22:38, 22:38])


def test_otsu_threshold_on_bimodal_sar_distribution():
    """Verify Otsu threshold accurately separates a bimodal water (-20 dB) and land (-9 dB) scene."""
    rng = np.random.default_rng(99)
    water_pixels = rng.normal(loc=-20.5, scale=1.2, size=(50, 50))
    land_pixels = rng.normal(loc=-9.0, scale=1.5, size=(50, 50))
    bimodal_scene = np.vstack([water_pixels, land_pixels])

    otsu_t = compute_otsu_threshold_db(bimodal_scene)
    assert -18.0 <= otsu_t <= -12.0


def test_bitemporal_change_separates_permanent_water_and_new_flood():
    """Verify bitemporal change detection separates permanent river channels from newly inundated floodplains."""
    pre_vv = np.full((80, 80), -9.5, dtype=np.float64)
    post_vv = np.full((80, 80), -9.5, dtype=np.float64)

    # Permanent river channel in columns 10..20 (low backscatter in BOTH Pre and Post)
    pre_vv[:, 10:20] = -21.5
    post_vv[:, 10:20] = -21.0

    # Newly inundated floodplain in rows 30..60, cols 35..65 (land in Pre, water in Post)
    post_vv[30:60, 35:65] = -19.5

    # Invalid nodata strip at bottom
    pre_vv[75:, :] = np.nan
    post_vv[75:, :] = -22.0

    change = run_bitemporal_change_detection(
        pre_vv_db=pre_vv,
        post_vv_db=post_vv,
        post_vv_threshold_db=-16.0,
        pre_perm_water_threshold_db=-17.0,
        delta_db_threshold=-3.0,
        apply_opening=True,
        apply_closing=True,
        morph_kernel_size=3,
        min_region_pixels=10,
    )

    # Permanent river must be class 1 (Permanent Water) and NOT in new_flood_mask
    assert np.all(change.permanent_water_mask[20:50, 12:18])
    assert not np.any(change.new_flood_mask[:, 10:20])
    assert np.all(change.classification_map[20:50, 12:18] == 1)

    # Floodplain must be class 2 (Newly Inundated Floodwater)
    assert np.all(change.new_flood_mask[35:55, 40:60])
    assert np.all(change.classification_map[35:55, 40:60] == 2)

    # Invalid/NaN strip must be excluded (-1 in classification_map, False in masks)
    assert not np.any(change.new_flood_mask[75:, :])
    assert not np.any(change.permanent_water_mask[75:, :])
    assert np.all(change.classification_map[75:, :] == -1)

