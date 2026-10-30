"""Tests for the adaptive Lee speckle filter (3x3, 5x5, 7x7 windows, edge preservation, and nodata)."""

import numpy as np
import pytest

from core.speckle_filter import lee_speckle_filter


@pytest.mark.parametrize("window_size", [3, 5, 7])
def test_lee_filter_window_sizes_reduce_speckle_variance(window_size: int):
    """Verify 3x3, 5x5, and 7x7 Lee filters reduce speckle variance in homogeneous regions."""
    rng = np.random.default_rng(123)
    enl = 4.4
    clean_linear = np.full((64, 64), 0.1, dtype=np.float64)  # -10 dB homogeneous region
    noisy_linear = clean_linear * rng.gamma(shape=enl, scale=1.0 / enl, size=clean_linear.shape)
    noisy_db = 10.0 * np.log10(noisy_linear)

    filtered_db = lee_speckle_filter(noisy_db, window_size=window_size, enl=enl, is_db=True)

    assert filtered_db.shape == noisy_db.shape
    assert np.nanstd(filtered_db) < np.nanstd(noisy_db) * 0.75
    np.testing.assert_allclose(np.nanmean(filtered_db), np.nanmean(noisy_db), atol=1.0)


def test_lee_filter_preserves_sharp_edges():
    """Verify high-contrast step edge between land (-8 dB) and water (-22 dB) remains sharp."""
    img_db = np.full((50, 50), -8.0, dtype=np.float64)
    img_db[:, 25:] = -22.0

    filtered = lee_speckle_filter(img_db, window_size=5, enl=4.4, is_db=True)

    # Away from the boundary, values should be exact
    np.testing.assert_allclose(filtered[:, :20], -8.0, atol=1e-4)
    np.testing.assert_allclose(filtered[:, 30:], -22.0, atol=1e-4)

    # Immediately adjacent to the step edge (columns 24 and 25), contrast must remain large (> 10 dB)
    edge_contrast = float(np.mean(filtered[:, 24] - filtered[:, 25]))
    assert edge_contrast > 10.0


def test_lee_filter_handles_nodata_and_nans():
    """Ensure NaN and nodata pixels do not corrupt neighboring valid pixels and remain NaN."""
    img_db = np.full((32, 32), -12.0, dtype=np.float64)
    img_db[10:15, 10:15] = np.nan
    img_db[0, 0] = -9999.0

    filtered = lee_speckle_filter(img_db, window_size=5, enl=4.4, is_db=True, nodata=-9999.0)

    assert np.all(np.isnan(filtered[10:15, 10:15]))
    assert np.isnan(filtered[0, 0])
    # Neighboring valid pixel at (9, 10) must still be finite and close to -12 dB
    assert np.isfinite(filtered[9, 10])
    np.testing.assert_allclose(filtered[9, 10], -12.0, atol=1e-3)


def test_lee_filter_rejects_invalid_window_size():
    """Verify unsupported window sizes raise ValueError."""
    img = np.zeros((16, 16), dtype=np.float64)
    with pytest.raises(ValueError, match="Invalid window_size"):
        lee_speckle_filter(img, window_size=9)

