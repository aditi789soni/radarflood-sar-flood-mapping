"""Tests for flood polygon vectorization, projected metric area calculation, and GeoJSON/CSV exports."""

import json
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from core.area_calculation import calculate_inundation_metrics, metrics_to_csv_bytes
from core.vectorization import export_polygons_to_geojson, vectorize_flood_mask
from utils.raster_utils import array_to_geotiff_bytes, generate_synthetic_sar_pair, load_sar_geotiff


def test_projected_area_calculation_and_polygon_generation():
    """
    Create a known 1.0 km x 1.0 km (1.0 km^2) flood block on a 10m UTM grid
    and verify both pixel-based and vector-based area calculations equal 1.0 km^2.
    """
    # 200 x 200 grid at 10m resolution = 2000m x 2000m (4.0 km^2 total scene)
    width, height = 200, 200
    pixel_size_m = 10.0
    crs = CRS.from_epsg(32646)  # UTM Zone 46N (meters)
    transform = from_bounds(600000.0, 2960000.0, 602000.0, 2962000.0, width, height)

    valid_mask = np.ones((height, width), dtype=bool)
    new_flood_mask = np.zeros((height, width), dtype=bool)
    # 100 x 100 pixels at 10m resolution = 1000m x 1000m = 1,000,000 m^2 = 1.0 km^2
    new_flood_mask[50:150, 50:150] = True

    gdf = vectorize_flood_mask(
        mask_or_class_map=new_flood_mask,
        crs=crs,
        transform=transform,
        min_polygon_area_m2=100.0,
    )

    assert len(gdf) == 1
    np.testing.assert_allclose(gdf["area_km2"].iloc[0], 1.0, rtol=1e-3)
    np.testing.assert_allclose(gdf["area_ha"].iloc[0], 100.0, rtol=1e-3)

    metrics = calculate_inundation_metrics(
        valid_mask=valid_mask,
        new_flood_mask=new_flood_mask,
        crs=crs,
        transform=transform,
        polygons_gdf=gdf,
    )

    np.testing.assert_allclose(metrics.valid_scene_area_km2, 4.0, rtol=1e-3)
    np.testing.assert_allclose(metrics.new_flood_area_km2, 1.0, rtol=1e-3)
    np.testing.assert_allclose(metrics.new_flood_percentage, 25.0, rtol=1e-3)
    assert metrics.vector_polygon_count == 1
    np.testing.assert_allclose(metrics.vector_new_flood_area_km2, 1.0, rtol=1e-3)


def test_geographic_crs_area_is_reprojected_to_metric_utm():
    """Ensure rasters in EPSG:4326 (degrees) are projected to UTM meters rather than computing degree-squared areas."""
    crs_wgs84 = CRS.from_epsg(4326)
    # ~0.05 deg x 0.05 deg near equator (~5.55 km x 5.55 km ~= 30.8 km^2)
    transform = from_bounds(94.0, 26.5, 94.05, 26.55, 100, 100)
    valid_mask = np.ones((100, 100), dtype=bool)
    flood_mask = np.zeros((100, 100), dtype=bool)
    flood_mask[25:75, 25:75] = True  # 25% of scene

    gdf = vectorize_flood_mask(flood_mask, crs=crs_wgs84, transform=transform, min_polygon_area_m2=100.0)
    metrics = calculate_inundation_metrics(valid_mask, flood_mask, crs=crs_wgs84, transform=transform, polygons_gdf=gdf)

    assert "32646" in metrics.metric_crs_epsg
    # Area must be in realistic km^2 (~6.9 km^2), never tiny degree^2 (~0.000625)
    assert 5.0 < metrics.new_flood_area_km2 < 9.0
    np.testing.assert_allclose(metrics.new_flood_percentage, 25.0, atol=0.1)


def test_empty_mask_vectorization_and_geojson_export():
    """Ensure a mask with zero flood pixels produces an empty GeoDataFrame and valid empty GeoJSON FeatureCollection."""
    crs = CRS.from_epsg(32646)
    transform = from_bounds(600000.0, 2960000.0, 601000.0, 2961000.0, 50, 50)
    empty_mask = np.zeros((50, 50), dtype=bool)
    valid_mask = np.ones((50, 50), dtype=bool)

    gdf = vectorize_flood_mask(empty_mask, crs=crs, transform=transform)
    assert gdf.empty

    geojson_str = export_polygons_to_geojson(gdf)
    parsed = json.loads(geojson_str)
    assert parsed["type"] == "FeatureCollection"
    assert parsed["features"] == []

    metrics = calculate_inundation_metrics(valid_mask, empty_mask, crs=crs, transform=transform, polygons_gdf=gdf)
    assert metrics.new_flood_area_km2 == 0.0
    assert metrics.vector_polygon_count == 0

    summary_csv, poly_csv = metrics_to_csv_bytes(metrics, gdf)
    assert b"new_flood_area_km2" in summary_csv
    assert b"polygon_id" in poly_csv


def test_synthetic_demo_pair_and_geotiff_roundtrip():
    """Verify synthetic SAR generator and in-memory GeoTIFF serialization/deserialization."""
    pre_ds, post_ds = generate_synthetic_sar_pair(height=80, width=80, pixel_size_m=20.0)
    assert pre_ds.is_demo is True
    assert post_ds.is_demo is True
    assert pre_ds.has_vh is True

    tiff_bytes = array_to_geotiff_bytes(
        np.stack([pre_ds.vv_band, pre_ds.vh_band]),
        crs=pre_ds.crs,
        transform=pre_ds.transform,
        band_descriptions=["VV", "VH"],
        tags=pre_ds.tags,
    )
    reloaded = load_sar_geotiff(tiff_bytes, name="roundtrip.tif")
    assert reloaded.shape == (80, 80)
    assert reloaded.has_vh is True
    assert reloaded.is_demo is True
    np.testing.assert_allclose(reloaded.vv_band, pre_ds.vv_band, rtol=1e-5)


def test_full_pipeline_and_streamlit_dashboard_execution():
    """Verify full end-to-end SAR pipeline execution, map/chart builders, exports, and Streamlit AppTest."""
    from streamlit.testing.v1 import AppTest
    from app import execute_sar_pipeline
    from visualization.maps import build_inundation_folium_map
    from visualization.charts import (
        create_backscatter_histogram,
        create_delta_histogram,
        create_vv_vh_comparison_chart,
        create_area_breakdown_chart,
    )

    pre_ds, post_ds = generate_synthetic_sar_pair(height=120, width=120, pixel_size_m=20.0)
    res = execute_sar_pipeline(
        pre_ds_raw=pre_ds,
        post_ds_raw=post_ds,
        input_scale="auto",
        allow_uncalibrated_dn=False,
        apply_lee=True,
        lee_window=5,
        lee_enl=4.4,
        vv_threshold_db=-16.0,
        use_otsu=False,
        include_vh=False,
        vh_threshold_db=-22.0,
        vh_combine_mode="AND",
        pre_perm_water_db=-17.0,
        delta_drop_db=-3.0,
        apply_opening=True,
        apply_closing=True,
        morph_kernel=3,
        min_flood_pixels=16,
        min_polygon_area_m2=2500.0,
    )

    metrics = res["area_metrics"]
    assert metrics.new_flood_area_km2 > 0.1
    assert metrics.permanent_water_area_km2 > 0.1
    assert metrics.vector_polygon_count >= 1

    # Verify Folium map and Plotly charts render cleanly
    fmap = build_inundation_folium_map(
        wgs84_bounds=pre_ds.wgs84_bounds,
        pre_vv_db=res["pre_vv_db"],
        post_vv_db=res["post_vv_db"],
        classification_map=res["change_res"].classification_map,
        polygons_gdf=res["polygons_gdf"],
        is_demo=True,
    )
    assert fmap is not None
    assert create_backscatter_histogram(res["pre_vv_db"], res["post_vv_db"], -16.0) is not None
    assert create_delta_histogram(res["change_res"].delta_vv_db, -3.0) is not None
    assert create_vv_vh_comparison_chart(res["post_vv_db"], res["post_vh_db"], res["change_res"].classification_map) is not None
    assert create_area_breakdown_chart(metrics) is not None

    # Execute Streamlit AppTest to verify the dashboard loads and renders with zero runtime exceptions
    from pathlib import Path
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception, f"Streamlit dashboard raised an exception: {at.exception}"


