"""
Accurate geospatial flood area calculation and statistical summary generation.

Always uses a projected metric CRS (e.g., appropriate UTM zone in meters) rather than
calculating area in geographic degrees. Excludes invalid / nodata pixels from all percentages.
"""

from dataclasses import asdict, dataclass
from typing import Optional
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.coords
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import box

from utils.raster_utils import get_suitable_utm_crs
from utils.validation import RasterValidationError


@dataclass
class FloodAreaMetrics:
    """Comprehensive inundation area and polygon analytics."""

    metric_crs_epsg: str
    pixel_area_m2: float
    total_scene_pixels: int
    valid_scene_pixels: int
    invalid_nodata_pixels: int
    valid_scene_area_km2: float
    new_flood_pixels: int
    new_flood_area_km2: float
    new_flood_area_ha: float
    new_flood_percentage: float
    permanent_water_pixels: int
    permanent_water_area_km2: float
    permanent_water_percentage: float
    total_post_water_area_km2: float
    total_post_water_percentage: float
    vector_polygon_count: int
    vector_new_flood_area_km2: float
    largest_polygon_km2: float
    mean_polygon_km2: float
    median_polygon_km2: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_projected_pixel_area_m2(
    crs: CRS,
    transform: Affine,
    width: int,
    height: int,
) -> tuple[float, CRS]:
    """
    Compute exact average pixel area in square meters by projecting the raster bounding
    box into a suitable metric CRS (UTM / meters). Never computes area in square degrees.
    """
    if crs is None:
        raise RasterValidationError("CRS is required to compute projected metric area.")

    if width <= 0 or height <= 0:
        raise ValueError("Raster width and height must be positive.")

    left, bottom, right, top = rasterio.transform.array_bounds(height, width, transform)
    bounds = rasterio.coords.BoundingBox(left, bottom, right, top)
    metric_crs = get_suitable_utm_crs(crs, bounds)

    bbox_gdf = gpd.GeoDataFrame({"geometry": [box(left, bottom, right, top)]}, crs=crs)
    bbox_metric = bbox_gdf.to_crs(metric_crs)
    total_area_m2 = float(bbox_metric.geometry.area.iloc[0])
    pixel_area_m2 = total_area_m2 / float(width * height)

    return pixel_area_m2, metric_crs


def calculate_inundation_metrics(
    valid_mask: np.ndarray,
    new_flood_mask: np.ndarray,
    crs: CRS,
    transform: Affine,
    permanent_water_mask: Optional[np.ndarray] = None,
    polygons_gdf: Optional[gpd.GeoDataFrame] = None,
) -> FloodAreaMetrics:
    """
    Calculate pixel-based and vector-based flood area statistics in km^2 and hectares,
    strictly excluding invalid/nodata pixels.
    """
    if valid_mask.shape != new_flood_mask.shape:
        raise ValueError("valid_mask and new_flood_mask must have the same shape.")

    height, width = valid_mask.shape
    pixel_area_m2, metric_crs = compute_projected_pixel_area_m2(crs, transform, width, height)

    v_mask = valid_mask.astype(bool)
    f_mask = new_flood_mask.astype(bool) & v_mask
    p_mask = (permanent_water_mask.astype(bool) & v_mask) if permanent_water_mask is not None else np.zeros_like(v_mask)

    total_pixels = int(v_mask.size)
    valid_pixels = int(np.count_nonzero(v_mask))
    invalid_pixels = total_pixels - valid_pixels

    new_flood_pixels = int(np.count_nonzero(f_mask))
    perm_water_pixels = int(np.count_nonzero(p_mask))
    total_water_pixels = int(np.count_nonzero(f_mask | p_mask))

    valid_area_km2 = (valid_pixels * pixel_area_m2) / 1e6
    new_flood_km2 = (new_flood_pixels * pixel_area_m2) / 1e6
    new_flood_ha = (new_flood_pixels * pixel_area_m2) / 1e4
    perm_water_km2 = (perm_water_pixels * pixel_area_m2) / 1e6
    total_water_km2 = (total_water_pixels * pixel_area_m2) / 1e6

    if valid_pixels > 0:
        new_flood_pct = (new_flood_pixels / valid_pixels) * 100.0
        perm_water_pct = (perm_water_pixels / valid_pixels) * 100.0
        total_water_pct = (total_water_pixels / valid_pixels) * 100.0
    else:
        new_flood_pct = 0.0
        perm_water_pct = 0.0
        total_water_pct = 0.0

    # Vector polygon metrics (for Newly Inundated Floodwater class_id == 2)
    poly_count = 0
    vec_flood_km2 = 0.0
    largest_km2 = 0.0
    mean_km2 = 0.0
    median_km2 = 0.0

    if polygons_gdf is not None and not polygons_gdf.empty:
        flood_polys = polygons_gdf[polygons_gdf["class_id"] == 2] if "class_id" in polygons_gdf.columns else polygons_gdf
        if not flood_polys.empty:
            poly_count = int(len(flood_polys))
            areas = flood_polys["area_km2"].to_numpy(dtype=float)
            vec_flood_km2 = float(np.sum(areas))
            largest_km2 = float(np.max(areas))
            mean_km2 = float(np.mean(areas))
            median_km2 = float(np.median(areas))

    return FloodAreaMetrics(
        metric_crs_epsg=metric_crs.to_string(),
        pixel_area_m2=round(pixel_area_m2, 4),
        total_scene_pixels=total_pixels,
        valid_scene_pixels=valid_pixels,
        invalid_nodata_pixels=invalid_pixels,
        valid_scene_area_km2=round(valid_area_km2, 4),
        new_flood_pixels=new_flood_pixels,
        new_flood_area_km2=round(new_flood_km2, 4),
        new_flood_area_ha=round(new_flood_ha, 2),
        new_flood_percentage=round(new_flood_pct, 2),
        permanent_water_pixels=perm_water_pixels,
        permanent_water_area_km2=round(perm_water_km2, 4),
        permanent_water_percentage=round(perm_water_pct, 2),
        total_post_water_area_km2=round(total_water_km2, 4),
        total_post_water_percentage=round(total_water_pct, 2),
        vector_polygon_count=poly_count,
        vector_new_flood_area_km2=round(vec_flood_km2, 4),
        largest_polygon_km2=round(largest_km2, 4),
        mean_polygon_km2=round(mean_km2, 4),
        median_polygon_km2=round(median_km2, 4),
    )


def metrics_to_csv_bytes(
    metrics: FloodAreaMetrics,
    polygons_gdf: Optional[gpd.GeoDataFrame] = None,
) -> tuple[bytes, bytes]:
    """
    Generate CSV byte streams for:
    1. Summary inundation metrics table.
    2. Individual flood polygon inventory table (without heavy WKT geometry).
    """
    summary_df = pd.DataFrame([metrics.to_dict()])
    summary_csv = summary_df.to_csv(index=False).encode("utf-8")

    if polygons_gdf is not None and not polygons_gdf.empty:
        cols = [c for c in polygons_gdf.columns if c != "geometry"]
        poly_df = pd.DataFrame(polygons_gdf[cols])
    else:
        poly_df = pd.DataFrame(
            columns=[
                "polygon_id",
                "class_id",
                "class_name",
                "area_m2",
                "area_km2",
                "area_ha",
                "perimeter_m",
                "centroid_lat",
                "centroid_lon",
            ]
        )
    poly_csv = poly_df.to_csv(index=False).encode("utf-8")
    return summary_csv, poly_csv

