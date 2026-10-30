"""
Georeferenced vectorization of binary and multi-class flood masks into GeoDataFrames and GeoJSON.
"""

import json
from typing import Optional
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.coords
from rasterio.crs import CRS
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape

from utils.config import CONFIG
from utils.raster_utils import get_suitable_utm_crs
from utils.validation import RasterValidationError

EMPTY_COLUMNS = [
    "polygon_id",
    "class_id",
    "class_name",
    "area_m2",
    "area_km2",
    "area_ha",
    "perimeter_m",
    "centroid_lat",
    "centroid_lon",
    "geometry",
]


def _create_empty_gdf(crs: CRS) -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame with the standard RadarFlood schema and CRS."""
    return gpd.GeoDataFrame(
        {col: pd.Series(dtype="float64" if "area" in col or "lat" in col or "lon" in col or "perim" in col else "object") for col in EMPTY_COLUMNS},
        geometry="geometry",
        crs=crs,
    )


def vectorize_flood_mask(
    mask_or_class_map: np.ndarray,
    crs: CRS,
    transform: Affine,
    bounds: Optional[rasterio.coords.BoundingBox] = None,
    min_polygon_area_m2: float = CONFIG.DEFAULT_MIN_POLYGON_AREA_M2,
    include_permanent_water: bool = False,
) -> gpd.GeoDataFrame:
    """
    Convert a binary flood mask (True/1 = flood) or a multi-class map
    (1 = Permanent Water, 2 = Newly Inundated Floodwater) into a georeferenced
    GeoDataFrame with accurate metric areas calculated in a projected UTM/metric CRS.
    """
    if crs is None:
        raise RasterValidationError("Cannot vectorize flood mask without a valid Coordinate Reference System (CRS).")

    if mask_or_class_map is None or mask_or_class_map.ndim != 2:
        raise ValueError("vectorize_flood_mask expects a 2D array.")

    # Determine if input is multi-class (contains 2) or binary boolean/0-1
    unique_vals = np.unique(mask_or_class_map[np.isfinite(mask_or_class_map)])
    is_multiclass = np.any(unique_vals == 2)

    if is_multiclass:
        work_arr = mask_or_class_map.astype(np.int16)
        if include_permanent_water:
            extract_mask = (work_arr == 1) | (work_arr == 2)
        else:
            extract_mask = work_arr == 2
    else:
        work_arr = np.where(mask_or_class_map.astype(bool), 2, 0).astype(np.int16)
        extract_mask = work_arr == 2

    if np.count_nonzero(extract_mask) == 0:
        return _create_empty_gdf(crs)

    records = []
    for geom_dict, val in shapes(work_arr, mask=extract_mask, transform=transform, connectivity=8):
        poly = shape(geom_dict)
        if poly.is_empty or not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        cls_id = int(val)
        cls_name = "Newly Inundated Floodwater" if cls_id == 2 else "Permanent Water"
        records.append({"class_id": cls_id, "class_name": cls_name, "geometry": poly})

    if not records:
        return _create_empty_gdf(crs)

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    # Project to suitable metric CRS (UTM / meters) for accurate area & perimeter calculation
    if bounds is None:
        total_bounds = gdf.total_bounds
        bounds = rasterio.coords.BoundingBox(*total_bounds)
    metric_crs = get_suitable_utm_crs(crs, bounds)

    gdf_metric = gdf.to_crs(metric_crs)
    gdf["area_m2"] = gdf_metric.geometry.area.astype(float)
    gdf["perimeter_m"] = gdf_metric.geometry.length.astype(float)

    # Filter out small noise polygons
    if min_polygon_area_m2 > 0:
        gdf = gdf[gdf["area_m2"] >= float(min_polygon_area_m2)].copy()

    if gdf.empty:
        return _create_empty_gdf(crs)

    gdf = gdf.sort_values(by="area_m2", ascending=False).reset_index(drop=True)
    gdf["polygon_id"] = np.arange(1, len(gdf) + 1, dtype=int)
    gdf["area_km2"] = gdf["area_m2"] / 1e6
    gdf["area_ha"] = gdf["area_m2"] / 1e4

    # Compute WGS84 centroids for reporting
    gdf_metric_filtered = gdf.to_crs(metric_crs)
    centroids_wgs84 = gpd.GeoSeries(gdf_metric_filtered.geometry.centroid, crs=metric_crs).to_crs("EPSG:4326")
    gdf["centroid_lon"] = np.round(centroids_wgs84.x.to_numpy(), 6)
    gdf["centroid_lat"] = np.round(centroids_wgs84.y.to_numpy(), 6)

    # Round numeric columns for clean export
    gdf["area_m2"] = gdf["area_m2"].round(2)
    gdf["area_km2"] = gdf["area_km2"].round(6)
    gdf["area_ha"] = gdf["area_ha"].round(4)
    gdf["perimeter_m"] = gdf["perimeter_m"].round(2)

    return gdf[EMPTY_COLUMNS]


def export_polygons_to_geojson(gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:4326") -> str:
    """
    Serialize a flood polygon GeoDataFrame to a valid RFC 7946 GeoJSON string (in EPSG:4326).
    Safely returns an empty FeatureCollection if the GeoDataFrame has 0 polygons.
    """
    if gdf is None or gdf.empty:
        return json.dumps(
            {
                "type": "FeatureCollection",
                "name": "RadarFlood_Inundation_Polygons",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": [],
            },
            indent=2,
        )

    gdf_out = gdf.to_crs(target_crs) if gdf.crs is not None else gdf
    return gdf_out.to_json(indent=2)

