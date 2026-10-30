"""Raster I/O utilities, metadata extraction, and synthetic SAR demo data generator."""

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import Affine, from_bounds
from rasterio.warp import transform_bounds
from pyproj import CRS as PyProjCRS, Transformer

from utils.config import CONFIG
from utils.validation import RasterValidationError, validate_raster_dataset


@dataclass
class SARDataset:
    """Container for a loaded SAR raster with VV and optional VH polarization bands."""

    name: str
    vv_band: np.ndarray
    vh_band: Optional[np.ndarray]
    crs: CRS
    transform: Affine
    bounds: rasterio.coords.BoundingBox
    nodata: Optional[float]
    width: int
    height: int
    count: int
    band_descriptions: tuple[str, ...]
    resolution_xy_m: tuple[float, float]
    wgs84_bounds: tuple[float, float, float, float]  # (west, south, east, north)
    is_demo: bool = False
    tags: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def has_vh(self) -> bool:
        return self.vh_band is not None

    @property
    def pixel_area_m2(self) -> float:
        return float(self.resolution_xy_m[0] * self.resolution_xy_m[1])


def estimate_resolution_meters(crs: CRS, transform: Affine, bounds: rasterio.coords.BoundingBox) -> tuple[float, float]:
    """Estimate pixel resolution (dx, dy) in meters for both projected and geographic CRS."""
    dx = abs(float(transform.a))
    dy = abs(float(transform.e))
    if dx == 0 or dy == 0:
        return (10.0, 10.0)

    proj_crs = PyProjCRS.from_user_input(crs.to_wkt())
    if proj_crs.is_projected:
        unit_factor = proj_crs.axis_info[0].unit_conversion_factor or 1.0
        return (dx * unit_factor, dy * unit_factor)

    # Geographic CRS (degrees): approximate meters at scene center latitude
    center_lat = 0.5 * (bounds.bottom + bounds.top)
    lat_rad = np.deg2rad(center_lat)
    meters_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    meters_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
    return (max(dx * abs(meters_per_deg_lon), 0.01), max(dy * abs(meters_per_deg_lat), 0.01))


def get_suitable_utm_crs(crs: CRS, bounds: rasterio.coords.BoundingBox) -> CRS:
    """Return an appropriate equal-area or UTM projected CRS for accurate metric area calculation."""
    proj_crs = PyProjCRS.from_user_input(crs.to_wkt())
    if proj_crs.is_projected:
        unit_name = (proj_crs.axis_info[0].unit_name or "").lower()
        if "metre" in unit_name or "meter" in unit_name:
            return crs

    wgs84_bounds = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    center_lon = 0.5 * (wgs84_bounds[0] + wgs84_bounds[2])
    center_lat = 0.5 * (wgs84_bounds[1] + wgs84_bounds[3])

    utm_zone = int(np.floor((center_lon + 180.0) / 6.0) % 60) + 1
    epsg_code = (32600 + utm_zone) if center_lat >= 0 else (32700 + utm_zone)
    return CRS.from_epsg(epsg_code)


def _identify_vv_vh_bands(dataset: rasterio.DatasetReader) -> tuple[int, Optional[int], tuple[str, ...]]:
    """Identify 1-based band indices for VV and VH polarizations from dataset descriptions/tags."""
    descriptions = []
    vv_idx: Optional[int] = None
    vh_idx: Optional[int] = None

    for b in range(1, dataset.count + 1):
        desc = (dataset.descriptions[b - 1] or "").upper()
        b_tags = dataset.tags(b)
        pol_tag = (
            b_tags.get("POLARIZATION", "")
            or b_tags.get("polarization", "")
            or b_tags.get("BandName", "")
        ).upper()
        combined = f"{desc} {pol_tag}".strip()
        descriptions.append(combined if combined else f"Band {b}")

        if "VV" in combined and vv_idx is None:
            vv_idx = b
        elif "VH" in combined and vh_idx is None:
            vh_idx = b

    if vv_idx is None:
        vv_idx = 1
    if vh_idx is None and dataset.count >= 2:
        vh_idx = 2 if vv_idx == 1 else 1

    return vv_idx, vh_idx, tuple(descriptions)


def load_sar_geotiff(
    source: Union[str, Path, bytes, BytesIO, Any],
    name: str = "Uploaded GeoTIFF",
) -> SARDataset:
    """Load and validate a GeoTIFF from path, bytes, or Streamlit UploadedFile."""
    try:
        if isinstance(source, (str, Path)):
            ctx = rasterio.open(source)
        elif isinstance(source, bytes):
            ctx = MemoryFile(source).open()
        elif isinstance(source, BytesIO):
            source.seek(0)
            ctx = MemoryFile(source.read()).open()
        elif hasattr(source, "getvalue"):
            ctx = MemoryFile(source.getvalue()).open()
        elif hasattr(source, "read"):
            if hasattr(source, "seek"):
                source.seek(0)
            ctx = MemoryFile(source.read()).open()
        else:
            raise RasterValidationError(f"{name}: Unsupported file input type {type(source).__name__}.")
    except RasterValidationError:
        raise
    except Exception as exc:
        raise RasterValidationError(f"{name}: Could not open file as a valid GeoTIFF raster ({exc}).") from exc

    with ctx as ds:
        val_res = validate_raster_dataset(ds, name=name, require_crs=True)
        if not val_res.is_valid:
            raise RasterValidationError(" | ".join(val_res.errors))

        vv_idx, vh_idx, band_descs = _identify_vv_vh_bands(ds)
        vv_arr = ds.read(vv_idx, masked=False).astype(np.float64)
        vh_arr = ds.read(vh_idx, masked=False).astype(np.float64) if vh_idx is not None else None

        nodata = ds.nodata
        if nodata is not None and not np.isnan(nodata):
            vv_arr = np.where(np.isclose(vv_arr, nodata, rtol=1e-5, atol=1e-8), np.nan, vv_arr)
            if vh_arr is not None:
                vh_arr = np.where(np.isclose(vh_arr, nodata, rtol=1e-5, atol=1e-8), np.nan, vh_arr)

        crs = ds.crs
        transform = ds.transform
        bounds = ds.bounds
        res_xy = estimate_resolution_meters(crs, transform, bounds)
        wgs84_bounds = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
        tags = dict(ds.tags())
        is_demo = str(tags.get("IS_SYNTHETIC_DEMO", "false")).lower() == "true"

        return SARDataset(
            name=name,
            vv_band=vv_arr,
            vh_band=vh_arr,
            crs=crs,
            transform=transform,
            bounds=bounds,
            nodata=nodata,
            width=ds.width,
            height=ds.height,
            count=ds.count,
            band_descriptions=band_descs,
            resolution_xy_m=res_xy,
            wgs84_bounds=wgs84_bounds,
            is_demo=is_demo,
            tags=tags,
            warnings=val_res.warnings,
        )


def array_to_geotiff_bytes(
    array: np.ndarray,
    crs: CRS,
    transform: Affine,
    nodata: float = -9999.0,
    dtype: str = "float32",
    band_descriptions: Optional[ list[str] ] = None,
    tags: Optional[dict[str, str]] = None,
) -> bytes:
    """Serialize a 2D or 3D numpy array with spatial reference into an in-memory GeoTIFF byte stream."""
    if array.ndim == 2:
        data = array[np.newaxis, :, :]
    elif array.ndim == 3:
        data = array
    else:
        raise ValueError(f"Expected 2D or 3D array for GeoTIFF export, got shape {array.shape}.")

    count, height, width = data.shape
    out_arr = np.where(np.isfinite(data), data, nodata).astype(dtype)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(out_arr)
            if band_descriptions:
                for idx, desc in enumerate(band_descriptions[:count], start=1):
                    dst.set_band_description(idx, desc)
            if tags:
                dst.update_tags(**tags)
        return memfile.read()


def generate_synthetic_sar_pair(
    height: int = 300,
    width: int = 300,
    pixel_size_m: float = 20.0,
    seed: int = 42,
    return_linear: bool = True,
) -> tuple[SARDataset, SARDataset]:
    """
    Generate a realistic, georeferenced synthetic Sentinel-1 SAR bitemporal pair
    (Pre-Flood and Post-Flood) with VV and VH polarizations.

    Positioned over a representative floodplain coordinate frame in Assam, India
    (Brahmaputra basin, UTM Zone 46N / EPSG:32646), explicitly labeled as
    SYNTHETIC DEMO DATA so it is never confused with actual satellite imagery.
    """
    rng = np.random.default_rng(seed)

    # Coordinate frame in UTM Zone 46N (meters), near Majuli / Brahmaputra floodplain
    west_m = 600000.0
    south_m = 2960000.0
    east_m = west_m + width * pixel_size_m
    north_m = south_m + height * pixel_size_m

    crs = CRS.from_epsg(32646)
    transform = from_bounds(west_m, south_m, east_m, north_m, width, height)
    bounds = rasterio.coords.BoundingBox(west_m, south_m, east_m, north_m)
    wgs84_bounds = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)

    y_idx, x_idx = np.meshgrid(
        np.linspace(0.0, 1.0, height),
        np.linspace(0.0, 1.0, width),
        indexing="ij",
    )

    # 1. Meandering permanent river channel running diagonally across the scene
    river_center_x = 0.45 + 0.14 * np.sin(2.4 * np.pi * y_idx) + 0.06 * np.cos(1.3 * np.pi * y_idx)
    dist_to_river = np.abs(x_idx - river_center_x)
    permanent_water_mask = dist_to_river < 0.048

    # 2. Floodplain expansion zones (inundated only in Post-Flood)
    floodplain_left = (dist_to_river >= 0.048) & (dist_to_river < 0.19) & (y_idx > 0.12) & (y_idx < 0.88)
    oxbow_basin = ((x_idx - 0.25) ** 2 / 0.022 + (y_idx - 0.62) ** 2 / 0.032) < 1.0
    lowland_paddy = ((x_idx - 0.72) ** 2 / 0.025 + (y_idx - 0.36) ** 2 / 0.040) < 1.0
    new_flood_mask = (floodplain_left | oxbow_basin | lowland_paddy) & (~permanent_water_mask)

    # 3. Urban/settlement ridge (high backscatter double-bounce)
    urban_patch = ((x_idx - 0.82) ** 2 / 0.006 + (y_idx - 0.78) ** 2 / 0.006) < 1.0

    # Base clean sigma0 in dB for Pre-Flood (VV and VH)
    pre_vv_clean_db = np.full((height, width), -9.5, dtype=np.float64)
    pre_vv_clean_db += 1.2 * np.sin(5.0 * np.pi * x_idx) * np.cos(4.0 * np.pi * y_idx)
    pre_vv_clean_db[urban_patch] = -3.5
    pre_vv_clean_db[permanent_water_mask] = -21.5

    pre_vh_clean_db = pre_vv_clean_db - 6.5
    pre_vh_clean_db[permanent_water_mask] = -26.5

    # Base clean sigma0 in dB for Post-Flood (VV and VH)
    post_vv_clean_db = pre_vv_clean_db.copy()
    post_vv_clean_db[new_flood_mask] = -19.8 - 1.5 * np.exp(-dist_to_river[new_flood_mask] * 5.0)
    # Slight soil moisture increase on non-flooded land (+0.5 dB)
    non_water_post = ~(permanent_water_mask | new_flood_mask | urban_patch)
    post_vv_clean_db[non_water_post] += 0.4

    post_vh_clean_db = pre_vh_clean_db.copy()
    post_vh_clean_db[new_flood_mask] = -25.2

    # Convert clean dB to linear intensity and apply realistic multi-look Gamma SAR speckle (ENL = 4.4 for S1 GRD IW)
    enl = 4.4

    def _apply_speckle(clean_db: np.ndarray) -> np.ndarray:
        linear_clean = np.power(10.0, clean_db / 10.0)
        speckle = rng.gamma(shape=enl, scale=1.0 / enl, size=clean_db.shape)
        linear_noisy = np.clip(linear_clean * speckle, 1e-7, 10.0)
        if return_linear:
            return linear_noisy
        return 10.0 * np.log10(linear_noisy)

    pre_vv = _apply_speckle(pre_vv_clean_db)
    pre_vh = _apply_speckle(pre_vh_clean_db)
    post_vv = _apply_speckle(post_vv_clean_db)
    post_vh = _apply_speckle(post_vh_clean_db)

    common_tags = {
        "IS_SYNTHETIC_DEMO": "true",
        "SENSOR": "Synthetic Sentinel-1 C-Band SAR (Simulated IW GRD)",
        "CALIBRATION": "Linear Sigma0 Intensity" if return_linear else "Calibrated Sigma0 (dB)",
        "REGION": "Simulated Brahmaputra Floodplain (UTM 46N)",
    }

    pre_ds = SARDataset(
        name="DEMO_S1A_IW_GRD_PreFlood_20260710.tif",
        vv_band=pre_vv,
        vh_band=pre_vh,
        crs=crs,
        transform=transform,
        bounds=bounds,
        nodata=CONFIG.DEFAULT_NODATA,
        width=width,
        height=height,
        count=2,
        band_descriptions=("VV", "VH"),
        resolution_xy_m=(pixel_size_m, pixel_size_m),
        wgs84_bounds=wgs84_bounds,
        is_demo=True,
        tags={**common_tags, "ACQUISITION_DATE": "2026-07-10 (Pre-Flood Baseline)"},
    )

    post_ds = SARDataset(
        name="DEMO_S1A_IW_GRD_PostFlood_20260722.tif",
        vv_band=post_vv,
        vh_band=post_vh,
        crs=crs,
        transform=transform,
        bounds=bounds,
        nodata=CONFIG.DEFAULT_NODATA,
        width=width,
        height=height,
        count=2,
        band_descriptions=("VV", "VH"),
        resolution_xy_m=(pixel_size_m, pixel_size_m),
        wgs84_bounds=wgs84_bounds,
        is_demo=True,
        tags={**common_tags, "ACQUISITION_DATE": "2026-07-22 (Peak Inundation)"},
    )

    return pre_ds, post_ds

