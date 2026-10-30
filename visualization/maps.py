"""Interactive Folium geospatial map generation with SAR raster and GeoJSON vector overlays."""

import base64
from io import BytesIO
from typing import Optional
import cv2
import folium
import geopandas as gpd
import numpy as np
from matplotlib import cm


def _downsample_for_overlay(arr: np.ndarray, max_dim: int = 600) -> np.ndarray:
    """Downsample a 2D array if its dimensions exceed max_dim to keep Folium HTML responsive."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_dim:
        return arr
    scale = max_dim / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_NEAREST if arr.dtype in (np.bool_, np.int8, np.uint8) else cv2.INTER_AREA
    return cv2.resize(arr.astype(np.float32), (new_w, new_h), interpolation=interp)


def _array_to_png_data_url(rgba: np.ndarray) -> str:
    """Encode an RGBA uint8 numpy array (H, W, 4) into a base64 PNG data URI for Folium ImageOverlay."""
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    success, encoded = cv2.imencode(".png", bgra)
    if not success:
        raise RuntimeError("Failed to encode RGBA overlay array to PNG.")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def sar_db_to_rgba(
    sigma0_db: np.ndarray,
    vmin: float = -26.0,
    vmax: float = 0.0,
    alpha: int = 215,
) -> np.ndarray:
    """Convert a 2D dB backscatter array into an RGBA uint8 image for map overlay."""
    small = _downsample_for_overlay(sigma0_db)
    valid = np.isfinite(small)
    norm = np.clip((np.nan_to_num(small, nan=vmin) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    rgba = np.zeros((*small.shape, 4), dtype=np.uint8)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = np.where(valid, alpha, 0).astype(np.uint8)
    return rgba


def classification_to_rgba(class_map: np.ndarray) -> np.ndarray:
    """
    Convert multi-class inundation map (-1=nodata, 0=land, 1=permanent water, 2=new flood)
    into a semi-transparent RGBA uint8 overlay.
    """
    small = _downsample_for_overlay(class_map).astype(np.int8)
    rgba = np.zeros((*small.shape, 4), dtype=np.uint8)

    # Class 1: Permanent Water -> Deep Ocean Blue (#0284c7)
    perm_mask = small == 1
    rgba[perm_mask] = [2, 132, 199, 195]

    # Class 2: Newly Inundated Floodwater -> High-Visibility Cyan-Red Alert (#ef4444 / #00f5d4)
    flood_mask = small == 2
    rgba[flood_mask] = [239, 68, 68, 220]

    return rgba


def build_inundation_folium_map(
    wgs84_bounds: tuple[float, float, float, float],
    pre_vv_db: np.ndarray,
    post_vv_db: np.ndarray,
    classification_map: np.ndarray,
    polygons_gdf: Optional[gpd.GeoDataFrame] = None,
    is_demo: bool = False,
) -> folium.Map:
    """
    Construct an interactive Folium map displaying Pre/Post SAR backscatter rasters,
    classified inundation raster, and vectorized flood polygons with layer controls and legend.
    """
    west, south, east, north = wgs84_bounds
    center_lat = 0.5 * (south + north)
    center_lon = 0.5 * (west + east)
    folium_bounds = [[south, west], [north, east]]

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Dark GIS Basemap",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite Imagery",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True,
    ).add_to(m)

    # 1. Pre-Flood SAR VV (dB) overlay
    pre_rgba = sar_db_to_rgba(pre_vv_db, vmin=-25.0, vmax=-2.0, alpha=210)
    folium.raster_layers.ImageOverlay(
        image=_array_to_png_data_url(pre_rgba),
        bounds=folium_bounds,
        name="Pre-Flood SAR VV (dB)",
        opacity=0.85,
        show=False,
        interactive=False,
        zindex=2,
    ).add_to(m)

    # 2. Post-Flood SAR VV (dB) overlay
    post_rgba = sar_db_to_rgba(post_vv_db, vmin=-25.0, vmax=-2.0, alpha=210)
    folium.raster_layers.ImageOverlay(
        image=_array_to_png_data_url(post_rgba),
        bounds=folium_bounds,
        name="Post-Flood SAR VV (dB)",
        opacity=0.85,
        show=True,
        interactive=False,
        zindex=3,
    ).add_to(m)

    # 3. Raster Classification Overlay (Permanent Water + New Floodwater)
    cls_rgba = classification_to_rgba(classification_map)
    folium.raster_layers.ImageOverlay(
        image=_array_to_png_data_url(cls_rgba),
        bounds=folium_bounds,
        name="Inundation Raster Mask",
        opacity=0.85,
        show=True,
        interactive=False,
        zindex=4,
    ).add_to(m)

    # 4. Vector Flood Polygons Overlay (GeoJSON)
    if polygons_gdf is not None and not polygons_gdf.empty:
        gdf_wgs84 = polygons_gdf.to_crs("EPSG:4326") if polygons_gdf.crs is not None else polygons_gdf

        def _style_fn(feature):
            cls_id = feature["properties"].get("class_id", 2)
            if cls_id == 1:
                return {
                    "fillColor": "#0284c7",
                    "color": "#38bdf8",
                    "weight": 1.5,
                    "fillOpacity": 0.45,
                }
            return {
                "fillColor": "#ef4444",
                "color": "#fde047",
                "weight": 2.0,
                "fillOpacity": 0.45,
            }

        folium.GeoJson(
            data=gdf_wgs84.to_json(),
            name="Vectorized Flood Polygons (GeoJSON)",
            style_function=_style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["polygon_id", "class_name", "area_km2", "area_ha", "perimeter_m"],
                aliases=["Polygon ID:", "Class:", "Area (km²):", "Area (ha):", "Perimeter (m):"],
                localize=True,
                sticky=False,
            ),
            show=True,
        ).add_to(m)

    m.fit_bounds(folium_bounds)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    # Floating scientific map legend
    demo_badge = (
        '<div style="background:#f59e0b;color:#0f172a;font-weight:700;padding:3px 6px;'
        'border-radius:4px;margin-bottom:6px;font-size:11px;text-align:center;">'
        'SYNTHETIC DEMO DATA</div>'
        if is_demo
        else ""
    )
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 28px;
        left: 28px;
        z-index: 9999;
        background-color: rgba(15, 23, 42, 0.92);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 14px;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        min-width: 210px;
    ">
        {demo_badge}
        <div style="font-weight: 700; font-size: 13px; margin-bottom: 8px; color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 4px;">
            SAR Flood Classification Legend
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="display:inline-block;width:16px;height:16px;background:#ef4444;border:1.5px solid #fde047;margin-right:8px;border-radius:3px;"></span>
            <span>Newly Inundated Floodwater</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="display:inline-block;width:16px;height:16px;background:#0284c7;border:1.5px solid #38bdf8;margin-right:8px;border-radius:3px;"></span>
            <span>Permanent Water Body</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="display:inline-block;width:16px;height:16px;background:linear-gradient(90deg,#111,#ddd);border:1px solid #64748b;margin-right:8px;border-radius:3px;"></span>
            <span>SAR Backscatter (&sigma;&sup0; dB)</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

