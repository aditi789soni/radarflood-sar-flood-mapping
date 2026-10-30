"""Streamlit UI theme styling and reusable scientific dashboard components."""

import streamlit as st

from core.area_calculation import FloodAreaMetrics
from utils.config import SCIENTIFIC_DISCLAIMER
from utils.raster_utils import SARDataset


def inject_dashboard_css() -> None:
    """Apply dark navy / slate / cyan scientific GIS aesthetic to the Streamlit app."""
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0b1120;
            color: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        .rf-header-box {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid #334155;
            border-left: 5px solid #38bdf8;
            border-radius: 10px;
            padding: 18px 24px;
            margin-bottom: 18px;
        }
        .rf-title {
            font-size: 1.65rem;
            font-weight: 800;
            color: #f8fafc;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .rf-subtitle {
            font-size: 0.95rem;
            color: #94a3b8;
            margin-top: 4px;
            margin-bottom: 0;
        }
        .rf-demo-banner {
            background-color: rgba(245, 158, 11, 0.14);
            border: 1.5px solid #f59e0b;
            color: #fde68a;
            padding: 10px 16px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 16px;
        }
        .rf-kpi-card {
            background-color: #111c33;
            border: 1px solid #1e293b;
            border-top: 3px solid #38bdf8;
            border-radius: 8px;
            padding: 14px 16px;
            text-align: left;
            height: 100%;
        }
        .rf-kpi-card.alert {
            border-top-color: #ef4444;
        }
        .rf-kpi-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 4px;
        }
        .rf-kpi-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .rf-kpi-sub {
            font-size: 0.78rem;
            color: #38bdf8;
            margin-top: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(is_demo: bool = False) -> None:
    """Render top platform banner and prominent DEMO DATA notice when in synthetic mode."""
    st.markdown(
        """
        <div class="rf-header-box">
            <div class="rf-title">RadarFlood — All-Weather SAR Flood Mapping &amp; Inundation Analytics</div>
            <div class="rf-subtitle">
                Sentinel-1 C-Band Synthetic Aperture Radar (&sigma;&sup0; dB) Backscatter Calibration &bull;
                Adaptive Lee Speckle Filtering &bull; Bitemporal Change Detection &bull; Georeferenced Vector Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if is_demo:
        st.markdown(
            """
            <div class="rf-demo-banner">
                &#9888;&#65039; <strong>SYNTHETIC DEMO DATA ACTIVE:</strong> Currently analyzing simulated georeferenced
                C-band SAR arrays (Brahmaputra Floodplain coordinate frame, EPSG:32646) generated for pipeline verification.
                Synthetic outputs must never be interpreted as real satellite observations. Upload Sentinel-1 GeoTIFFs in the
                sidebar for real scene analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_kpi_row(metrics: FloodAreaMetrics) -> None:
    """Render 5 key performance indicators for flood analytics."""
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f"""
            <div class="rf-kpi-card alert">
                <div class="rf-kpi-label">Newly Inundated Area</div>
                <div class="rf-kpi-value">{metrics.new_flood_area_km2:.3f} km&sup2;</div>
                <div class="rf-kpi-sub">{metrics.new_flood_area_ha:.1f} ha ({metrics.new_flood_percentage:.2f}% of scene)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="rf-kpi-card">
                <div class="rf-kpi-label">Permanent Water Baseline</div>
                <div class="rf-kpi-value">{metrics.permanent_water_area_km2:.3f} km&sup2;</div>
                <div class="rf-kpi-sub">{metrics.permanent_water_percentage:.2f}% of valid scene</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="rf-kpi-card">
                <div class="rf-kpi-label">Total Post-Event Water</div>
                <div class="rf-kpi-value">{metrics.total_post_water_area_km2:.3f} km&sup2;</div>
                <div class="rf-kpi-sub">{metrics.total_post_water_percentage:.2f}% total inundation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="rf-kpi-card alert">
                <div class="rf-kpi-label">Detected Flood Polygons</div>
                <div class="rf-kpi-value">{metrics.vector_polygon_count}</div>
                <div class="rf-kpi-sub">Max patch: {metrics.largest_polygon_km2:.3f} km&sup2;</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="rf-kpi-card">
                <div class="rf-kpi-label">Valid Analyzed Footprint</div>
                <div class="rf-kpi-value">{metrics.valid_scene_area_km2:.2f} km&sup2;</div>
                <div class="rf-kpi-sub">Projected CRS: {metrics.metric_crs_epsg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_dataset_metadata_table(pre_ds: SARDataset, post_ds: SARDataset) -> None:
    """Display side-by-side raster metadata comparison table."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Pre-Flood Reference Raster")
        st.json(
            {
                "File Name": pre_ds.name,
                "Dimensions (H x W)": f"{pre_ds.height} x {pre_ds.width} px",
                "Bands Available": ", ".join(pre_ds.band_descriptions),
                "Dual-Pol (VV+VH)": pre_ds.has_vh,
                "Coordinate Reference System": pre_ds.crs.to_string() if pre_ds.crs else "None",
                "Pixel Resolution (m)": f"{pre_ds.resolution_xy_m[0]:.2f} m x {pre_ds.resolution_xy_m[1]:.2f} m",
                "WGS84 Bounds (W, S, E, N)": [round(x, 5) for x in pre_ds.wgs84_bounds],
                "Synthetic Demo Flag": pre_ds.is_demo,
            }
        )
    with col2:
        st.markdown("##### Post-Flood Target Raster")
        st.json(
            {
                "File Name": post_ds.name,
                "Dimensions (H x W)": f"{post_ds.height} x {post_ds.width} px",
                "Bands Available": ", ".join(post_ds.band_descriptions),
                "Dual-Pol (VV+VH)": post_ds.has_vh,
                "Coordinate Reference System": post_ds.crs.to_string() if post_ds.crs else "None",
                "Pixel Resolution (m)": f"{post_ds.resolution_xy_m[0]:.2f} m x {post_ds.resolution_xy_m[1]:.2f} m",
                "WGS84 Bounds (W, S, E, N)": [round(x, 5) for x in post_ds.wgs84_bounds],
                "Synthetic Demo Flag": post_ds.is_demo,
            }
        )


def render_scientific_methodology_notes() -> None:
    """Render scientific methodology and operational SAR limitations."""
    st.info(SCIENTIFIC_DISCLAIMER)

