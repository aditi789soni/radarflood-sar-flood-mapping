"""
RadarFlood – All-Weather SAR Flood Mapping & Inundation Analytics Platform
Main Streamlit Application Entry Point (app.py)
"""

import json
from datetime import datetime, timezone
from typing import Any, Optional
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

from core.area_calculation import (
    FloodAreaMetrics,
    calculate_inundation_metrics,
    metrics_to_csv_bytes,
)
from core.calibration import CalibrationResult, InputScaleOption, to_sigma0_db
from core.change_detection import ChangeDetectionResult, run_bitemporal_change_detection
from core.flood_detection import compute_otsu_threshold_db
from core.preprocessing import align_bitemporal_datasets
from core.speckle_filter import lee_speckle_filter
from core.vectorization import export_polygons_to_geojson, vectorize_flood_mask
from utils.config import CONFIG
from utils.raster_utils import (
    SARDataset,
    array_to_geotiff_bytes,
    generate_synthetic_sar_pair,
    load_sar_geotiff,
)
from utils.validation import RasterValidationError
from visualization.charts import (
    create_area_breakdown_chart,
    create_backscatter_histogram,
    create_classification_heatmap,
    create_delta_histogram,
    create_sar_heatmap,
    create_vv_vh_comparison_chart,
)
from visualization.dashboard import (
    inject_dashboard_css,
    render_dataset_metadata_table,
    render_header,
    render_kpi_row,
    render_scientific_methodology_notes,
)
from visualization.maps import build_inundation_folium_map

st.set_page_config(
    page_title="RadarFlood | SAR Flood Mapping & Inundation Analytics",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def get_cached_demo_datasets(
    height: int = 300,
    width: int = 300,
    pixel_size_m: float = 20.0,
    return_linear: bool = True,
) -> tuple[SARDataset, SARDataset]:
    """Generate and cache the synthetic georeferenced Sentinel-1 SAR dataset pair."""
    return generate_synthetic_sar_pair(
        height=height,
        width=width,
        pixel_size_m=pixel_size_m,
        return_linear=return_linear,
    )


@st.cache_data(show_spinner=False, ttl=600)
def search_planetary_computer_stac(
    bbox: tuple[float, float, float, float],
    datetime_range: str = "2024-06-01/2024-09-30",
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Query the free Microsoft Planetary Computer STAC API for Sentinel-1 GRD/RTC scenes.
    Gracefully handles offline/network errors without interrupting local or demo workflows.
    """
    url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    payload = {
        "collections": ["sentinel-1-grd"],
        "bbox": list(bbox),
        "datetime": datetime_range,
        "limit": limit,
    }
    resp = requests.post(url, json=payload, timeout=8)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    results = []
    for feat in features:
        props = feat.get("properties", {})
        results.append(
            {
                "Scene ID": feat.get("id"),
                "Acquisition UTC": props.get("datetime"),
                "Platform": props.get("platform", "sentinel-1"),
                "Instrument Mode": props.get("sar:instrument_mode", "IW"),
                "Polarizations": ", ".join(props.get("sar:polarizations", [])),
                "Orbit State": props.get("sat:orbit_state", "N/A"),
            }
        )
    return results


def execute_sar_pipeline(
    pre_ds_raw: SARDataset,
    post_ds_raw: SARDataset,
    input_scale: InputScaleOption,
    allow_uncalibrated_dn: bool,
    apply_lee: bool,
    lee_window: int,
    lee_enl: float,
    vv_threshold_db: float,
    use_otsu: bool,
    include_vh: bool,
    vh_threshold_db: float,
    vh_combine_mode: str,
    pre_perm_water_db: float,
    delta_drop_db: float,
    apply_opening: bool,
    apply_closing: bool,
    morph_kernel: int,
    min_flood_pixels: int,
    min_polygon_area_m2: float,
) -> dict[str, Any]:
    """Run the complete end-to-end SAR flood detection and bitemporal analytics pipeline."""
    # Step 1: Spatial alignment & co-registration
    pre_ds, post_ds, alignment_notes = align_bitemporal_datasets(pre_ds_raw, post_ds_raw)

    # Step 2: Radiometric scale detection & sigma0 dB conversion
    pre_vv_cal: CalibrationResult = to_sigma0_db(
        pre_ds.vv_band,
        input_scale=input_scale,
        nodata=None,
        metadata_tags=pre_ds.tags,
        allow_uncalibrated_dn_log=allow_uncalibrated_dn,
    )
    post_vv_cal: CalibrationResult = to_sigma0_db(
        post_ds.vv_band,
        input_scale=input_scale,
        nodata=None,
        metadata_tags=post_ds.tags,
        allow_uncalibrated_dn_log=allow_uncalibrated_dn,
    )

    pre_vh_db: Optional[np.ndarray] = None
    post_vh_db: Optional[np.ndarray] = None
    if pre_ds.vh_band is not None:
        pre_vh_db = to_sigma0_db(
            pre_ds.vh_band,
            input_scale=input_scale,
            nodata=None,
            metadata_tags=pre_ds.tags,
            allow_uncalibrated_dn_log=allow_uncalibrated_dn,
        ).sigma0_db
    if post_ds.vh_band is not None:
        post_vh_db = to_sigma0_db(
            post_ds.vh_band,
            input_scale=input_scale,
            nodata=None,
            metadata_tags=post_ds.tags,
            allow_uncalibrated_dn_log=allow_uncalibrated_dn,
        ).sigma0_db

    # Step 3: Adaptive Lee Speckle Filtering
    if apply_lee:
        pre_vv_proc = lee_speckle_filter(pre_vv_cal.sigma0_db, window_size=lee_window, enl=lee_enl, is_db=True)
        post_vv_proc = lee_speckle_filter(post_vv_cal.sigma0_db, window_size=lee_window, enl=lee_enl, is_db=True)
        pre_vh_proc = (
            lee_speckle_filter(pre_vh_db, window_size=lee_window, enl=lee_enl, is_db=True)
            if pre_vh_db is not None
            else None
        )
        post_vh_proc = (
            lee_speckle_filter(post_vh_db, window_size=lee_window, enl=lee_enl, is_db=True)
            if post_vh_db is not None
            else None
        )
    else:
        pre_vv_proc = pre_vv_cal.sigma0_db
        post_vv_proc = post_vv_cal.sigma0_db
        pre_vh_proc = pre_vh_db
        post_vh_proc = post_vh_db

    # Compute reference Otsu threshold on Post-Flood VV for comparison
    otsu_post_vv_db = compute_otsu_threshold_db(post_vv_proc)

    # Step 4: Bitemporal Change Detection & Flood Classification
    change_res: ChangeDetectionResult = run_bitemporal_change_detection(
        pre_vv_db=pre_vv_proc,
        post_vv_db=post_vv_proc,
        pre_vh_db=pre_vh_proc,
        post_vh_db=post_vh_proc,
        post_vv_threshold_db=vv_threshold_db,
        pre_perm_water_threshold_db=pre_perm_water_db,
        delta_db_threshold=delta_drop_db,
        use_otsu_post=use_otsu,
        include_vh=include_vh,
        post_vh_threshold_db=vh_threshold_db,
        vh_combine_mode="AND" if vh_combine_mode == "AND" else "OR",
        apply_opening=apply_opening,
        apply_closing=apply_closing,
        morph_kernel_size=morph_kernel,
        min_region_pixels=min_flood_pixels,
    )

    # Step 5: Georeferenced Vectorization
    polygons_gdf = vectorize_flood_mask(
        mask_or_class_map=change_res.classification_map,
        crs=pre_ds.crs,
        transform=pre_ds.transform,
        bounds=pre_ds.bounds,
        min_polygon_area_m2=min_polygon_area_m2,
        include_permanent_water=True,
    )

    # Step 6: Metric Area Calculation
    area_metrics: FloodAreaMetrics = calculate_inundation_metrics(
        valid_mask=change_res.valid_mask,
        new_flood_mask=change_res.new_flood_mask,
        crs=pre_ds.crs,
        transform=pre_ds.transform,
        permanent_water_mask=change_res.permanent_water_mask,
        polygons_gdf=polygons_gdf,
    )

    return {
        "pre_ds": pre_ds,
        "post_ds": post_ds,
        "alignment_notes": alignment_notes,
        "pre_vv_cal": pre_vv_cal,
        "post_vv_cal": post_vv_cal,
        "pre_vv_unfiltered_db": pre_vv_cal.sigma0_db,
        "post_vv_unfiltered_db": post_vv_cal.sigma0_db,
        "pre_vv_db": pre_vv_proc,
        "post_vv_db": post_vv_proc,
        "pre_vh_db": pre_vh_proc,
        "post_vh_db": post_vh_proc,
        "otsu_post_vv_db": otsu_post_vv_db,
        "change_res": change_res,
        "polygons_gdf": polygons_gdf,
        "area_metrics": area_metrics,
    }


def main() -> None:
    inject_dashboard_css()

    if "demo_active" not in st.session_state:
        st.session_state["demo_active"] = True

    # ==================== SIDEBAR CONTROLS ====================
    with st.sidebar:
        st.markdown("### 🛰️ Data Acquisition & Mode")
        data_mode = st.radio(
            "Select Input Source",
            options=[
                "Synthetic SAR Demonstration Mode",
                "Upload Local Pre/Post GeoTIFFs",
            ],
            index=0 if st.session_state["demo_active"] else 1,
        )

        is_demo_mode = data_mode == "Synthetic SAR Demonstration Mode"
        st.session_state["demo_active"] = is_demo_mode

        uploaded_pre = None
        uploaded_post = None
        demo_scale_linear = True

        if is_demo_mode:
            st.caption(
                "Generates a georeferenced C-band SAR bitemporal dataset (UTM Zone 46N, 20m resolution) "
                "with realistic multi-look Gamma speckle, permanent river channel, and flood inundation zones."
            )
            demo_scale_choice = st.selectbox(
                "Simulated Sensor Output Scale",
                options=["Linear Sigma0 Intensity (σ⁰ linear)", "Already Calibrated Decibels (σ⁰ dB)"],
                index=0,
            )
            demo_scale_linear = demo_scale_choice.startswith("Linear")
            if st.button("⚡ One-Click Run Full Demo Pipeline", type="primary", use_container_width=True):
                st.toast("Synthetic SAR Demonstration Pipeline executed!", icon="🛰️")
        else:
            st.caption("Upload co-located Sentinel-1 GRD/RTC GeoTIFF rasters (1-band VV or 2-band VV+VH).")
            uploaded_pre = st.file_uploader("1. Pre-Flood Baseline GeoTIFF (.tif)", type=["tif", "tiff"], key="pre_up")
            uploaded_post = st.file_uploader("2. Post-Flood Event GeoTIFF (.tif)", type=["tif", "tiff"], key="post_up")

        st.markdown("---")
        st.markdown("### ⚙️ Radiometric Calibration")
        scale_map: dict[str, InputScaleOption] = {
            "Auto-Detect Scale (Recommended)": "auto",
            "Linear Intensity (Apply 10·log₁₀)": "linear_sigma0",
            "Already in Decibels (dB — Do Not Convert)": "dB",
        }
        scale_label = st.selectbox("Input Backscatter Scale", options=list(scale_map.keys()), index=0)
        input_scale: InputScaleOption = scale_map[scale_label]
        allow_uncalibrated_dn = st.checkbox(
            "Allow relative dB scaling if uncalibrated DN is detected",
            value=False,
            help="Raw Sentinel-1 DNs require LUT calibration. Enable only for exploratory viewing of uncalibrated rasters.",
        )

        st.markdown("---")
        st.markdown("### 🎛️ Lee Speckle Filtering")
        apply_lee = st.checkbox("Enable Adaptive Lee Filter", value=True)
        lee_window = st.select_slider(
            "Lee Filter Window Size",
            options=list(CONFIG.ALLOWED_LEE_WINDOWS),
            value=CONFIG.DEFAULT_LEE_WINDOW,
            format_func=lambda w: f"{w}×{w} px",
            disabled=not apply_lee,
        )
        lee_enl = st.slider(
            "Equivalent Number of Looks (ENL)",
            min_value=1.0,
            max_value=12.0,
            value=4.4,
            step=0.2,
            disabled=not apply_lee,
            help="Sentinel-1 IW GRD has a nominal ENL of ~4.4.",
        )

        st.markdown("---")
        st.markdown("### 🌊 Flood Classification & Change")
        use_otsu = st.checkbox("Use Otsu Automatic VV Threshold", value=False)
        vv_threshold_db = st.slider(
            "Post-Flood VV Water Threshold (dB)",
            min_value=float(CONFIG.MIN_VV_THRESHOLD_DB),
            max_value=float(CONFIG.MAX_VV_THRESHOLD_DB),
            value=float(CONFIG.DEFAULT_VV_THRESHOLD_DB),
            step=0.25,
            disabled=use_otsu,
            help="Pixels with σ⁰ VV ≤ threshold are candidate smooth water surfaces (default: -16.0 dB).",
        )
        delta_drop_db = st.slider(
            "Bitemporal Δσ⁰ Drop Threshold (Post − Pre, dB)",
            min_value=float(CONFIG.MIN_CHANGE_DROP_DB),
            max_value=float(CONFIG.MAX_CHANGE_DROP_DB),
            value=float(CONFIG.DEFAULT_CHANGE_DROP_DB),
            step=0.25,
            help="Minimum backscatter decrease required to classify a pixel as newly inundated floodwater.",
        )
        pre_perm_water_db = st.slider(
            "Pre-Flood Permanent Water Threshold (dB)",
            min_value=-26.0,
            max_value=-10.0,
            value=float(CONFIG.DEFAULT_PERMANENT_WATER_DB),
            step=0.5,
        )

        include_vh = st.checkbox("Include VH Cross-Polarization Criterion", value=False)
        vh_threshold_db = st.slider(
            "Post-Flood VH Water Threshold (dB)",
            min_value=float(CONFIG.MIN_VH_THRESHOLD_DB),
            max_value=float(CONFIG.MAX_VH_THRESHOLD_DB),
            value=float(CONFIG.DEFAULT_VH_THRESHOLD_DB),
            step=0.5,
            disabled=not include_vh,
        )
        vh_combine_mode = st.radio(
            "VV / VH Combination Rule",
            options=["AND", "OR"],
            index=0,
            horizontal=True,
            disabled=not include_vh,
        )

        st.markdown("---")
        st.markdown("### 🧩 Morphological & Vector Filters")
        apply_opening = st.checkbox("Morphological Opening (Remove Speckle False Positives)", value=True)
        apply_closing = st.checkbox("Morphological Closing (Fill Internal Pinholes)", value=True)
        morph_kernel = st.select_slider(
            "Structuring Element Size",
            options=[3, 5, 7],
            value=CONFIG.DEFAULT_MORPH_KERNEL_SIZE,
            format_func=lambda k: f"{k}×{k} px",
        )
        min_flood_pixels = st.slider(
            "Min Connected Raster Patch (pixels)",
            min_value=1,
            max_value=100,
            value=CONFIG.DEFAULT_MIN_FLOOD_PIXELS,
        )
        min_polygon_area_m2 = st.number_input(
            "Min Vector Polygon Area (m²)",
            min_value=0.0,
            max_value=100000.0,
            value=float(CONFIG.DEFAULT_MIN_POLYGON_AREA_M2),
            step=500.0,
        )

    # ==================== LOAD DATASETS ====================
    pre_ds_raw: Optional[SARDataset] = None
    post_ds_raw: Optional[SARDataset] = None

    if is_demo_mode:
        pre_ds_raw, post_ds_raw = get_cached_demo_datasets(return_linear=demo_scale_linear)
    else:
        if uploaded_pre is None and uploaded_post is None:
            render_header(is_demo=False)
            st.warning(
                "📂 **Awaiting GeoTIFF Uploads:** Please upload both a **Pre-Flood Baseline GeoTIFF** and a "
                "**Post-Flood Event GeoTIFF** in the sidebar, or switch to **Synthetic SAR Demonstration Mode** "
                "to explore the full processing pipeline immediately."
            )
            with st.expander("💡 Need sample GeoTIFF files to test the local upload workflow?", expanded=True):
                demo_pre, demo_post = get_cached_demo_datasets(return_linear=True)
                pre_bytes = array_to_geotiff_bytes(
                    np.stack([demo_pre.vv_band, demo_pre.vh_band]),
                    crs=demo_pre.crs,
                    transform=demo_pre.transform,
                    band_descriptions=["VV", "VH"],
                    tags=demo_pre.tags,
                )
                post_bytes = array_to_geotiff_bytes(
                    np.stack([demo_post.vv_band, demo_post.vh_band]),
                    crs=demo_post.crs,
                    transform=demo_post.transform,
                    band_descriptions=["VV", "VH"],
                    tags=demo_post.tags,
                )
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        "⬇️ Download Sample Pre-Flood GeoTIFF",
                        data=pre_bytes,
                        file_name="Sample_S1_PreFlood_UTM46N.tif",
                        mime="image/tiff",
                        use_container_width=True,
                    )
                with dl_col2:
                    st.download_button(
                        "⬇️ Download Sample Post-Flood GeoTIFF",
                        data=post_bytes,
                        file_name="Sample_S1_PostFlood_UTM46N.tif",
                        mime="image/tiff",
                        use_container_width=True,
                    )
            render_scientific_methodology_notes()
            return

        if uploaded_pre is None or uploaded_post is None:
            render_header(is_demo=False)
            missing_which = "Pre-Flood Baseline GeoTIFF" if uploaded_pre is None else "Post-Flood Event GeoTIFF"
            st.info(
                f"⏳ **Single Raster Uploaded:** Please also upload the **{missing_which}** in the sidebar "
                "to enable bitemporal change detection and flood mapping."
            )
            return

        try:
            with st.spinner("Validating and loading uploaded SAR GeoTIFF rasters..."):
                pre_ds_raw = load_sar_geotiff(uploaded_pre, name=uploaded_pre.name)
                post_ds_raw = load_sar_geotiff(uploaded_post, name=uploaded_post.name)
        except RasterValidationError as val_err:
            render_header(is_demo=False)
            st.error(f"🚨 **SAR Raster Validation Error:** {val_err}")
            return
        except Exception as exc:
            render_header(is_demo=False)
            st.error(f"🚨 **Unexpected Error Loading GeoTIFFs:** {exc}")
            return

    # ==================== EXECUTE PIPELINE ====================
    is_any_demo = bool(pre_ds_raw.is_demo or post_ds_raw.is_demo)
    render_header(is_demo=is_any_demo)

    try:
        with st.spinner("Running SAR calibration, Lee speckle filtering, bitemporal change detection, and vectorization..."):
            results = execute_sar_pipeline(
                pre_ds_raw=pre_ds_raw,
                post_ds_raw=post_ds_raw,
                input_scale=input_scale,
                allow_uncalibrated_dn=allow_uncalibrated_dn,
                apply_lee=apply_lee,
                lee_window=lee_window,
                lee_enl=lee_enl,
                vv_threshold_db=vv_threshold_db,
                use_otsu=use_otsu,
                include_vh=include_vh,
                vh_threshold_db=vh_threshold_db,
                vh_combine_mode=vh_combine_mode,
                pre_perm_water_db=pre_perm_water_db,
                delta_drop_db=delta_drop_db,
                apply_opening=apply_opening,
                apply_closing=apply_closing,
                morph_kernel=morph_kernel,
                min_flood_pixels=min_flood_pixels,
                min_polygon_area_m2=min_polygon_area_m2,
            )
    except RasterValidationError as proc_err:
        st.error(f"🚨 **SAR Processing & Validation Error:** {proc_err}")
        return
    except Exception as exc:
        st.error(f"🚨 **Pipeline Execution Failure:** {exc}")
        return

    pre_ds: SARDataset = results["pre_ds"]
    post_ds: SARDataset = results["post_ds"]
    pre_vv_cal: CalibrationResult = results["pre_vv_cal"]
    post_vv_cal: CalibrationResult = results["post_vv_cal"]
    pre_vv_db: np.ndarray = results["pre_vv_db"]
    post_vv_db: np.ndarray = results["post_vv_db"]
    pre_vh_db: Optional[np.ndarray] = results["pre_vh_db"]
    post_vh_db: Optional[np.ndarray] = results["post_vh_db"]
    change_res: ChangeDetectionResult = results["change_res"]
    polygons_gdf = results["polygons_gdf"]
    area_metrics: FloodAreaMetrics = results["area_metrics"]

    # Top-level KPI Summary Row
    render_kpi_row(area_metrics)
    st.write("")

    # ==================== 6 DASHBOARD TABS ====================
    tabs = st.tabs(
        [
            "1. Overview",
            "2. SAR Image Explorer",
            "3. Processing Laboratory",
            "4. Flood Detection Map",
            "5. Analytics",
            "6. Export Center",
        ]
    )

    # -------------------- TAB 1: OVERVIEW --------------------
    with tabs[0]:
        st.markdown("#### Platform Overview & Processing Telemetry")
        st.markdown(
            "RadarFlood processes dual-polarization (**VV / VH**) C-band Synthetic Aperture Radar imagery "
            "to map surface inundation extent under all-weather, day-and-night conditions. By combining "
            "radiometric $\\sigma^0$ decibel calibration, adaptive **Lee MMSE speckle filtering**, and "
            "**bitemporal backscatter change detection** ($\\Delta\\sigma^0 = \\sigma^0_{\\text{post}} - \\sigma^0_{\\text{pre}}$), "
            "the platform separates pre-existing permanent water bodies from newly inundated floodplains."
        )

        # Processing status log
        st.markdown("##### Pipeline Execution Status")
        status_items = (
            [f"✅ **Spatial Co-Registration:** {n}" for n in results["alignment_notes"]]
            + [f"✅ **Pre-Flood Calibration ({pre_vv_cal.detected_scale}):** {n}" for n in pre_vv_cal.notes]
            + [f"✅ **Post-Flood Calibration ({post_vv_cal.detected_scale}):** {n}" for n in post_vv_cal.notes]
            + [
                f"✅ **Speckle Filtering:** {'Adaptive Lee Filter (' + str(lee_window) + '×' + str(lee_window) + f', ENL={lee_enl})' if apply_lee else 'Disabled (Raw dB)'}"
            ]
            + [f"✅ **Flood Classification:** {n}" for n in change_res.notes]
        )
        for item in status_items:
            st.markdown(item)

        st.markdown("---")
        render_dataset_metadata_table(pre_ds, post_ds)

        st.markdown("---")
        render_scientific_methodology_notes()

        with st.expander("🌐 Query Microsoft Planetary Computer STAC Catalog (Live Sentinel-1 Metadata Search)"):
            st.caption(
                "Search open-access Sentinel-1 GRD metadata over the current scene bounding box via the "
                "Microsoft Planetary Computer STAC API."
            )
            stac_col1, stac_col2 = st.columns([3, 1])
            with stac_col1:
                stac_dates = st.text_input("ISO Date Range", value="2024-05-01/2024-09-30")
            with stac_col2:
                st.write("")
                run_stac = st.button("Search STAC API", use_container_width=True)
            if run_stac:
                try:
                    stac_items = search_planetary_computer_stac(pre_ds.wgs84_bounds, datetime_range=stac_dates)
                    if stac_items:
                        st.dataframe(pd.DataFrame(stac_items), use_container_width=True)
                    else:
                        st.info("No Sentinel-1 GRD scenes matched the specified date range and bounding box.")
                except Exception as stac_err:
                    st.warning(f"Planetary Computer STAC API unreachable in current network environment: {stac_err}")

    # -------------------- TAB 2: SAR IMAGE EXPLORER --------------------
    with tabs[1]:
        st.markdown("#### Multi-Polarization & Bitemporal SAR Image Explorer")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            pol_options = ["VV Polarization"]
            if pre_vh_db is not None and post_vh_db is not None:
                pol_options.append("VH Cross-Polarization")
                pol_options.append("VH − VV Cross-Pol Ratio (dB)")
            selected_pol = st.selectbox("Select Polarization Channel", options=pol_options)
        with exp_col2:
            contrast_range = st.slider(
                "Backscatter Display Dynamic Range (dB)",
                min_value=-35.0,
                max_value=5.0,
                value=(-25.0, -2.0),
                step=0.5,
            )

        if selected_pol == "VV Polarization":
            disp_pre = pre_vv_db
            disp_post = post_vv_db
            label_unit = "σ⁰ VV (dB)"
        elif selected_pol == "VH Cross-Polarization" and pre_vh_db is not None and post_vh_db is not None:
            disp_pre = pre_vh_db
            disp_post = post_vh_db
            label_unit = "σ⁰ VH (dB)"
        else:
            disp_pre = pre_vh_db - pre_vv_db if pre_vh_db is not None else pre_vv_db
            disp_post = post_vh_db - post_vv_db if post_vh_db is not None else post_vv_db
            label_unit = "VH − VV (dB)"

        img_c1, img_c2 = st.columns(2)
        with img_c1:
            st.plotly_chart(
                create_sar_heatmap(
                    disp_pre,
                    title=f"Pre-Flood Baseline — {label_unit}",
                    colorbar_title=label_unit,
                    zmin=contrast_range[0],
                    zmax=contrast_range[1],
                ),
                use_container_width=True,
            )
        with img_c2:
            st.plotly_chart(
                create_sar_heatmap(
                    disp_post,
                    title=f"Post-Flood Event — {label_unit}",
                    colorbar_title=label_unit,
                    zmin=contrast_range[0],
                    zmax=contrast_range[1],
                ),
                use_container_width=True,
            )

        st.markdown("##### Bitemporal Difference & Classification Overview")
        diff_c1, diff_c2 = st.columns(2)
        with diff_c1:
            st.plotly_chart(
                create_sar_heatmap(
                    change_res.delta_vv_db,
                    title="Bitemporal Backscatter Change (Δσ⁰ VV = Post − Pre dB)",
                    colorbar_title="Δσ⁰ (dB)",
                    colorscale="RdBu_r",
                    zmin=-12.0,
                    zmax=12.0,
                ),
                use_container_width=True,
            )
        with diff_c2:
            st.plotly_chart(
                create_classification_heatmap(
                    change_res.classification_map,
                    title="Classified Surface Water & New Flood Inundation",
                ),
                use_container_width=True,
            )

    # -------------------- TAB 3: PROCESSING LABORATORY --------------------
    with tabs[2]:
        st.markdown("#### SAR Speckle Filtering & Thresholding Laboratory")
        st.markdown(
            f"**Active Parameters:** Lee Window = `{lee_window}×{lee_window} px` (`Enabled={apply_lee}`) | "
            f"Effective Post-Flood VV Threshold = `{change_res.effective_post_vv_thresh_db:.2f} dB` "
            f"(Otsu Reference = `{results['otsu_post_vv_db']:.2f} dB`) | "
            f"Δσ⁰ Drop Threshold = `{change_res.delta_drop_thresh_db:.2f} dB` | "
            f"Morphological Kernel = `{morph_kernel}×{morph_kernel} px`"
        )

        lab_c1, lab_c2, lab_c3 = st.columns(3)
        with lab_c1:
            st.plotly_chart(
                create_sar_heatmap(
                    results["post_vv_unfiltered_db"],
                    title="1. Unfiltered Post-Flood σ⁰ VV (dB)",
                    zmin=-25.0,
                    zmax=-2.0,
                    height=350,
                ),
                use_container_width=True,
            )
        with lab_c2:
            st.plotly_chart(
                create_sar_heatmap(
                    post_vv_db,
                    title=f"2. Lee Speckle Filtered ({lee_window}×{lee_window}, ENL={lee_enl})",
                    zmin=-25.0,
                    zmax=-2.0,
                    height=350,
                ),
                use_container_width=True,
            )
        with lab_c3:
            speckle_suppression = post_vv_db - results["post_vv_unfiltered_db"]
            st.plotly_chart(
                create_sar_heatmap(
                    speckle_suppression,
                    title="3. Speckle Residual (Filtered − Unfiltered dB)",
                    colorbar_title="Δ (dB)",
                    colorscale="Picnic",
                    zmin=-4.0,
                    zmax=4.0,
                    height=350,
                ),
                use_container_width=True,
            )

        # Quantitative noise reduction metrics
        raw_std = float(np.nanstd(results["post_vv_unfiltered_db"]))
        filt_std = float(np.nanstd(post_vv_db))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Unfiltered Post-Flood σ⁰ StdDev", f"{raw_std:.2f} dB")
        m2.metric("Lee Filtered Post-Flood σ⁰ StdDev", f"{filt_std:.2f} dB", delta=f"{filt_std - raw_std:.2f} dB")
        m3.metric("Active VV Threshold", f"{change_res.effective_post_vv_thresh_db:.2f} dB")
        m4.metric("Computed Otsu VV Threshold", f"{results['otsu_post_vv_db']:.2f} dB")

    # -------------------- TAB 4: FLOOD DETECTION MAP --------------------
    with tabs[3]:
        st.markdown("#### Interactive Geospatial Flood Inundation Map")
        st.caption(
            "Use the layer control in the top-right corner of the map to toggle between Pre-Flood SAR VV, "
            "Post-Flood SAR VV, the Raster Inundation Mask, and Vectorized GeoJSON Flood Polygons. "
            "Hover over polygons to inspect metric area (km² / ha) and perimeter."
        )
        folium_map = build_inundation_folium_map(
            wgs84_bounds=pre_ds.wgs84_bounds,
            pre_vv_db=pre_vv_db,
            post_vv_db=post_vv_db,
            classification_map=change_res.classification_map,
            polygons_gdf=polygons_gdf,
            is_demo=is_any_demo,
        )
        st_folium(folium_map, width=None, height=600, returned_objects=[])

    # -------------------- TAB 5: ANALYTICS --------------------
    with tabs[4]:
        st.markdown("#### Quantitative Inundation & Backscatter Analytics")
        ch_r1_c1, ch_r1_c2 = st.columns(2)
        with ch_r1_c1:
            st.plotly_chart(
                create_backscatter_histogram(
                    pre_vv_db=pre_vv_db,
                    post_vv_db=post_vv_db,
                    vv_threshold_db=change_res.effective_post_vv_thresh_db,
                    otsu_threshold_db=results["otsu_post_vv_db"],
                ),
                use_container_width=True,
            )
        with ch_r1_c2:
            st.plotly_chart(
                create_delta_histogram(
                    delta_vv_db=change_res.delta_vv_db,
                    delta_threshold_db=change_res.delta_drop_thresh_db,
                ),
                use_container_width=True,
            )

        ch_r2_c1, ch_r2_c2 = st.columns(2)
        with ch_r2_c1:
            st.plotly_chart(
                create_vv_vh_comparison_chart(
                    post_vv_db=post_vv_db,
                    post_vh_db=post_vh_db,
                    classification_map=change_res.classification_map,
                ),
                use_container_width=True,
            )
        with ch_r2_c2:
            st.plotly_chart(
                create_area_breakdown_chart(area_metrics),
                use_container_width=True,
            )

        st.markdown("##### Vectorized Flood Polygon Inventory (Top Polygons by Area)")
        if polygons_gdf is not None and not polygons_gdf.empty:
            display_cols = [c for c in polygons_gdf.columns if c != "geometry"]
            st.dataframe(polygons_gdf[display_cols], use_container_width=True, height=280)
        else:
            st.info("No flood polygons exceeded the minimum polygon area threshold under current settings.")

    # -------------------- TAB 6: EXPORT CENTER --------------------
    with tabs[5]:
        st.markdown("#### Geospatial & Statistical Export Center")
        st.markdown(
            "Download standard GIS-compatible products generated by the current processing run. "
            "All vector geometries are exported in **WGS84 (EPSG:4326)** RFC 7946 GeoJSON format, "
            "while raster masks preserve the native Coordinate Reference System."
        )

        geojson_str = export_polygons_to_geojson(polygons_gdf)
        summary_csv_bytes, poly_csv_bytes = metrics_to_csv_bytes(area_metrics, polygons_gdf)
        mask_tiff_bytes = array_to_geotiff_bytes(
            array=change_res.classification_map.astype(np.int16),
            crs=pre_ds.crs,
            transform=pre_ds.transform,
            nodata=-1,
            dtype="int16",
            band_descriptions=["Flood_Classification_0Land_1PermWater_2NewFlood"],
            tags={
                "CLASSES": "-1:Nodata, 0:Land, 1:Permanent_Water, 2:Newly_Inundated_Floodwater",
                "IS_SYNTHETIC_DEMO": str(is_any_demo),
            },
        )

        processing_report = {
            "platform": "RadarFlood SAR Inundation Analytics Platform",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "is_synthetic_demo_data": is_any_demo,
            "inputs": {
                "pre_flood_raster": pre_ds.name,
                "post_flood_raster": post_ds.name,
                "crs": pre_ds.crs.to_string(),
                "dimensions_hw": [pre_ds.height, pre_ds.width],
                "pixel_resolution_xy_m": list(pre_ds.resolution_xy_m),
            },
            "parameters": {
                "pre_detected_scale": pre_vv_cal.detected_scale,
                "post_detected_scale": post_vv_cal.detected_scale,
                "lee_filter_enabled": apply_lee,
                "lee_window_size": lee_window if apply_lee else None,
                "lee_enl": lee_enl if apply_lee else None,
                "effective_post_vv_threshold_db": change_res.effective_post_vv_thresh_db,
                "otsu_reference_vv_threshold_db": round(results["otsu_post_vv_db"], 3),
                "bitemporal_delta_drop_threshold_db": change_res.delta_drop_thresh_db,
                "pre_permanent_water_threshold_db": change_res.effective_pre_water_thresh_db,
                "include_vh": include_vh,
                "morphological_opening": apply_opening,
                "morphological_closing": apply_closing,
                "morph_kernel_size": morph_kernel,
                "min_flood_pixels": min_flood_pixels,
                "min_polygon_area_m2": min_polygon_area_m2,
            },
            "metrics": area_metrics.to_dict(),
        }
        report_json_bytes = json.dumps(processing_report, indent=2).encode("utf-8")

        ex_c1, ex_c2 = st.columns(2)
        with ex_c1:
            st.download_button(
                "🌐 Download Flood Polygons (GeoJSON)",
                data=geojson_str.encode("utf-8"),
                file_name="radarflood_inundation_polygons.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )
            st.download_button(
                "🗺️ Download Classified Flood Mask (GeoTIFF)",
                data=mask_tiff_bytes,
                file_name="radarflood_classification_mask.tif",
                mime="image/tiff",
                use_container_width=True,
            )
        with ex_c2:
            st.download_button(
                "📊 Download Inundation Summary Metrics (CSV)",
                data=summary_csv_bytes,
                file_name="radarflood_summary_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "📋 Download Flood Polygon Inventory (CSV)",
                data=poly_csv_bytes,
                file_name="radarflood_polygon_inventory.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.download_button(
            "📄 Download Complete Processing Summary & Provenance Report (JSON)",
            data=report_json_bytes,
            file_name="radarflood_processing_report.json",
            mime="application/json",
            use_container_width=True,
        )

        with st.expander("Preview Processing Summary JSON"):
            st.json(processing_report)


if __name__ == "__main__":
    main()

