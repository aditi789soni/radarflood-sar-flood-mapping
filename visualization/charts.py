"""Plotly scientific charts and SAR raster visualizations for RadarFlood."""

from typing import Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.area_calculation import FloodAreaMetrics
from utils.config import CONFIG

DARK_BG = "#0f172a"
PANEL_BG = "#1e293b"
TEXT_COLOR = "#f8fafc"
GRID_COLOR = "#334155"
ACCENT_CYAN = "#38bdf8"
ACCENT_RED = "#ef4444"
ACCENT_AMBER = "#f59e0b"
ACCENT_GREEN = "#10b981"


def _downsample_2d(arr: np.ndarray, max_dim: int = 350) -> np.ndarray:
    """Downsample 2D array by stride for fast interactive Plotly rendering."""
    h, w = arr.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / float(max_dim))))
    return arr[::step, ::step]


def create_sar_heatmap(
    arr: np.ndarray,
    title: str,
    colorbar_title: str = "σ⁰ (dB)",
    colorscale: str = "Gray",
    zmin: Optional[float] = -26.0,
    zmax: Optional[float] = 0.0,
    height: int = 400,
) -> go.Figure:
    """Render a 2D SAR backscatter or change array as a clean interactive Plotly heatmap."""
    disp = _downsample_2d(arr)
    fig = go.Figure(
        data=go.Heatmap(
            z=np.flipud(disp),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=colorbar_title, thickness=14),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT_COLOR)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=30, r=30, t=45, b=30),
        height=height,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x"),
    )
    return fig


def create_classification_heatmap(
    classification_map: np.ndarray,
    title: str = "Bitemporal Flood Classification Map",
    height: int = 400,
) -> go.Figure:
    """Render the discrete classification map (0=Land, 1=Permanent Water, 2=New Floodwater)."""
    disp = _downsample_2d(classification_map).astype(float)
    disp = np.where(disp < 0, np.nan, disp)

    colorscale = [
        [0.0, "#1e293b"],   # 0: Non-water Land
        [0.33, "#1e293b"],
        [0.34, "#0284c7"],  # 1: Permanent Water
        [0.66, "#0284c7"],
        [0.67, "#ef4444"],  # 2: Newly Inundated Floodwater
        [1.0, "#ef4444"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=np.flipud(disp),
            colorscale=colorscale,
            zmin=0,
            zmax=2,
            colorbar=dict(
                title="Class",
                tickvals=[0.33, 1.0, 1.67],
                ticktext=["Land", "Perm. Water", "New Flood"],
                thickness=14,
            ),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Class ID: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT_COLOR)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=30, r=30, t=45, b=30),
        height=height,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x"),
    )
    return fig


def create_backscatter_histogram(
    pre_vv_db: np.ndarray,
    post_vv_db: np.ndarray,
    vv_threshold_db: float,
    otsu_threshold_db: Optional[float] = None,
) -> go.Figure:
    """Plot overlapping Pre-Flood and Post-Flood VV backscatter distributions with threshold lines."""
    pre_valid = pre_vv_db[np.isfinite(pre_vv_db)].ravel()
    post_valid = post_vv_db[np.isfinite(post_vv_db)].ravel()

    # Subsample up to 40,000 points for responsive browser rendering
    if pre_valid.size > 40000:
        rng = np.random.default_rng(42)
        pre_valid = rng.choice(pre_valid, size=40000, replace=False)
    if post_valid.size > 40000:
        rng = np.random.default_rng(43)
        post_valid = rng.choice(post_valid, size=40000, replace=False)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=pre_valid,
            nbinsx=90,
            name="Pre-Flood σ⁰ VV (dB)",
            marker_color="#38bdf8",
            opacity=0.60,
            histnorm="probability density",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=post_valid,
            nbinsx=90,
            name="Post-Flood σ⁰ VV (dB)",
            marker_color="#f43f5e",
            opacity=0.60,
            histnorm="probability density",
        )
    )

    fig.add_vline(
        x=vv_threshold_db,
        line_width=2.5,
        line_dash="dash",
        line_color="#fde047",
        annotation_text=f"Active Threshold ({vv_threshold_db:.1f} dB)",
        annotation_position="top left",
    )

    if otsu_threshold_db is not None and abs(otsu_threshold_db - vv_threshold_db) > 0.15:
        fig.add_vline(
            x=otsu_threshold_db,
            line_width=2.0,
            line_dash="dot",
            line_color="#10b981",
            annotation_text=f"Otsu ({otsu_threshold_db:.1f} dB)",
            annotation_position="top right",
        )

    fig.update_layout(
        title=dict(text="Pre-Flood vs. Post-Flood Backscatter Distribution (σ⁰ dB)", font=dict(size=14)),
        barmode="overlay",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(title="Backscatter σ⁰ (dB)", gridcolor=GRID_COLOR),
        yaxis=dict(title="Probability Density", gridcolor=GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=60, b=40),
        height=380,
    )
    return fig


def create_delta_histogram(
    delta_vv_db: np.ndarray,
    delta_threshold_db: float,
) -> go.Figure:
    """Plot histogram of bitemporal backscatter change Delta sigma0 = Post - Pre (dB)."""
    valid = delta_vv_db[np.isfinite(delta_vv_db)].ravel()
    if valid.size > 40000:
        rng = np.random.default_rng(44)
        valid = rng.choice(valid, size=40000, replace=False)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=valid,
            nbinsx=90,
            name="Δσ⁰ = Post − Pre (dB)",
            marker_color="#a855f7",
            opacity=0.75,
            histnorm="probability density",
        )
    )

    fig.add_vline(
        x=delta_threshold_db,
        line_width=2.5,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Flood Drop Threshold ({delta_threshold_db:.1f} dB)",
        annotation_position="top left",
    )
    fig.add_vline(x=0.0, line_width=1.2, line_dash="solid", line_color="#94a3b8")

    fig.update_layout(
        title=dict(text="Bitemporal Backscatter Change Histogram (Δσ⁰ = Post − Pre dB)", font=dict(size=14)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(title="Δσ⁰ (dB)", gridcolor=GRID_COLOR),
        yaxis=dict(title="Probability Density", gridcolor=GRID_COLOR),
        margin=dict(l=40, r=30, t=55, b=40),
        height=380,
    )
    return fig


def create_vv_vh_comparison_chart(
    post_vv_db: np.ndarray,
    post_vh_db: Optional[np.ndarray],
    classification_map: np.ndarray,
) -> go.Figure:
    """Plot VV vs VH polarization comparison grouped by surface inundation class."""
    fig = go.Figure()

    if post_vh_db is None:
        # Single-pol box plot across classes
        for cls_id, cls_name, color in [
            (0, "Non-Water Land", "#38bdf8"),
            (1, "Permanent Water", "#0284c7"),
            (2, "Newly Inundated Flood", "#ef4444"),
        ]:
            vals = post_vv_db[(classification_map == cls_id) & np.isfinite(post_vv_db)]
            if vals.size > 4000:
                vals = np.random.default_rng(cls_id).choice(vals, size=4000, replace=False)
            if vals.size > 0:
                fig.add_trace(go.Box(y=vals, name=cls_name, marker_color=color, boxmean=True))

        fig.update_layout(
            title=dict(text="Post-Flood VV Backscatter by Class (Single-Pol Scene)", font=dict(size=14)),
            paper_bgcolor=DARK_BG,
            plot_bgcolor=PANEL_BG,
            font=dict(color=TEXT_COLOR),
            yaxis=dict(title="Post-Flood σ⁰ VV (dB)", gridcolor=GRID_COLOR),
            height=380,
            margin=dict(l=40, r=30, t=55, b=40),
        )
        return fig

    # Dual-pol scatter plot sampled per class
    rng = np.random.default_rng(101)
    for cls_id, cls_name, color in [
        (0, "Non-Water Land", "#64748b"),
        (1, "Permanent Water", "#38bdf8"),
        (2, "Newly Inundated Flood", "#ef4444"),
    ]:
        mask = (classification_map == cls_id) & np.isfinite(post_vv_db) & np.isfinite(post_vh_db)
        vv_pts = post_vv_db[mask]
        vh_pts = post_vh_db[mask]
        if vv_pts.size > 1200:
            idx = rng.choice(vv_pts.size, size=1200, replace=False)
            vv_pts = vv_pts[idx]
            vh_pts = vh_pts[idx]
        if vv_pts.size > 0:
            fig.add_trace(
                go.Scattergl(
                    x=vv_pts,
                    y=vh_pts,
                    mode="markers",
                    name=cls_name,
                    marker=dict(size=5, color=color, opacity=0.65),
                )
            )

    fig.update_layout(
        title=dict(text="Dual-Polarization Feature Space (Post-Flood VV vs. VH in dB)", font=dict(size=14)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(title="Post-Flood σ⁰ VV (dB)", gridcolor=GRID_COLOR),
        yaxis=dict(title="Post-Flood σ⁰ VH (dB)", gridcolor=GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=30, t=60, b=40),
        height=380,
    )
    return fig


def create_area_breakdown_chart(metrics: FloodAreaMetrics) -> go.Figure:
    """Create a horizontal bar chart comparing scene land cover and water inundation areas (km²)."""
    land_km2 = max(
        0.0,
        metrics.valid_scene_area_km2 - metrics.permanent_water_area_km2 - metrics.new_flood_area_km2,
    )
    categories = ["Newly Inundated Floodwater", "Permanent Water Body", "Non-Flooded Valid Land"]
    values = [metrics.new_flood_area_km2, metrics.permanent_water_area_km2, round(land_km2, 4)]
    colors = ["#ef4444", "#0284c7", "#334155"]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=categories,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f} km²" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=dict(text="Scene Surface Coverage Breakdown (km²)", font=dict(size=14)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(title="Area (km²)", gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        margin=dict(l=40, r=30, t=55, b=40),
        height=380,
    )
    return fig

