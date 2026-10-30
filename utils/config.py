"""Configuration constants and scientific defaults for RadarFlood."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SARConfig:
    """Default parameters and bounds for SAR flood mapping."""

    # Default thresholds in decibels (dB)
    DEFAULT_VV_THRESHOLD_DB: float = -16.0
    MIN_VV_THRESHOLD_DB: float = -28.0
    MAX_VV_THRESHOLD_DB: float = -5.0

    DEFAULT_VH_THRESHOLD_DB: float = -22.0
    MIN_VH_THRESHOLD_DB: float = -32.0
    MAX_VH_THRESHOLD_DB: float = -10.0

    # Bitemporal change detection defaults (Delta sigma0 = Post - Pre in dB)
    DEFAULT_CHANGE_DROP_DB: float = -3.0
    MIN_CHANGE_DROP_DB: float = -12.0
    MAX_CHANGE_DROP_DB: float = -0.5

    # Permanent water reference threshold in Pre-flood image (dB)
    DEFAULT_PERMANENT_WATER_DB: float = -17.0

    # Lee filter allowed window sizes
    ALLOWED_LEE_WINDOWS: tuple[int, ...] = (3, 5, 7)
    DEFAULT_LEE_WINDOW: int = 5

    # Morphological & post-processing defaults
    DEFAULT_MORPH_KERNEL_SIZE: int = 3
    DEFAULT_MIN_FLOOD_PIXELS: int = 16
    DEFAULT_MIN_POLYGON_AREA_M2: float = 2500.0  # 0.0025 km^2

    # Value heuristics for auto-detecting dB vs linear scale
    DB_PLAUSIBLE_MIN: float = -60.0
    DB_PLAUSIBLE_MAX: float = 15.0

    # Safe numerical floor for linear intensity before log10 conversion
    EPSILON_LINEAR: float = 1e-10

    # Standard nodata sentinel
    DEFAULT_NODATA: float = -9999.0

    # Maximum dimension for interactive preview rendering to keep memory low
    MAX_PREVIEW_DIM: int = 800


CONFIG = SARConfig()

SCIENTIFIC_DISCLAIMER = (
    "**Scientific & Operational Disclaimer:**\n"
    "1. **Radiometric Calibration & Terrain Correction:** Raw Sentinel-1 Level-1 GRD digital numbers (DN) "
    "must undergo radiometric calibration to $\\sigma^0$ (sigma nought) and Range-Doppler terrain correction "
    "(e.g., via ESA SNAP or Copernicus workflows) prior to rigorous quantitative inundation analysis. "
    "Arbitrary uncalibrated DN values are not equivalent to calibrated $\\sigma^0$.\n"
    "2. **C-Band SAR Physics & Limitations:** Sentinel-1 operates in C-band (5.405 GHz, $\\sim 5.55\\text{ cm}$ wavelength). "
    "While C-band SAR penetrates clouds and operates day/night, it can be attenuated by heavy tropical precipitation cells "
    "and does **not** fully penetrate dense forest canopies. Smooth open water acts as a specular reflector yielding low "
    "backscatter, whereas wind-roughened water, emergent vegetation, or urban double-bounce corner reflections can increase "
    "backscatter and cause omission/commission errors.\n"
    "3. **Threshold Applicability:** A fixed threshold (such as $-16\\text{ dB}$ for VV polarization) is a baseline heuristic "
    "that varies with incidence angle, polarization, wind state, and sensor calibration; it is not universally valid across scenes."
)

