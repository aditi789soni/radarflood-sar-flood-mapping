# RadarFlood – All-Weather SAR Flood Mapping & Inundation Analytics Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![Sensor: Sentinel-1 SAR](https://img.shields.io/badge/SAR-Sentinel--1%20C--Band-0284c7.svg)](https://dataspace.copernicus.eu/)

**RadarFlood** is an end-to-end, open-source Synthetic Aperture Radar (SAR) flood mapping, bitemporal change detection, and geospatial analytics platform designed for **Sentinel-1 C-band GRD/RTC** satellite imagery. Built to demonstrate practical remote sensing and geospatial software engineering capabilities (including for NRSC / ISRO earth observation workflows), the platform implements radiometric $\sigma^0$ decibel calibration, adaptive Lee speckle filtering, bitemporal inundation classification, projected metric area calculation, and interactive GIS visualization.

---

## 1. Key Features

- **Dual-Polarization SAR Input & Validation:** Supports single-pol (`VV`) and dual-pol (`VV + VH`) GeoTIFF rasters, validating Coordinate Reference Systems (CRS), affine transforms, spatial overlap, resolution, and nodata masks.
- **Automatic Backscatter Scale Identification ($\sigma^0$ dB):** Distinguishes between linear power ratio ($\sigma^0_{\text{linear}}$), already-calibrated decibel backscatter ($\sigma^0_{\text{dB}}$), and uncalibrated digital numbers ($\text{DN}$). Converts positive linear intensity via $\sigma^0_{\text{dB}} = 10 \log_{10}(\sigma^0_{\text{linear}})$ while preventing accidental double logarithmic conversion.
- **Adaptive Lee MMSE Speckle Filter:** NumPy/SciPy implementation of the Lee (1980) Minimum Mean Square Error filter supporting configurable `3×3`, `5×5`, and `7×7` sliding windows, Equivalent Number of Looks ($\text{ENL}$) tuning, edge preservation across land-water boundaries, and normalized convolution over `NaN`/nodata pixels.
- **Multi-Criteria Flood Detection:**
  - Interactive VV backscatter threshold slider (default $-16.0\text{ dB}$).
  - Automatic **Otsu global thresholding** on bimodal water/land histograms.
  - Optional cross-polarized **VH** criterion (`AND` / `OR` fusion).
  - Morphological opening, closing (`3×3`, `5×5`, `7×7` elliptical kernels), and connected-component small-patch removal.
- **Bitemporal Change Detection & Co-Registration:**
  - Automatically reprojects and co-registers misaligned or multi-resolution pre-flood and post-flood rasters onto a common reference grid.
  - Computes $\Delta\sigma^0 = \sigma^0_{\text{post}} - \sigma^0_{\text{pre}}$ ($\text{dB}$) to separate **Permanent Water Bodies** from **Newly Inundated Floodwater**.
- **Georeferenced Vectorization & Metric Area Analytics:**
  - Converts raster inundation masks into vector polygons (`GeoDataFrame` / RFC 7946 `GeoJSON`).
  - Automatically projects geometries to the appropriate metric **UTM Zone** (`EPSG:326xx` / `327xx`) to compute exact areas in $\text{km}^2$ and hectares ($\text{ha}$) rather than geographic degrees.
- **Built-In Synthetic SAR Demonstration Mode:** Includes a one-click georeferenced synthetic Sentinel-1 SAR generator (`EPSG:32646`, $20\text{ m}$ resolution, $\text{ENL}=4.4$) clearly badged as **`SYNTHETIC DEMO DATA`** so the full pipeline runs out-of-the-box without external credentials.

---

## 2. System Architecture

```text
radarflood/
├── app.py                      # Main Streamlit application & 6-tab GIS dashboard
├── requirements.txt            # Open-source Python dependencies
├── README.md                   # Documentation, scientific methodology & deployment guide
├── LICENSE                     # MIT License
├── .gitignore                  # Git exclusions
│
├── core/
│   ├── preprocessing.py        # Nodata masking & bitemporal raster co-registration
│   ├── calibration.py          # Scale detection (linear vs dB vs DN) & 10*log10(sigma0)
│   ├── speckle_filter.py       # Adaptive Lee MMSE speckle filter (3x3, 5x5, 7x7)
│   ├── flood_detection.py      # VV/VH thresholding, Otsu's method & morphological cleanup
│   ├── change_detection.py     # Bitemporal Delta sigma0 & permanent vs new flood separation
│   ├── vectorization.py        # Raster-to-polygon vectorization & GeoJSON serialization
│   └── area_calculation.py     # Projected UTM metric area (km²/ha) & CSV analytics
│
├── visualization/
│   ├── maps.py                 # Interactive Folium map with raster & GeoJSON overlays
│   ├── charts.py               # Plotly backscatter histograms, Delta sigma0 & VV/VH plots
│   └── dashboard.py            # Dark navy/slate GIS theme & KPI components
│
├── utils/
│   ├── config.py               # Scientific constants, thresholds & disclaimers
│   ├── validation.py           # Raster CRS, dimension, overlap & pixel validity checks
│   └── raster_utils.py         # GeoTIFF I/O, UTM CRS selector & synthetic SAR generator
│
├── sample_data/
│   └── README.md               # Guide for synthetic demo & acquiring real Sentinel-1 data
│
└── tests/
    ├── test_preprocessing.py   # Calibration, scale detection, alignment & validation tests
    ├── test_filter.py          # Lee speckle filter variance & edge preservation tests
    ├── test_detection.py       # Thresholding, Otsu, morphology & change detection tests
    └── test_area.py            # Vectorization, UTM area calculation & export tests
```

---

## 3. Scientific Methodology

### 3.1 SAR Backscatter & Sigma Nought ($\sigma^0$)
Calm open water acts as a specular reflector at C-band microwave frequencies ($5.405\text{ GHz}$, wavelength $\lambda \approx 5.55\text{ cm}$), directing radar pulses away from the side-looking antenna and yielding very low normalized radar cross-section ($\sigma^0$). Surroundingrough terrestrial surfaces, agricultural fields, and settlements produce higher diffuse or double-bounce backscatter.

Positive calibrated linear intensity ($\sigma^0_{\text{linear}}$) is converted to decibels ($\text{dB}$):

$$\sigma^0_{\text{dB}} = 10 \cdot \log_{10}\!\left(\sigma^0_{\text{linear}}\right)$$

### 3.2 Adaptive Lee Speckle Filtering
Coherent interference of radar echoes within a resolution cell produces multiplicative **speckle noise**. In the linear intensity domain, the adaptive Lee Minimum Mean Square Error (MMSE) estimator computes:

$$\hat{I}(x,y) = \bar{I}(x,y) + W(x,y)\,\bigl(I(x,y) - \bar{I}(x,y)\bigr)$$

where the adaptive weight $W \in [0, 1]$ is defined by the ratio of local window variance $\sigma_{\text{local}}^2$ to multiplicative speckle noise variance $\sigma_{\text{noise}}^2 = \bar{I}^2 / \text{ENL}$:

$$W = \operatorname{clip}\!\left(\frac{\sigma_{\text{local}}^2 - \sigma_{\text{noise}}^2}{\sigma_{\text{local}}^2},\; 0,\; 1\right)$$

In homogeneous water or agricultural regions ($\sigma_{\text{local}}^2 \le \sigma_{\text{noise}}^2$), $W \to 0$ and speckle is smoothed toward the local mean $\bar{I}$. Across sharp land-water boundaries ($\sigma_{\text{local}}^2 \gg \sigma_{\text{noise}}^2$), $W \to 1$, preserving shoreline edges.

### 3.3 Bitemporal Inundation Change Detection
To prevent permanent rivers, reservoirs, and lakes from inflating emergency flood extent estimates, RadarFlood co-registers the pre-event baseline ($\sigma^0_{\text{pre}}$) and post-event target ($\sigma^0_{\text{post}}$) rasters and evaluates:

$$\Delta\sigma^0_{\text{dB}} = \sigma^0_{\text{post, dB}} - \sigma^0_{\text{pre, dB}}$$

1. **Permanent Water (`Class 1`):** $\sigma^0_{\text{pre}} \le T_{\text{perm}}$ and remains low backscatter in the post-flood scene.
2. **Newly Inundated Floodwater (`Class 2`):** Non-water in pre-flood ($\sim \text{land}$), exhibits low post-flood backscatter ($\sigma^0_{\text{post}} \le T_{\text{VV}}$), and undergoes a pronounced backscatter drop ($\Delta\sigma^0 \le \Delta T_{\text{drop}}$, default $-3.0\text{ dB}$).

---

## 4. Scientific & Operational Limitations

1. **Radiometric Calibration & Terrain Correction:** Raw Sentinel-1 Level-1 GRD digital numbers ($\text{DN}$) are **not** calibrated $\sigma^0$. Operational scenes must undergo radiometric LUT calibration and Range-Doppler Terrain Correction (e.g., via ESA SNAP `S1TBX` or Copernicus RTC workflows) prior to analysis.
2. **C-Band Propagation & Surface Scattering:** While C-band SAR penetrates cloud cover and operates day/night, intense convective rain cells can attenuate C-band signals, and C-band microwaves do **not** penetrate dense forest canopies.
3. **Omission & Commission Sources:**
   - *Omission errors:* Strong wind roughening (capillary waves / Bragg scattering), partially submerged emergent vegetation, or urban double-bounce corner reflections can raise water backscatter above $-16\text{ dB}$.
   - *Commission errors:* Topographic radar shadow in steep terrain, smooth dry sand dunes, or airport runways can exhibit low backscatter similar to smooth water.

---

## 5. Installation & Local Execution

### Prerequisites
- Python 3.10+

### Setup & Run
```bash
# 1. Clone the repository
git clone https://github.com/aditi789soni/radarflood-sar-flood-mapping.git
cd radarflood-sar-flood-mapping

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite
python -m pytest -v

# 5. Launch the interactive Streamlit platform
streamlit run app.py
```

---

## 6. Sample Outputs (Synthetic Demonstration Benchmark)

On the default $300 \times 300\text{ px}$ ($20\text{ m}$ resolution, $36.0\text{ km}^2$ total footprint in `EPSG:32646`) simulated Brahmaputra floodplain scene (`Lee 5×5`, `VV Threshold = -16.0 dB`, `Δσ⁰ Drop = -3.0 dB`):

| Metric | Value |
| :--- | :--- |
| **Coordinate Reference System** | `EPSG:32646` (WGS 84 / UTM Zone 46N) |
| **Valid Scene Area** | `36.00 km²` (`3,600.0 ha`) |
| **Permanent Water Baseline Area** | `~3.45 km²` (`9.58%` of scene) |
| **Newly Inundated Floodwater Area** | `~9.15 km²` (`25.42%` of scene) |
| **Total Post-Event Water Area** | `~12.60 km²` (`35.00%` of scene) |
| **Export Formats** | `.geojson` (`EPSG:4326`), `.tif` (`GeoTIFF`), `.csv`, `.json` |

---

## 7. Deployment on Streamlit Community Cloud

1. Push this repository to a public GitHub repository named `radarflood-sar-flood-mapping` (`https://github.com/aditi789soni/radarflood-sar-flood-mapping`).
2. Visit [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** (or **New app**), select repository `aditi789soni/radarflood-sar-flood-mapping`, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy!** Streamlit Community Cloud will automatically install `requirements.txt` and launch the interactive platform with **Synthetic SAR Demonstration Mode** ready out-of-the-box.

---

## 8. License

Released under the [MIT License](LICENSE).

