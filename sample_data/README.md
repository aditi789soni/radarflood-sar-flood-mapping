# Sample Data & Synthetic SAR Demonstration Mode

RadarFlood is designed to operate out-of-the-box without requiring paid API keys or bundling multi-gigabyte raw Sentinel-1 SAFE archives in the Git repository.

## 1. Built-In Synthetic SAR Demonstration Mode
When you launch `streamlit run app.py`, RadarFlood initializes in **Synthetic SAR Demonstration Mode** (clearly badged as `SYNTHETIC DEMO DATA`).
- Generates a georeferenced **Pre-Flood** and **Post-Flood** dual-polarization (**VV + VH**) C-band SAR raster pair in **UTM Zone 46N (`EPSG:32646`)** over a simulated floodplain coordinate frame at $20\text{ m} \times 20\text{ m}$ spatial resolution.
- Models realistic multi-look Gamma speckle noise ($\text{ENL} = 4.4$, matching Sentinel-1 Interferometric Wide Swath GRD imagery), a meandering permanent river channel ($\sigma^0_{\text{VV}} \approx -21.5\text{ dB}$), newly inundated floodplain zones ($\sigma^0_{\text{VV}} \approx -20.5\text{ dB}$), agricultural backscatter, and urban corner-reflector targets.
- You can also download the sample Pre-Flood and Post-Flood GeoTIFFs directly from the sidebar/upload screen to test local file uploads.

## 2. Acquiring Real Sentinel-1 GRD / RTC GeoTIFFs
To analyze real flood events:
1. **Copernicus Data Space Ecosystem (CDSE):** Download Sentinel-1 Level-1 IW GRD scenes from [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/). Apply Radiometric Calibration ($\sigma^0$) and Range-Doppler Terrain Correction using **ESA SNAP** (`S1TBX`) and export as GeoTIFF.
2. **Microsoft Planetary Computer:** Access radiometrically terrain-corrected (`sentinel-1-rtc`) or GRD (`sentinel-1-grd`) Cloud-Optimized GeoTIFFs (COGs) via the free STAC catalog ([https://planetarycomputer.microsoft.com/catalog](https://planetarycomputer.microsoft.com/catalog)).
3. **ASF Data Search Vertex:** Download RTC GeoTIFF products from the Alaska Satellite Facility ([https://search.asf.alaska.edu/](https://search.asf.alaska.edu/)).

