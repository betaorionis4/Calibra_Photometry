# Photometry with Calibra: Comprehensive User Manual

Welcome to **Calibra** (:an automated photometric analysis & calibration toolkit), a professional-grade astronomical image analysis suite. 
This short manual provides some background on the software's architecture, mathematical principles, and operational workflow.

The code can identify stars, perform PSF fitting to extract fluxes (using aperture photometry), and compares instrumental magnitudes with refernces magnitudes from catalogues available online (e.g. APASS DR9). Note that the provided fits file(s) need to have a WCS (i.e. they need to be plate solved).

Make sure that the code uses the right filter. I use Johnson V and B filters, that are labled 'V_mag' and 'B_mag' by my imaging software - I use N.I.N.A. - in the FITS header.

Future development will be towards using pictures of reference regions taken with both the V and the B filter, and have the code analysze both to extract transformation coeeficients.

Useful background information on the latter and lots of other useful information is provided by the AAVSO Guide to CCD/CMOS Photometry (available for free via aavso.org).

---

## 1. Overview & Core Philosophy

This pipeline utilizes standard algorithms and mathematical modeling to ensure sub-pixel accuracy and rigorous error propagation. It is designed to handle batch processing of FITS images while giving the user granular control via an interactive GUI.

### Key Mathematical Principles
*   **Exact Fractional Integration**: We use the `photutils` library with `method='exact'`, which calculates the precise overlap between circular boundaries and the square pixel grid. This avoids rounding errors common in simpler "center-point" inclusion models.
*   **PSF Modeling**: By fitting a 2D Gaussian profile to every star, we derive centers to **sub-pixel accuracy** (e.g., $X=412.34$) and calculate the true **Full Width at Half Maximum (FWHM)**.
*   **Poisson CCD Noise Propagation**: Every flux measurement is accompanied by a formal uncertainty ($\sigma$) derived from the camera's Gain, Read Noise, and Background variance.

---

## 2. Theory of Operation

### 2.1 The "Target Aperture vs. Sky Annulus" Model
At the heart of the pipeline is a classic geometric subtraction model used to isolate the true optical flux of a star from the ambient "sky glow."

1.  **Geometry Layout**:
    - **Target Aperture**: A circular mask enclosing the star.
    - **Background Annulus**: A hollow ring explicitly spaced outward from the star to sample empty, unpolluted sky.
2.  **Subtraction Math**: 
    - The algorithm calculates the **Median Background per Pixel** within the annulus using sigma-clipping to reject nearby polluting stars or hot pixels.
    - This constant is multiplied by the area of the central aperture to determine the *estimated footprint of background pollution* inside the star measurement.
    - **Net Star Flux = (Total Aperture Sum) - (Background per Pixel × Aperture Area)**.

### 2.2 PSF Fitting & Sub-Pixel Refinement
Raw detection provides integer coordinates. To achieve scientific precision, the pipeline performs **2D Gaussian PSF Fitting**:
- A small "cutout" (stamp) is isolated around every candidate star.
- A Levenberg-Marquardt least-squares algorithm fits a mathematical profile: $f(x,y) = A \exp\left(-\frac{(x-x_0)^2 + (y-y_0)^2}{2\sigma^2}\right)$.
- This yields the precise optical center ($x_0, y_0$) and the stellar width ($\sigma$).

### 2.3 Error Math & SNR
The pipeline calculates the statistical uncertainty using the standard CCD noise equation:
$$\sigma_{flux}^2 = \frac{Flux}{Gain} + Area_{ap} \cdot \sigma_{bg}^2 + \frac{Area_{ap}^2 \cdot \sigma_{bg}^2}{Area_{annulus}}$$
- **SNR (Signal-to-Noise Ratio)**: Calculated as $Net Flux / \sigma_{flux}$. 
- **Filtering**: Detections with $SNR < 3.0$ are automatically flagged as unreliable.

### 2.4 Online Catalog Transformations
Since catalogs like **ATLAS-RefCat2** and **Gaia DR3** do not natively use the Johnson V/B filters, Calibra applies rigorous mathematical transformations to convert their native photometry for zero-point calibration.

#### ATLAS (Pan-STARRS) to Johnson V/B
Based on **Kostov et al. (2017)**, using the specific refined coefficients currently implemented in Calibra:
- $V = g - 0.020 - 0.498(g-r) - 0.008(g-r)^2$
- $B = g + 0.199 + 0.540(g-r) + 0.016(g-r)^2$
- *(Note: Alternative Jester et al. (2005) equations are preserved as comments in the source code).*

#### Gaia DR3 to Johnson V/B
Based on **Riello et al. (2021) Table C.2**, which provides coefficients for the full color range ($-0.5 < G_{BP} - G_{RP} < 5.0$):
- $V = G - 0.02704 + 0.01424 \cdot C - 0.1604 \cdot C^2 + 0.02334 \cdot C^3$
- $B = G + 0.2162 + 0.7303 \cdot C - 0.1518 \cdot C^2 + 0.02447 \cdot C^3$
- *where $C = G_{BP} - G_{RP}$*

---

## 3. The Processing Pipeline (A-F)

The pipeline processes each FITS file through six sequential modules:

### Stage A: Star Detection (`star_detection.py`)
Uses `DAOStarFinder` to scan for density peaks matching a stellar profile. It enforces strict morphological constraints (Sharpness and Roundness) to prevent hot pixels or satellite trails from being flagged as stars.

### Stage B: Coordinate Refinement (`psf_fitting.py`)
Isolates every detected star and fits the 2D Gaussian. 
- **Diagnostics**: Generates radial profile plots and residual images to verify the quality of the fit.
- **Saturation Check**: Flags stars exceeding the hardware linearity limit.

### Stage C: Aperture Photometry (`aperture_phot.py`)
Applies the "Aperture vs. Annulus" math described in Section 2.1. It reports the background-subtracted ADU counts and applies the CCD noise equation.

### Stage D: Data Filtering (`main.py`)
Purges bad data. Any star with negative net flux or an $SNR < 3.0$ is discarded. The pipeline reports the instrumental magnitude of the faintest surviving star as the **Detection Limit**.

### Stage E: Calibration & Zero Points (`calibration.py`)
Matches detected stars against a reference catalog using a spatial **KD-Tree**.
- **Catalog Sources**: Supports local CSV files or **Automated Online Retrieval** from **ATLAS-RefCat2**, **APASS DR9**, or **Gaia DR3** via VizieR.
- **Local Caching**: Online query results are cached locally in `photometry_refstars/cache/` to minimize network usage and allow offline repeat analysis.
- **Filtering**: Only stars with an SNR above the user-defined threshold (default 10.0) are used to calculate the zero point.
- **Zero Point (ZP)**: Derives a **Median Zero Point (ZP)** and applies it to all targets to find their **Calibrated Magnitudes**.

### Stage F: Shift Analysis (`shift_analysis.py`)
Computes the difference between where the FITS header (WCS) *thinks* the star is and where the pixel math *proves* it is. 
- **Console Summary**: Provides a quick median shift report (dX, dY, dRA, dDec) on screen.
- **Detailed Report**: Generates a full markdown report detailing individual tracking errors and rotational shifts.

---

## 4. Running the Pipeline: The Configuration GUI

Launch the pipeline via `python main.py` to open the **Configuration GUI**.

1.  **I/O & Filtering Tab**: Define input file patterns (e.g., `fitsfiles/*.fits`) and restrict analysis to specific pixel or RA/DEC windows.
2.  **Camera & Detection Tab**: Input your sensor's specific **Gain** and **Read Noise**. These are mandatory for accurate error bars.
3.  **Photometry & Calibration Tab**: 
    - Set your aperture radii (rule of thumb: $\approx 2 \times FWHM$).
    - Select your **Ref Catalog**: Choose **ATLAS** (ATLAS-RefCat2), **APASS** (DR9), or **GAIA_DR3** for online calibration, or select a local CSV.
    - Set **Min SNR for Calib**: Filter out noisy stars from the zero-point calculation (default 10.0).
4.  **Output Toggles Tab**: 
    - **Print Detailed Calibration**: Toggle the individual "Match" logs in the console. (Summary always shown).
    - Enable/disable diagnostic plots and massive data tables.

---

## 5. Understanding the Output

### 5.1 Results CSV (`photometry_output/`)
The primary output for every image. Key columns include:
- `refined_x` / `refined_y`: The high-precision sub-pixel coordinates.
- `ra_hms` / `dec_dms`: Celestial coordinates from the WCS header.
- `net_flux`: Background-subtracted ADU counts.
- `mag_calibrated`: The final, zero-point corrected true magnitude.
- `mag_calibrated_err`: The ± uncertainty of the final magnitude.

### 5.2 Diagnostic Plots (`photometry_plots/`)
If enabled, the pipeline saves a four-panel graphic for each star showing:
1.  **Raw Data**: The original pixel cutout.
2.  **Gaussian Model**: The idealized mathematical fit.
3.  **Residuals**: The difference (should look like random noise if the fit is good).
4.  **Radial Profile**: A 1D cross-section showing pixel intensity vs. distance from the center.

---

## 6. Troubleshooting & Tips
- **No Stars Found?** Verify your `Detection Sigma`. Lower it (e.g., to 3.0) for faint targets or increase it (e.g., to 10.0) for crowded fields.
- **Calibration Failures?** 
    - Check if the FITS header has valid `RA`/`DEC` keywords for online queries.
    - If using online catalogs, ensure you have an active internet connection for the first run (subsequent runs use the local cache).
    - Verify that your `Min SNR for Calib` is not set so high that no stars are matched.
- **Positional Drift?** Check the **Shift Analysis** console summary. Large consistent shifts indicate a mount tracking issue or an inaccurate WCS header.
- **Slow Performance?** Check if `Print Detailed Calibration` is on; for batch processing, turning this off keeps the console clean and fast.
