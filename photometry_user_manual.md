# Photometry with Calibra: Comprehensive User Manual

Welcome to **Calibra** (an automated photometric analysis & calibration toolkit), a professional-grade astronomical image analysis suite. 
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

#### 1.1 Session Logging
Calibra automatically generates a permanent record of every analysis run. 
- **Console Window**: A separate "Process Console" window opens to show real-time progress.
- **Log Files**: All console output is simultaneously saved to a `.txt` file in the `photometry_output/logs/` directory.
- **Naming Convention**: Logs are named uniquely to prevent overwriting, e.g., `log_[FileName]_[Filter]_[Catalog].txt`. This allows you to track results across different calibration attempts or reference catalogs.

## 2.0 FITS Pre-processing (Calibration)
Calibra can perform basic instrumental calibration (Bias and Flat-fielding) for raw FITS files directly from the camera.

> [!IMPORTANT]
> **Plate Solving Required**: Even when using Calibra's pre-processing, your raw FITS files **must already be plate solved** (i.e., contain valid WCS headers like RA and DEC). Calibra uses these coordinates to match stars with online catalogs. If your file is not plate solved, automated calibration will fail.

## 2.1 Enabling Calibration
In the **"Pre-processing"** tab of the Configuration GUI:
1.  Check **"Enable Pre-processing (Apply Bias/Flats)"**.
2.  Select your **Master Bias** file (Default: `C:\Astro\Photometry_Calibra\bias_and_flats\Master_Bias_1x1_gain_0.fits`).
3.  Select your **Master Flat** files for both V-mag and B-mag.

## 2.2 Automatic Filter Detection
Calibra automatically reads the `FILTER` keyword from the FITS header of your target image.
- If the filter contains **"B"**, the B-mag Master Flat is used.
- Otherwise, the V-mag Master Flat is used by default.

## 2.3 Output of Calibrated Files
When pre-processing is enabled, Calibra performs the following operation:
`Calibrated = (Raw - Bias) / (Flat / median(Flat))`
I.e. it is assumed that the flat is already corrected for bias!

The resulting calibrated images are saved as new FITS files in a `calibrated/` subfolder located within your input FITS directory (e.g., `C:\Astro\Photometry_Calibra\fitsfiles\calibrated\`). These files are then used for the subsequent star detection and photometry steps.

## 2.4 Online Catalog Transformations
Calibra defaults to **ATLAS-RefCat2** for high-precision zero-point calibration, but also supports **APASS DR9**, **GAIA_DR3**, and the **Landolt Standard Star Catalogue**.

Since **ATLAS-RefCat2** and **GAIA_DR3** do not natively use the Johnson V/B filters, Calibra applies rigorous mathematical transformations to convert their native photometry for zero-point calibration. 

**APASS DR9** and **Landolt Standard Star Catalogue** provide native Johnson V/B measurements natively.

#### ATLAS-RefCat2 
ATLAS-RefCat2 provides high-quality, all-sky photometry in g, r, i, z, and y bands. The catalog is derived from the Legacy Survey of Space and Time (LSST) Pre-Operations Color Camera (POC) and is widely used for photometric calibration due to its high precision and comprehensive sky coverage.

Transformations to Johnson V/B:
Based on **Kostov et al. (2017)**, using the specific refined coefficients currently implemented in Calibra:
- $V = g - 0.020 - 0.498(g-r) - 0.008(g-r)^2$
- $B = g + 0.199 + 0.540(g-r) + 0.016(g-r)^2$
- *(Note: Alternative Jester et al. (2005) equations are preserved as comments in the source code).*

#### Landolt Standard Star Catalogue
Calibra automatically aggregates data from four Landolt standard fields catalogs (VizieR identifiers II/183A, J/AJ/137/4186, J/AJ/133/2502, J/AJ/146/131). 
*Note: Landolt standard stars are only present in specific celestial regions (e.g., equatorial SA fields).*

#### APASS DR9
APASS provides high-quality, Johnson-band calibrated magnitudes for the entire visible sky, derived from the AAVSO Photometric All-Sky Survey.

#### Gaia DR3
Gaia is a space observatory mission led by the European Space Agency (ESA), providing astrometric, photometric, and spectrophotometric data for celestial objects. 

Transformations to Johnson V/B:
Based on **GAIA DR3 Documentation,Table 5.9**, which provides coefficients for V in the color range ($-0.5 < G_{BP} - G_{RP} < 5.0$) and for B in the color range ($-0.5 < G_{BP} - G_{RP} < 4.0$):
- $V = G + 0.02704 - 0.01424 \cdot C + 0.2156 \cdot C^2 - 0.01426 \cdot C^3$
- $B = G - 0.01448 + 0.6874 \cdot C + 0.3604 \cdot C^2 - 0.06718 \cdot C^3 + 0.006061 \cdot C^4$
- *where $C = G_{BP} - G_{RP}$*

## 2.5 AAVSO VSX Integration (Variable Star Exclusion)
To ensure the highest photometric rigor, Calibra automatically cross-matches all detected stars and reference catalogs against the **AAVSO International Variable Star Index (VSX)** via VizieR (`B/vsx/vsx`).

- **Calibration Rigor**: Any reference star found within 2 arcseconds of a known VSX variable is automatically excluded from the Zero-Point derivation.
- **Statistical Purity**: Known variables are excluded from the Gaussian fits in Accuracy Evaluation plots to prevent their intrinsic fluctuations from artificially inflating the reported scatter ($\sigma$).
- **Identification**: All output CSVs and reports include an `is_variable` flag ("Yes"/"No") to help you identify known variable sources in your field at a glance.
- **Local Caching**: VSX data is cached in `photometry_refstars/cache/` to minimize network overhead during repeat analysis.

---

## 3. The Processing Pipeline (A-G)

The pipeline processes each FITS file through seven sequential modules:

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

### Stage G: Color Transformation Calibration (`color_calibration.py`)
A specialized post-processing tool for deriving instrumental **Color Terms**. 
- **Requirements**: Requires a B-filter result CSV and a V-filter result CSV of the same field.
- **Airmass & Extinction**: Automatically handles airmass extraction from results and applies atmospheric extinction correction ($k_B, k_V$).
- **Iterative Fitting**: 
    - Applies a **2-sigma outlier rejection** algorithm. 
    - First pass calculates the initial spread; second pass performs a high-precision regression on the cleaned data.
- **Visual Diagnostics**:
    - **Red 'X'**: Outliers that were statistically rejected.
    - **Red Dotted Lines**: The initial 2-sigma boundary of the raw data.
    - **Grey Shaded Corridor**: The final, refined 2-sigma confidence window.
- **Coefficients Derived**:
    - **$\mu$ (Color Scale)**: Transformation from $(b-v)$ to $(B-V)_{std}$.
    - **$\psi$ & $\epsilon$**: Color terms for B and V filters respectively.
- **Reporting**: Generates a regression report and three diagnostic plots showing color residuals and the statistical cleaning process.

---

## 4. Differential Photometry

The Differential Photometry module is intended to compute AAVSO-ready standard magnitudes for every star in a field relative to a designated reference star. The reference star can be either automatically selected by the pipeline or manually specified by the user to focus on measuring magnitudes of variable stars using a known comparison star.

### 4.1 Methodology
- **Target Matching**: The module automatically cross-matches stars from your B-filter and V-filter results (using a 2-arcsecond search radius) to form reliable $(b-v)$ pairs.
- **Reference Catalog Query**: The matched coordinates are queried against your chosen standard catalog (e.g., ATLAS, APASS, Landolt). The catalog search covers a field of view that is defined by the "Catalog Search Radius" in the GUI.
- **Reference Star Selection**:
  - **Automatic Mode**: The pipeline filters the catalog matches to automatically select the optimal reference star. It strictly chooses a star that is:
    1. Unsaturated (peak ADU below the non-linear regime).
    2. Not a known variable (cross-matched against VSX).
    3. Of moderate color (catalog $0.4 \leq (B-V) \leq 0.8$) to minimize extreme transformation residuals.
    4. The brightest available instrumental $V$ magnitude among the remaining candidates.
  - **Search by Name Mode**: The user inputs a specific star name (e.g., "AE UMa"). The pipeline resolves the exact RA and Dec coordinates dynamically via the SIMBAD astronomical database and anchors the photometry to this object. You can instantly verify the resolved coordinates using the Check button in the GUI before executing.
  - **Manual Mode**: The user inputs specific RA and Dec coordinates ($h, m, s$ and $d, m, s$). The pipeline finds the matched detection within a 4-arcsecond tolerance of those coordinates, verifies it has catalog data, and strictly forces it to be the reference anchor.
- **Zero Point Calculation**: Using the Color Transformation Coefficients ($T_{bv}, T_{b\_bv}, T_{v\_bv}$) and atmospheric extinction ($k_B, k_V$), the pipeline derives the instrumental zero points ($Z_{BV}, Z_B, Z_V$) relative to this reference star. The color transformation coefficients can be taken from the Color Transformation Calibration module (see Section 3.G) or manually entered by the user.

### 4.2 Target Selection Modes
In the Differential Photometry tab, you can define which stars to process:
- **Analyze All Stars (Default)**: The pipeline computes standard magnitudes for every common B/V pair in the image. This is useful for survey work or verifying field-wide accuracy.
- **Analyze a specific Target Pair**: The pipeline isolates a single star (via Name/SIMBAD or Manual Coordinates) and computes standard magnitudes for only that object. This mode skips the population-wide Accuracy Evaluation plotting as it is not statistically valid for a single target.

### 4.3 Standard Magnitude Output
These zero points are instantly applied to all other stars in the field to get standard magnitudes ($B, V$) and color index ($B-V$). The final standard magnitudes and color index are saved in a Markdown table (`differential_photometry_results.md`) alongside a more detailed CSV file (with all the instrumental data, errors, and variability flags).

### 4.4 Accuracy Evaluation
To evaluate the calibration quality, the pipeline automatically compares the internally computed standard magnitudes against the actual catalog magnitudes for all matching stars (excluding the reference anchor and any known VSX variables).
- **Statistical Fitting**: It calculates the deviations ($\Delta B$, $\Delta V$, $\Delta(B-V)$) and fits a Gaussian distribution to determine the mean offset ($\mu$) and standard deviation/scatter ($\sigma$).
- **Plotting**: It generates a 3-panel histogram plot with the Gaussian fits overlaid (`photometry_plots/diff_photometry_deviations.png`), allowing you to rapidly identify any systematic errors or estimate your measurement uncertainties. The statistical fits are appended to the `differential_photometry_report.md`.

---

## 5. Running the Pipeline: The Configuration GUI

Launch the pipeline via `python main.py` to open the **Configuration GUI**.

1.  **I/O & Filtering Tab**: Define input file patterns (e.g., `fitsfiles/*.fits`) and restrict analysis to specific pixel or RA/DEC windows.
2.  **Camera & Detection Tab**: Input your sensor's specific **Gain** and **Read Noise**. These are mandatory for accurate error bars.
3.  **Photometry & Calibration Tab**: 
    - Set your aperture radii (rule of thumb: $\approx 2 \times FWHM$).
    - Select your **Ref Catalog**: Choose **ATLAS** (ATLAS-RefCat2), **APASS** (DR9), or **GAIA_DR3** for online calibration, or select a local CSV.
    - Set **Catalog Search Radius (arcmin)**: Set this to encompass your Field of View. This dictates how much catalog data is pulled from VizieR for Zero-Point and Differential operations.
    - Set **Min SNR for Calib**: Filter out noisy stars from the zero-point calculation (default 10.0).
    - **Print Detailed Calibration**: Toggle the individual "Match" logs in the console. (Summary always shown).
    - Enable/disable diagnostic plots and massive data tables.
4.  **Color Calibration Tab**: 
    - **Auto-Hand-off**: After running a B and V image, these fields will auto-populate with the correct CSV paths.
    - **Extinction Correction**: Enter or verify the k-coefficients ($k_B, k_V$) and airmass.
    - Click **"Run Color Transformation Analysis"** to calculate your equipment's color terms.
5.  **Differential Photometry Tab**:
    - **Auto-Fill**: The pipeline automatically loads your most recent `targets_auto` CSV files and your last derived Color Coefficients.
    - **Reference Star Selection**: Toggle between "Automatic" (picks the brightest valid star) or "Manual Coordinates" to strictly anchor your analysis to a specific star.
    - Click **"Execute Differential Photometry"** to perform the full magnitude calculation workflow.

---

## 6. Understanding the Output

### 6.1 Results CSV (`photometry_output/`)
The primary output for every image. Key columns include:
- `refined_x` / `refined_y`: The high-precision sub-pixel coordinates.
- `ra_hms` / `dec_dms`: Celestial coordinates from the WCS header.
- `net_flux`: Background-subtracted ADU counts.
- `mag_calibrated`: The final, zero-point corrected true magnitude.
- `mag_calibrated_err`: The ± uncertainty of the final magnitude.
- `is_variable`: A "Yes/No" flag indicating if the star matched a record in the AAVSO VSX catalog.
- `airmass`: The atmospheric airmass calculated from the FITS header.

### 6.2 Diagnostic Plots (`photometry_plots/`)
If enabled, the pipeline saves a four-panel graphic for each star showing:
1.  **Raw Data**: The original pixel cutout.
2.  **Gaussian Model**: The idealized mathematical fit.
3.  **Residuals**: The difference (should look like random noise if the fit is good).
4.  **Radial Profile**: A 1D cross-section showing pixel intensity vs. distance from the center.

---

## 7. Troubleshooting & Tips
- **No Stars Found?** Verify your `Detection Sigma`. Lower it (e.g., to 3.0) for faint targets or increase it (e.g., to 10.0) for crowded fields.
- **Calibration Failures?** 
    - Check if the FITS header has valid `RA`/`DEC` keywords for online queries.
    - If using online catalogs, ensure you have an active internet connection for the first run (subsequent runs use the local cache).
    - Verify that your `Min SNR for Calib` is not set so high that no stars are matched.
- **Positional Drift?** Check the **Shift Analysis** console summary. Large consistent shifts indicate a mount tracking issue or an inaccurate WCS header.
- **Slow Performance?** Check if `Print Detailed Calibration` is on; for batch processing, turning this off keeps the console clean and fast.

## References

**GAIA DR3**
Gaia Collaboration, Vallenari, A., et al. (2023), "Gaia Data Release 3. Summary of the contents and survey properties"
https://doi.org/10.1051/0004-6361/202243940 
https://www.cosmos.esa.int/web/gaia/dr3
For the transformation between G and other photometric systems:
https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html

**APASS DR9**
Henden, A. A., et al. (2016), "AAVSO Photometric All Sky Survey (APASS) DR9"
https://cdsarc.cds.unistra.fr/viz-bin/cat/II/336 

**ATLAS-RefCat2**
Tonry, J. L., et al. (2018), "The ATLAS All-Sky Stellar Reference Catalog"
https://doi.org/10.3847/1538-4357/aae386
https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/867/105

**Landolt Standard Star Catalogue**
Aggregates four primary standard field catalogs via VizieR:
- Landolt, A. U. (1992), "UBVRI photometric standard stars in the magnitude range 11.5 < V < 16.0 around the celestial equator" (Vizier:VII/183A)
https://ui.adsabs.harvard.edu/scan/manifest/1992AJ....104..340L
https://cdsarc.cds.unistra.fr/viz-bin/cat/II/183A
- Landolt, A. U. (2009), "UBVRI photometric standard stars around the celestial equator: Updates and Additions" (Vizier:J/AJ/137/4186)
http://dx.doi.org/10.1088/0004-6256/137/5/4186 
https://cdsarc.cds.unistra.fr/viz-bin/cat/J/AJ/137/4186
- Landolt, A. U. (2007), "UBVRI photometric standard stars around the sky at -50 deg declination" (Vizier:J/AJ/133/2502)
https://iopscience.iop.org/article/10.1086/518000/pdf
https://cdsarc.cds.unistra.fr/viz-bin/cat/J/AJ/133/2502
- Landolt, A. U. (2013), "UBVRI photometric standard stars around the sky at +50 deg declination" (Vizier:J/AJ/146/131)
https://iopscience.iop.org/article/10.1088/0004-6256/146/5/131
https://cdsarc.cds.unistra.fr/viz-bin/cat/J/AJ/146/131
