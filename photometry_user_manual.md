# Calibra: Automated Photometric Analysis & Calibration Toolkit
## Comprehensive User Manual & Technical Guide
**Version:** 4.13  
**Release Date:** May 28, 2026  
**Document Classification:** Scientific-Grade Photometry Guide  

---

### Abstract
Calibra is an automated, high-precision Python toolkit designed for processing astronomical FITS files. It extracts scientific-grade stellar photometry, corrects for atmospheric extinction and instrumental color transformations, generates time-series ensemble differential light curves, and scans image sequences to detect variable star candidates. This manual serves as a complete step-by-step operational guide, detailing every workflow, user configuration parameter, and the interactive graphical tools available.

---

# Table of Contents
1. [Introduction & Core Philosophy](#introduction--core-philosophy)
2. [Chapter 1: Pre-processing](#chapter-1-pre-processing)
   - [1.1 FITS Calibration (Bias & Flat Correction)](#11-fits-calibration-bias--flat-correction)
   - [1.2 WCS Plate Solving (via ASTAP)](#12-wcs-plate-solving-via-astap)
   - [1.3 OSC Channel Split (Bayer Demultiplexing)](#13-osc-channel-split-bayer-demultiplexing)
3. [Chapter 2: Analysis & Calibration](#chapter-2-analysis--calibration)
   - [2.1 Star Detection & Zero-Point Calibration](#21-star-detection--zero-point-calibration)
   - [2.2 Color Coefficient Derivation](#22-color-coefficient-derivation)
   - [2.3 Differential Photometry](#23-differential-photometry)
4. [Chapter 3: Time-Series Light Curves](#chapter-3-time-series-light-curves)
   - [3.1 Setting Up the Target & Check Stars](#31-setting-up-the-target--check-stars)
   - [3.2 Ensemble Comparison Star Selection](#32-ensemble-comparison-star-selection)
   - [3.3 Mathematical Zero-Point Averaging & Uncertainty Propagation](#33-mathematical-zero-point-averaging--uncertainty-propagation)
   - [3.4 Single Reference Star Photometry Mode](#34-single-reference-star-photometry-mode)
   - [3.5 Interactive Plot Customization](#35-interactive-plot-customization)
   - [3.7 Data Rebinning for Improved SNR](#37-data-rebinning-for-improved-snr)
5. [Chapter 4: Data Mining & Variable Star Discovery](#chapter-4-data-mining--variable-star-discovery)
   - [4.1 Statistical Star Tracking & Excess Scatter Detection](#41-statistical-star-tracking--excess-scatter-detection)
   - [4.2 VSX Catalog Integration & Candidate Validation](#42-vsx-catalog-integration--candidate-validation)
   - [4.3 Interactivity & Light Curve Previews](#43-interactivity--light-curve-previews)
6. [Chapter 5: The Interactive FITS Viewer](#chapter-5-the-interactive-fits-viewer)
   - [5.1 Layout & Visual Controls](#51-layout--visual-controls)
   - [5.2 Star Role Assignment (Target, Check, Reference)](#52-star-role-assignment-target-check-reference)
   - [5.3 Real-Time PSF Radial Profile & Aperture Customization](#53-real-time-psf-radial-profile--aperture-customization)
   - [5.4 VSX Variable Overlays & Dynamic Filters](#54-vsx-variable-overlays-and-dynamic-filters)
   - [5.5 Bidirectional Sync with Photometry Tab](#55-bidirectional-sync-with-photometry-tab)
7. [References](#references)
8. [Index](#index)

---

# Introduction & Core Philosophy
Calibra bridges the gap between raw telescope imagery and publishable, AAVSO-compliant photometric results. At the heart of the toolkit are two core principles:
*   **Physical Purity**: Avoiding unnecessary pixel interpolation or smoothing that correlates noise and distorts raw flux readings.
*   **Rigorous Error Propagation**: Every magnitude measurement is accompanied by a formal statistical error derived from the physical characteristics of the CCD/CMOS sensor (Gain, Read Noise) and background sky statistics.

---

# Chapter 1: Pre-processing

Pre-processing prepares raw camera images for analysis. The pre-processing workflows are navigated via three sub-tabs within the **Pre-processing** tab in the main configuration panel.

### 1.1 FITS Calibration (Bias & Flat Correction)
Calculates and removes the camera sensor's thermal/electronic signature (Bias) and compensates for optical vignetting, dust donuts, and pixel-to-pixel sensitivity differences (Flats).

*   **Workflow Steps**:
    1. In the **Pre-processing** tab under **FITS Calibration**, check **Enable Pre-processing (Apply Bias/Flats)**.
    2. Load your Master Bias FITS file into the **Master Bias Path** field.
    3. Load your Master Flat FITS files into the **Master Flat V Path** and **Master Flat B Path** fields.
    4. When images are processed in subsequent analysis steps, the calibration is executed on-the-fly:
       $$\text{Calibrated Image} = \frac{\text{Raw} - \text{Bias}}{\text{Flat} / \text{median}(\text{Flat})}$$
*   **Steering Parameters**:
    *   `bias_path` (*Pre-processing -> FITS Calibration*): File path to Master Bias.
    *   `flat_v_path` (*Pre-processing -> FITS Calibration*): File path to Master V Flat.
    *   `flat_b_path` (*Pre-processing -> FITS Calibration*): File path to Master B Flat.
    *   `filter_b_keyword` / `filter_v_keyword` / `filter_g_keyword` / `filter_r_keyword` / `filter_l_keyword` (*Settings -> Astro & Catalog*): Used to identify the filter in the FITS header (e.g. `Bmag` / `Vmag` / `Gmag` / `Rmag` / `Lmag`) so the pipeline applies the correct flat file or filters the files by channel automatically.

### 1.2 WCS Plate Solving (via ASTAP)
To extract coordinate-aligned catalogs and cross-match stars against online databases (Gaia, APASS, VSX), FITS images must be "plate solved," containing World Coordinate System (WCS) headers (RA, Dec, and pixel scale).

*   **Workflow Steps**:
    1. Navigate to the **WCS Plate Solving** sub-tab.
    2. Input the path to your ASTAP executable in the **ASTAP Executable Path** field.
    3. Select a **Search Radius** (in degrees) to constrain the sky search around the nominal coordinates.
    4. Select checked images in the FITS File Manager and click **Run WCS Plate Solving**. This calls ASTAP command-line arguments to append WCS headers to a copy of each image (named with a suffix, e.g., `_wcs.fits`).
*   **Steering Parameters**:
    *   `astap_path` (*Pre-processing -> WCS Plate Solving*): Path to ASTAP program.
    *   `plate_radius` (*Pre-processing -> WCS Plate Solving*): Extent of sky search (default `5.0` deg).
    *   `plate_suffix` (*Pre-processing -> WCS Plate Solving*): Appended suffix to plate-solved files (default `wcs`).
    *   `plate_annotate` (*Pre-processing -> WCS Plate Solving*): Toggles whether ASTAP generates an annotated catalog overlay on the image.

### 1.3 OSC Channel Split (Bayer Demultiplexing)
One-Shot Color (OSC) camera sensors record images through a color filter array (Bayer matrix). For scientific photometry, raw OSC Bayer frames must be split into separate R, G, and B channels before zero-point or differential analysis to avoid systematic errors.

*   **Workflow Steps**:
    1. Navigate to the **OSC Channel Split** sub-tab.
    2. Select the sensor's **Bayer Pattern** (e.g., `RGGB`, `BGGR`, `GRBG`, `GBRG`).
    3. Choose the **Splitting Method**:
        *   **Superpixel (2x2 -> 1px)** *(Recommended)*: Combines 2x2 grids of Bayer pixels into a single pixel. Red, blue, and green intensities are mapped to separate FITS files. Green intensities are averaged. No interpolation is performed, preserving physical flux. The coordinate projection (`CRPIX1`/`CRPIX2`) is scaled by `0.5`, and the pixel sizes (`CDELT1`/`CDELT2`) are doubled, preserving WCS accuracy.
        *   **Bilinear (Interpolated)**: Interpolates missing pixels in each channel to maintain the original image dimensions. (Caution: This correlates noise and is not recommended for high-precision science).
    4. Select FITS files in the manager and click **Execute Channel Split**. Split files (e.g. `_SP_R.fits`, `_SP_G.fits`, `_SP_B.fits`) are saved and imported back into the manager.
*   **Steering Parameters**:
    *   `osc_pattern` (*Pre-processing -> OSC Channel Split*): Sensor Bayer pattern.
    *   `osc_method` (*Pre-processing -> OSC Channel Split*): Superpixel vs Bilinear.
    *   `osc_chan_r` / `osc_chan_g` / `osc_chan_b` (*Pre-processing -> OSC Channel Split*): Enable checkboxes to generate the specific color fits files.

---

# Chapter 2: Analysis & Calibration

Photometry and transformation operations are housed under the **Analysis & Calibration** tab, split into three sequential workflow tabs.

### 2.1 Star Detection & Zero-Point Calibration
Extracts stars from images and calibrates their instrumental magnitudes using reference catalogs.

*   **Workflow Steps**:
    1. Set up camera parameters under **Settings -> Camera & Photometry** (Gain, Read Noise, and Saturation Limit).
    2. Select reference catalog under **Settings -> Astro & Catalog** (ATLAS refcat2, APASS DR9, Gaia DR3, or Landolt).
    3. Check files in the manager and click **Run Analysis & Calibration**.
    4. The pipeline performs:
       *   **Star Detection**: Uses `DAOStarFinder` to identify star peaks above background.
       *   **Coordinate Refinement**: Fits a 2D Gaussian to star cutouts.
       *   **Aperture Photometry**: Measures raw flux within the aperture radius and subtracts the median background per pixel from the sky annulus:
           $$\text{Net Flux} = F_{\text{ap}} - (\text{Sky}_{\text{median}} \times \text{Area}_{\text{ap}})$$
       *   **Uncertainty Calculation**: Applies the standard CCD noise equation:
           $$\sigma_{\text{flux}}^2 = \frac{\text{Flux}}{G} + \text{Area}_{\text{ap}} \cdot \sigma_{\text{bg}}^2 + \frac{\text{Area}_{\text{ap}}^2 \cdot \sigma_{\text{bg}}^2}{\text{Area}_{\text{annulus}}}$$
       *   **Catalog Matching**: Matches detected stars against catalog coordinates.
       *   **Zero-Point Calibration**: Computes zero points ($ZP_i = m_{\text{cat},i} - m_{\text{inst},i}$) for stars with $\text{SNR} > \text{SNR Threshold}$. Applies the median ZP to calibrate all targets.
*   **Steering Parameters**:
    *   `ccd_gain` (*Settings -> Camera & Photometry*): CCD/CMOS gain in $e^-/\text{ADU}$.
    *   `ccd_read_noise` (*Settings -> Camera & Photometry*): Readout noise in $e^-$.
    *   `saturation_limit` (*Settings -> Camera & Photometry*): Value above which pixels are marked saturated (default `63000` ADU).
    *   `detect_sigma` (*Settings -> Camera & Photometry*): Detection threshold standard deviation above background (default `5.0`).
    *   `aperture_radius` / `annulus_inner` / `annulus_outer` (*Settings -> Camera & Photometry*): Photometry aperture radii.
    *   `use_flexible_aperture` (*Settings -> Camera & Photometry*): Toggles whether aperture radius scales dynamically based on the star's FWHM.
    *   `reference_catalog` (*Settings -> Astro & Catalog*): Standard catalog source.
    *   `calib_snr_threshold` (*Settings -> Astro & Catalog*): Minimum SNR for a star to be used to calculate the zero point (default `10.0`).
    *   `default_zp_v` / `default_zp_b` (*Settings -> Astro & Catalog*): Fallback zero points if catalog query fails.

### 2.2 Color Coefficient Derivation
Computes the instrumental color transformation coefficients ($\mu$, $\psi$, $\epsilon$) using matched V-filter and B-filter calibration outputs.

*   **Workflow Steps**:
    1. Select the B-filter results CSV and the V-filter results CSV.
    2. Provide Extinction Coefficients ($k_B, k_V$) and standard Airmass values (if not reading them from FITS headers).
    3. Click **Compute Color Coefficients**.
    4. The pipeline fits regressions comparing instrumental color index $(b-v)$ to catalog color index $(B-V)$:
       $$(B-V) = \mu \cdot (b-v)_0$$
       $$V = v_0 + \epsilon \cdot (B-V) + ZP_V$$
       $$B = b_0 + \psi \cdot (B-V) + ZP_B$$
*   **Steering Parameters**:
    *   `color_b_csv` / `color_v_csv` (*Analysis & Calibration -> 2. Color Coefficients*): Results CSV paths.
    *   `extinction_kb` / `extinction_kv` (*Settings -> Astro & Catalog*): Extinction coefficients.
    *   `override_airmass` (*Analysis & Calibration -> 2. Color Coefficients*): Toggles manual airmass override.

### 2.3 Differential Photometry
Measures a target star's magnitude relative to standard reference comparison stars, outputting AAVSO-compliant magnitudes.

*   **Workflow Steps**:
    1. Load B-filter and V-filter calibration CSVs.
    2. Enter transformation coefficients ($\mu$, $\psi$, $\epsilon$) derived in Sub-chapter 2.2.
    3. Set **Reference Selection Mode**:
       *   `auto`: Automatically chooses the brightest, unsaturated, non-variable standard star in the field with a color range between $0.4 \leq (B-V) \leq 0.8$.
       *   `name`: Queries SIMBAD by object name (e.g. `AE UMa`) to resolve coordinates.
       *   `manual`: Forces the user-specified RA and Dec coordinates as the comparison anchor.
    4. Set **Target Selection Mode** (`all` stars in the field or a `specific` star).
    5. Click **Compute Differential Photometry**.
*   **Steering Parameters**:
    *   `diff_tbv` / `diff_tbbv` / `diff_tvbv` (*Analysis & Calibration -> 3. Differential Photometry*): Transformation coefficients ($\mu$, $\psi$, $\epsilon$ respectively).
    *   `ref_mode` (*Analysis & Calibration -> 3. Differential Photometry*): `auto`, `name`, or `manual`.
    *   `ref_star_name` (*Analysis & Calibration -> 3. Differential Photometry*): Comparison star name for `name` mode.
    *   `target_mode` (*Analysis & Calibration -> 3. Differential Photometry*): `all` or `specific`.

---

# Chapter 3: Time-Series Light Curves

Generates high-precision light curves from sequences of FITS files, correcting for systematic variations using an ensemble of up to 5 comparison stars.

### 3.1 Setting Up the Target & Check Stars
*   **Workflow Steps**:
    1. Navigate to the **Time Series** tab.
    2. Provide the Target Star details: Select **Target Selection Mode** (`name` or `manual`). Input the target's name or coordinates, and provide its standard color index ($B-V$) in the **Target B-V** field.
    3. (Optional) Define a **Check Star** under the Check Star section. This acts as an independent constant check star to verify photometric precision.
*   **Steering Parameters**:
    *   `ts_target_mode` (*Time Series -> Target Star*): Target lookup mode (`name` or `manual`).
    *   `ts_target_name` (*Time Series -> Target Star*): Target identifier.
    *   `ts_target_bv` (*Time Series -> Target Star*): Assumed target color (default `0.0` if unknown).
    *   `ts_check_star_idx` (*Time Series -> Check Star*): Comparison index to use as the check star.

### 3.2 Ensemble Comparison Star Selection
Ensemble photometry utilizes multiple comparison stars to calculate a robust zero point, protecting the data from light curve fluctuations if a single reference star varies or falls on a bad pixel.
*   **Workflow Steps**:
    1. Enter up to 5 comparison star designations in the **Ensemble Stars** fields.
    2. Click **Fetch Coords & Mags** to resolve coordinates and retrieve catalog magnitudes (V and B-V) from the database automatically.
*   **Steering Parameters**:
    *   `ts_ref_i_name` (*Time Series -> Ensemble Stars*): Comparison star identifier.
    *   `ts_ref_i_mag` / `ts_ref_i_bv` (*Time Series -> Ensemble Stars*): Standard catalog magnitude and color index.
    *   `ts_ref_i_use` (*Time Series -> Ensemble Stars*): Toggles reference inclusion in the ensemble.

### 3.3 Zero-Point Averaging & Uncertainty Propagation
For each frame, the pipeline independently calculates the zero point ($ZP_i$) for each of the $N$ reference stars:
$$ZP_i = V_{\text{std},i} - \left( m_{\text{inst},i} - k \cdot X + T \cdot (B-V)_i \right)$$
The ensemble Zero Point is the average:
$$\overline{ZP} = \frac{1}{N} \sum_{i=1}^{N} ZP_i$$
The target's calibrated magnitude is:
$$V_{\text{target}} = m_{\text{inst,target}} - k \cdot X + T \cdot (B-V)_{\text{target}} + \overline{ZP}$$
The total propagated uncertainty is:
$$\sigma_V = \sqrt{\sigma_{\text{phot,target}}^2 + \sigma_{\overline{ZP}}^2}$$
where $\sigma_{\overline{ZP}} = \sigma_{ZP} / \sqrt{N}$ represents the standard error of the ensemble mean.
*   **Steering Parameters**:
    *   `ts_do_trans` (*Time Series -> Coefficients & Metadata*): Toggle to enable color transformations.
    *   `ts_coeff` (*Time Series -> Coefficients & Metadata*): Color coefficient ($T$).
    *   `ts_k` (*Time Series -> Coefficients & Metadata*): Extinction coefficient ($k$).

### 3.4 Single Reference Star Photometry Mode
When only a single comparison star is used ($N=1$), zero-point variance calculations ($\sigma_{ZP}$) are mathematically undefined.
*   **Uncertainty Adjustment**: The pipeline applies a zero-point uncertainty floor of `0.01` mag:
    $$\sigma_{\overline{ZP}} = 0.01 \text{ mag}$$
    This prevents underestimating the calibration uncertainty, which could hide frame-to-frame offsets.

### 3.5 Interactive Plot Customization
*   **Workflow Steps**:
    1. Run the time-series analysis to generate the light curve preview.
    2. Click **Customize Plot** to open the **Advanced Plot Customizer** window.
    3. Modify plot settings: Y-axis min/max ranges, plot title, axis labels, specific series colors, and marker shapes.
    4. Click **Apply Settings** to refresh the canvas instantly. Export publication-grade plots using **Save High-Res Plot**.

### 3.6 Automatic Suffix-Based Filtering (OSC / Monochrome)
To ensure the pipeline is robust against duplicate or mixed data configurations, Calibra implements automatic suffix-based filtering when executing time-series analyses:
*   **Behavior**:
    If a selected filter (e.g. `"G"`, `"R"`, `"B"`, `"L"`, or `"V"` if mapped to a color band) has both original/raw Bayer FITS files (e.g., `frame_001.fits`) and decomposed/monochrome channel FITS files (e.g., `frame_001_SP_G.fits` or `frame_001_G.fits`) loaded and checked in the File Manager:
    *   The time-series runner will identify the presence of the split suffixes (e.g., `_SP_G.fits`, `_G.fits`, etc.).
    *   It will automatically isolate and process **only** the split channel files.
    *   The original raw frames are automatically excluded from the run, preventing coordinate alignment, epoch duplication, and shape/size mismatch errors.

### 3.7 Data Rebinning for Improved SNR
When individual exposures are short and signal-to-noise ratios (SNR) are low, consecutive measurements can be combined ("rebinned") to produce a smoother light curve with reduced scatter. This is particularly useful for faint targets or rapid-cadence time-series observations.

*   **Accessing Rebinning**:
    1. Run the time-series analysis to generate the light curve.
    2. Click **⚙️ Customize Plot** to open the **Advanced Plot Customizer** window.
    3. In the **Data Rebinning** section on the left panel, select a **Bin Size** from the dropdown: `None` (original data), `2`, `4`, `8`, or `16`.
    4. The plot updates immediately. An info label shows the transformation, e.g. "80 points → 10 bins (bin size 8)".

*   **Rebinning Algorithm**:
    Consecutive groups of $n$ measurements (where $n$ is the chosen bin size) are combined into a single data point. If the total number of measurements is not evenly divisible by $n$, the final bin contains the remaining points.

    **Magnitude**: The rebinned magnitude is computed as the **inverse-variance weighted mean**:
    $$m_{\text{bin}} = \frac{\sum_{i=1}^{n} w_i \cdot m_i}{\sum_{i=1}^{n} w_i}, \quad w_i = \frac{1}{\sigma_i^2}$$
    where $m_i$ and $\sigma_i$ are the magnitude and uncertainty of the $i$-th measurement. This weighting scheme ensures that higher-precision measurements contribute more strongly to the average.

    **Uncertainty**: The formal uncertainty of the weighted mean is:
    $$\sigma_{\text{formal}} = \frac{1}{\sqrt{\sum_{i=1}^{n} w_i}}$$
    For bins with more than one point, the pipeline also estimates a scatter-based uncertainty from the weighted residuals:
    $$\sigma_{\text{scatter}} = \frac{1}{\sqrt{n}} \sqrt{\frac{\sum_{i=1}^{n} w_i \cdot (m_i - m_{\text{bin}})^2}{\sum_{i=1}^{n} w_i}}$$
    The final reported uncertainty is the **larger** of $\sigma_{\text{formal}}$ and $\sigma_{\text{scatter}}$, ensuring that the error bar is never underestimated when the data within a bin show excess scatter beyond what the formal errors predict.

    **Time**: The rebinned HJD (and JD) is the arithmetic mean of the constituent times, representing the mid-point of the bin.

    **SNR**: Combined in quadrature: $\text{SNR}_{\text{bin}} = \sqrt{\sum_{i=1}^{n} \text{SNR}_i^2}$.

    **Check star and reference stars**: Averaged with equal weights (mean of constituent values).

    **Airmass and FWHM**: Arithmetic mean of constituent values.

*   **Outlier Status**:
    Rebinned data points are always assigned an `OK` flag regardless of whether the original constituent points were marked as `SUSPECT`. The rationale is that the averaging process itself smooths out individual outlier measurements.

*   **Non-Destructive Operation**:
    Rebinning never modifies the original photometry results. Selecting `None` in the dropdown always returns to the full-resolution light curve. Different bin sizes can be tested interactively without re-running the photometry pipeline.

*   **Exporting Rebinned Data**:
    Click **Export Rebinned CSV** in the Data Rebinning section to save the rebinned light curve to a CSV file. The exported file includes columns for `file`, `hjd`, `jd`, `mag`, `mag_err`, `check_mag`, `snr`, `airmass`, `fwhm`, `flag`, and `n_binned` (the number of original points in each bin).

---

# Chapter 4: Data Mining & Variable Star Discovery

Scans FITS file sequences to identify potential variable stars statistically, cross-matching suspects against the VSX database.

### 4.1 Statistical Star Tracking & Excess Scatter Detection
Data Mining blindly tracks all stars in a target FITS folder.
*   **Workflow Steps**:
    1. Select the FITS file folder in the manager.
    2. Under **Settings -> Camera & Photometry**, set your detector gain and read noise.
    3. Navigate to the **Data Mining** tab and click **Run Data Mining**.
    4. The pipeline performs:
       *   **Star Tracking**: Detects stars on a reference frame and tracks their sub-pixel centroids across all subsequent frames.
       *   **Ensemble Detrending**: Builds a local ensemble from stable stars to subtract systematic airmass or cloud variations.
       *   **Noise Floor Modeling**: Fits a statistical curve representing the expected physical RMS noise floor for constant stars.
       *   **Excess Scatter Check**: Identifies stars whose light curve RMS scatter lies significantly above the noise floor. Stars exceeding the threshold are flagged as **Suspect**.
*   **Steering Parameters**:
    *   `dm_detect_sigma` (*Settings -> Camera & Photometry*): Star detection threshold.
    *   `dm_suspect_sigma` (*Settings -> Camera & Photometry*): Multiplier above the noise floor to trigger a suspect flag (default `3.0`).
    *   `dm_suspect_excess` (*Settings -> Camera & Photometry*): Minimum absolute magnitude scatter (default `0.03` mag).

### 4.2 VSX Catalog Integration & Candidate Validation
Flagged suspects are checked against the online Variable Star Index (VSX).
*   **VSX Cross-Match**: Queries VizieR within a `10.0` arcsecond search radius.
    *   If matched, Calibra extracts the official variable name and type (e.g. `RRAB`, `EA`, `DSCT`).
    *   If unmatched, the star is highlighted in green as a **Potential New Discovery**.
    *   Known constant stars (`CST`) are automatically excluded from variable markings.

### 4.3 Interactivity & Light Curve Previews
*   **Interactive Controls**:
    *   **Variability Diagram**: Scatter plot of RMS vs Mean Magnitude. Clicking any star highlights its row in the suspects table; selecting a row in the table highlights the star with a yellow halo on the plot.
    *   **Preview Light Curve**: Select a suspect in the table and click **Preview Light Curve** to view a quick plot of its magnitude across all frames without leaving the tab.
    *   **Send to Light Curves**: Right-click a suspect to export its coordinates directly to the Time Series tab.

---

# Chapter 5: The Interactive FITS Viewer

The **Interactive FITS Viewer** is a graphical environment to inspect images, verify photometry apertures, select comp stars, and visualize variable star overlays. Access it by double-clicking any file in the File Manager list.

### 5.1 Layout & Visual Controls
The viewer splits into three main areas:
*   **Main Canvas**: Displays the FITS image using `ZScaleInterval` contrast stretching. Drag to pan; scroll to zoom.
*   **Left Panel**: Displays Live Inspection details, a Radial Profile plot, and VSX variable details.
*   **Right Panel**: Lists selected reference stars and houses the Aperture Settings inputs.

### 5.2 Star Role Assignment (Target, Check, Reference)
Assign roles directly to stars on the canvas:
*   **Canvas Marking**: Hover your cursor over a star and press hotkeys (or right-click to open the context menu):
    *   `V` key: Mark as **Variable Star** (drawn as a red circle).
    *   `C` key: Mark as **Check Star** (drawn as a blue circle).
    *   `R` key: Mark as **Reference Star** (drawn as a green circle).
    *   `Delete` / `Backspace` key: Remove the marker role.
*   Persistent role markers display aperture and annulus bounds.

### 5.3 Real-Time PSF Radial Profile & Aperture Customization
Clicking any star on the canvas performs an immediate Levenberg-Marquardt 2D Gaussian PSF fit.
*   **PSF Radial Profile**: The left sidebar renders the star's physical pixel intensities against the Gaussian fit curve.
*   **Aperture Boundary Visualizer**: Vertical dashed and dotted lines mark the aperture radius, inner annulus, and outer annulus. Users can visually confirm if the aperture footprint captures all stellar flux while the annulus samples clean background sky.
*   **Manual Adjustments**: Change values in the **Aperture Settings** fields (*Aperture*, *Annulus In*, *Annulus Out*) in the bottom-right panel to apply new sizes immediately.

### 5.4 VSX Variable Overlays & Dynamic Filters
Allows users to visualize catalog variables directly on the FITS image.
*   **WCS Overlay Projections**: If the image contains WCS headers, Calibra projects known VSX variables immediately onto the canvas as hollow red squares without requiring prior photometry runs.
*   **Dynamic Filtering**:
    *   **Show VSX Variables (Red Sq)** Checkbox: Toggles variable overlays on/off.
    *   **Min Amp (mag)** (*Sidebar -> Ref Panel*): Filters out variables with an amplitude below the set value (e.g. `0.2` mag) to clean up crowded fields.
    *   **Max Mag (mag)** (*Sidebar -> Ref Panel*): Filters out faint variables with a peak magnitude dimmer than the limit (default `15.0`).
*   **Non-Modal Details Popups**: Clicking near any red square (within 15 pixels) opens a non-modal popup window detailing the star's catalog info (Name, Type, Period, and Magnitudes).

### 5.5 Bidirectional Sync with Photometry Tab
*   **Import Sync**: Opening the viewer automatically pre-marks any target, check, or reference stars already configured in the **Time Series** tab.
*   **Export Sync**: Click **Export Stars to LC Tab** in the right panel to push marked coordinates back into the Time Series tab, resolving catalog names and magnitudes automatically.
*   **Aperture Sync**: Click **Export Aps to Settings** to write the viewer's current aperture parameters back to the main Settings tab.

---

# References

1.  **ASTAP (plate solving)**: ASTAP astrometric solver & photometry engine by Han Kleijn. https://www.hnsky.org/astap.html
2.  **GAIA DR3**: Gaia Collaboration, Vallenari, A., et al. (2023). "Gaia Data Release 3: Summary of the contents and survey properties". *Astronomy & Astrophysics*, 674, A1. https://doi.org/10.1051/0004-6361/202243940
3.  **Gaia Transformation Coefficients**: European Space Agency (ESA) Gaia DR3 Photometric Relations. https://gea.esac.esa.int/archive/documentation/GDR3/
4.  **APASS DR9**: Henden, A. A., et al. (2016). "AAVSO Photometric All Sky Survey (APASS) Data Release 9". *VizieR Catalog II/336*.
5.  **ATLAS-RefCat2**: Tonry, J. L., et al. (2018). "The ATLAS All-Sky Stellar Reference Catalog". *The Astrophysical Journal*, 867(2), 105. https://doi.org/10.3847/1538-4357/aae386
6.  **Landolt Catalog SA Fields**: Landolt, A. U. (1992). "UBVRI Photometric Standard Stars in the Magnitude Range 11.5 < V < 16.0 Around the Celestial Equator". *The Astronomical Journal*, 104, 340. https://doi.org/10.1086/116242

---

# Index

### A
*   AAVSO Chart ID: [Section 3.1]
*   AAVSO Observer Code: [Section 3.1]
*   Airmass ($X$): [Section 2.2], [Section 3.3]
*   Annulus Inner / Outer: [Section 2.1], [Section 5.3]
*   Aperture Photometry: [Section 2.1], [Section 3.3]
*   Aperture Radius: [Section 2.1], [Section 5.3]
*   ASTAP Path: [Section 1.2]
*   ATLAS refcat2: [Section 2.1]

### B
*   Bayer Pattern: [Section 1.3]
*   Bias Calibration: [Section 1.1]
*   Bilinear Interpolation: [Section 1.3]
*   B-V Color Index: [Section 2.2], [Section 3.1]

### C
*   Check Star: [Section 3.1], [Section 5.2]
*   Centroid Tracking: [Section 2.1], [Section 4.1]
*   Color Term Transformation: [Section 2.2], [Section 3.3]
*   Constant Star (CST): [Section 2.3], [Section 4.2]

### D
*   DAOStarFinder: [Section 2.1], [Section 4.1]
*   Data Mining: [Section 4]
*   Detection Sigma: [Section 2.1]
*   Differential Photometry: [Section 2.3], [Section 3.4]
*   Dynamic Aperture: [Section 2.1], [Section 5.3]

### E
*   Ensemble Comparison Stars: [Section 3.2]
*   Ensemble Detrending: [Section 4.1]
*   Extinction Coefficient ($k$): [Section 2.2], [Section 3.3]

### F
*   FITS Viewer Layout: [Section 5.1]
*   Flat-Field Calibration: [Section 1.1]
*   Flexible Aperture: [Section 2.1], [Section 5.3]
*   FWHM: [Section 2.1], [Section 5.3]

### G
*   Gain (CCD/CMOS): [Section 2.1]
*   Gaussian PSF Fit: [Section 2.1], [Section 5.3]

### H
*   Heliocentric Julian Date (HJD): [Section 3.3]
*   Hotkeys (Viewer): [Section 5.2]

### I
*   Instrumental Magnitude: [Section 2.1]
*   Interactive Plot Customizer: [Section 3.5]
*   Inverse-Variance Weighted Mean: [Section 3.7]

### L
*   Light Curve: [Section 3]

### M
*   Master Catalog: [Section 4.1]
*   Max Mag Filter: [Section 5.4]
*   Min Amp Filter: [Section 5.4]

### N
*   Noise Floor Model: [Section 4.1]
*   Noisy Star Flag: [Section 4.1]

### O
*   OSC Bayer demultiplexing: [Section 1.3]
*   Outlier Flagging: [Section 3.3], [Section 4.1]

### P
*   Plate Solving: [Section 1.2]
*   Poisson CCD Noise: [Section 2.1]
*   PSF Profile: [Section 2.1], [Section 5.3]

### R
*   Read Noise: [Section 2.1]
*   Rebinning (Data): [Section 3.7]
*   Reference Selection: [Section 2.3]

### S
*   Saturation Limit: [Section 2.1]
*   SNR Threshold: [Section 2.1]
*   Superpixel Splitting: [Section 1.3]
*   Suspect Star Flag: [Section 4.1]

### T
*   Target Star: [Section 3.1], [Section 5.2]
*   Transformation Coefficients: [Section 2.2], [Section 3.3]

### V
*   Variable Star Markings: [Section 5.4]
*   VSX catalog overlay: [Section 4.2], [Section 5.4]

### W
*   WCS coordinates: [Section 1.2], [Section 5.4]

### Z
*   Zero Point ($ZP$): [Section 2.1], [Section 3.3]
*   ZScaleInterval: [Section 5.1]
