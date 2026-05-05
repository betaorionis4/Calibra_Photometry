# Calibra: Automated Photometric Analysis & Calibration Toolkit

**Version:** 1.4 - 2026-05-05
**Description:** 
An automated, highly robust Python toolkit for extracting scientific-grade photometry, calibrating zero points, obtaining color transformations coefficients for V and B filters, and estimating formal CCD/CMOS errors from astronomical monochrome FITS images. The toolkit features a **Differential Photometry** module for computing standard magnitudes and integrates **AAVSO VSX** cross-matching to automatically exclude known variable stars from calibration sets.

The code was created with lots of help from Google Antigravity (various agents), guided, tested and debugged by me. I have made several tests to verify the results and ensure the results are reasonable. This code is far from what codes like, e.g., AIJ or Tycho Tracker can do. The purpose is simply to have a playground for understanding the principles of CCD/CMOS based photometry and having a tool to compare to what AIJ or Tycho Tracker provide as fluxes, zero points, etc.

---

## Quick Start
1.  **Installation**:
    ```bash
    pip install numpy scipy matplotlib Pillow astropy photutils astroquery
    ```
2.  **Run**:
    ```bash
    python main.py
    ```
    This launches the **Configuration GUI** where you can set your CCD parameters and file paths. The pipeline supports **Automated Color Calibration** and **Differential Photometry** via ATLAS refcat2, APASS DR9, GAIA DR3 and Landolt standard stars.

---

## Documentation
For a more detailed dive into the mathematical principles, theoretical background, and stage-by-stage processing details, please refer to the comprehensive **User Manual**:

👉 **[photometry_user_manual.md](file:///c:/Astro/StarID/photometry_user_manual.md)**

### Key Manual Sections:
*   **Theory of Operation**: Aperture Photometry vs. Sky Annulus math.
*   **Mathematical Principles**: PSF Fitting, Sub-pixel Refinement, and Error Propagation.
*   **The 7 Processing Stages**: From Star Detection to Shift Analysis and color calibration.
*   **Differential Photometry**: Computing formal AAVSO-ready magnitudes ($B$, $V$, $B-V$) using reference stars. Supports automated, manual (coordinate-based), and name-based (SIMBAD resolution) reference star selection.
*   **Accuracy Evaluation**: Statistical fitting and plotting of deviations against catalog magnitudes.
*   **GUI Guide**: How to tune the pipeline for your specific sensor.
*   **Diagnostics**: Understanding radial profiles and calibration reports.

---

## Directory Structure
- `main.py`: The main script to run the pipeline.
- `photometry/`: Python modules for photometry and calibration.
- `photometry_refstars/`: Reference catalogs (e.g., `reference_stars.csv`).
- `photometry_output/`: Auto-generated results, logfiles and reports.
- `photometry_plots/`: Optional diagnostic PSF plots.
- `fitsfiles/`: Input `.fits` images.
