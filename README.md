# StarID Astronomical Photometry Pipeline

**Version:** 1.1  
**Description:** 
An automated, highly robust Python pipeline for extracting scientific-grade photometry, calibrating zero points, and estimating formal CCD/CMOS errors from astronomical FITS images. The code was created with lots of help from Google Antigravity (various agents), guided by me. I have made several tests to verify the results and ensure the results are reasonable. 
This code is far from what codes like, e.g., AIJ or Tycho Tracker can do. 
The purpose is simply to have a playground for understanding the principles of CCD/CMOS based photometry and having a tool to compare to what AIJ or Tycho Tracker provide as fluxes, zero points, etc, 

---

## Quick Start
1.  **Installation**:
    ```bash
    pip install numpy matplotlib astropy photutils astroquery
    ```
2.  **Run**:
    ```bash
    python main.py
    ```
    This launches the **Configuration GUI** where you can set your CCD/CMOS parameters and file paths. The pipeline now supports **Automated Online Calibration** via ATLAS RefCat2 and APASS DR9.

---

## Documentation
For a dive into the mathematical principles, theoretical background, and stage-by-stage processing details, please refer to the comprehensive **User Manual**:

👉 **[photometry_user_manual.md](file:///c:/Astro/StarID/photometry_user_manual.md)**

### Key Manual Sections:
*   **Theory of Operation**: Aperture Photometry vs. Sky Annulus math.
*   **Mathematical Principles**: PSF Fitting, Sub-pixel Refinement, and Error Propagation.
*   **The 6 Processing Stages**: From Star Detection to Shift Analysis.
*   **GUI Guide**: How to tune the pipeline for your specific sensor.
*   **Diagnostics**: Understanding radial profiles and calibration reports.

---

## Directory Structure
- `fitsfiles/`: Input `.fits` images.
- `photometry_refstars/`: Reference catalogs (e.g., `reference_stars.csv`).
- `photometry_output/`: Auto-generated results and reports.
- `photometry_plots/`: Optional diagnostic PSF plots.
