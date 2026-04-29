# StarID Astronomical Photometry Pipeline

**Version:** 1.0  
**Description:** An automated, highly robust Python pipeline for extracting scientific-grade photometry, calibrating zero points, and estimating formal CCD errors from astronomical FITS images.

---

## Quick Start
1.  **Installation**:
    ```bash
    pip install numpy matplotlib astropy photutils
    ```
2.  **Run**:
    ```bash
    python main.py
    ```
    This launches the **Configuration GUI** where you can set your CCD parameters and file paths.

---

## Documentation
For a deep dive into the mathematical principles, theoretical background, and stage-by-stage processing details, please refer to the comprehensive **User Manual**:

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
