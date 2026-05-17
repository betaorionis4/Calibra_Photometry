# Calibra: Automated Photometric Analysis & Calibration Toolkit

**Version:** 4.0 — 2026-05-17
**Description:** 
An automated, highly robust Python toolkit for extracting scientific-grade photometry, calibrating zero points, obtaining color transformation coefficients for V and B filters, and estimating formal CCD/CMOS errors from astronomical monochrome FITS images. Calibra features **Ensemble Differential Photometry** with multiple comparison stars, **Time-Series Light Curve** generation with AAVSO-format reporting, an **Interactive FITS Viewer** with role-based star marking, real-time PSF analysis, and bidirectional synchronization with the photometry pipeline. It integrates **AAVSO VSX** cross-matching to automatically exclude known variable stars from calibration sets.

The code was created with lots of help from Google Antigravity (various agents), guided, tested and debugged by me. I have made several tests to verify the results and ensure the results are reasonable. This code is far from what codes like, e.g., AIJ, ASTAP or Tycho Tracker can do. The purpose is simply to have a playground for understanding the principles of CCD/CMOS based photometry and having a tool to compare to what AIJ, ASTAP or Tycho Tracker provide as fluxes, zero points, etc. However, I have used the code to produce reports for AAVSO (time-series) with my observations as I am quite happy with the results so far.

---

## What's New in v4.0

- **Integrated FITS Viewer Launchers**: Added `Open Selected FITS in Viewer` buttons in both the main File Manager toolbar and the Light Curves analysis panel. Allows users to instantly open any selected image, view stellar profiles, and synchronize targets/check stars bidirectionally with a single click.
- **Educational Quick-Help Modals**: Added `❓ What does this do?` popup helpers next to all execution buttons in FITS Calibration, ASTAP Plate Solving, and Time-Series Photometry tabs, offering detailed guides on theoretical principles.

### Previous: v3.2 (2026-05-17)

- **AAVSO Extended Format Compliance**: Fully standardized light curve reports meeting strict AAVSO requirements (GROUP, CHART, precision limits, and COMMENTS headers).
- **Persistent Settings Integration**: AAVSO Observer Code and Observer Name are now persisted globally and synchronized across the software.
- **AUID & Alias Awareness**: The Light Curve tab and viewer display official AAVSO aliases (AUID) while calculating true fluxes from the underlying catalog magnitudes.
- **Flexible Color Transformations**: Explicit on/off switches for transformations with automatic #TRANS flag handling.

### Previous: v3.1 (2026-05-13)

- **Interactive FITS Viewer — Role-Based Star Marking**: Right-click any star to assign it as **Variable** (red), **Check** (blue), or **Reference** (green). Markers are persistent with aperture/annulus overlays and are listed in dedicated side panels.
- **Real-Time Radial Profile Analysis**: Clicking any star instantly displays a PSF radial profile plot with Gaussian fit overlay, aperture boundary, and annulus boundary — enabling immediate visual verification of photometry settings.
- **In-Viewer Aperture Tuning**: Aperture radius, annulus inner, and annulus outer values are displayed and editable directly inside the viewer. Changes take effect immediately for subsequent markings.
- **Bidirectional Star Synchronization**: Stars defined in the Light Curves tab (target, check, references) are automatically pre-marked when opening the viewer. Conversely, clicking "Export Stars to LC Tab" pushes viewer selections back to the main GUI, automatically triggering catalog magnitude lookups.
- **Bidirectional Aperture Synchronization**: Click "Export Aps to Settings" to push refined aperture/annulus values from the viewer back to the main Settings tab.
- **Darkened FITS Display**: The image rendering now floors the display range to the median, providing high-contrast star visibility against a dark background.

### Previous: v3.0 (2026-05-10)

- Centralized File Management with persistent FITS File Manager.
- Unified Analysis & Calibration workflow tab.
- Plate Solving integration via ASTAP.
- Modern deep-blue professional theme with scrollable interfaces.
- Ensemble Photometry scaling for time-series analysis.

---

## Quick Start
1.  **Installation**:
    ```bash
    pip install numpy scipy matplotlib Pillow astropy photutils astroquery
    ```
2.  **Run**:
    ```bash
    python calibra.py
    ```
    This launches the **Configuration GUI** where you can set your CCD parameters and file paths. The pipeline supports **Automated Color Calibration**, **Differential Photometry**, and **Time-Series Light Curves** via ATLAS refcat2, APASS DR9, GAIA DR3 and Landolt standard stars.

---

## Documentation
For a more detailed dive into the mathematical principles, theoretical background, and stage-by-stage processing details, please refer to the comprehensive **User Manual**:

👉 **[photometry_user_manual.md](photometry_user_manual.md)**

### Key Manual Sections:
*   **Theory of Operation**: Aperture Photometry vs. Sky Annulus math.
*   **Mathematical Principles**: PSF Fitting, Sub-pixel Refinement, and Error Propagation.
*   **The Processing Pipeline**: From Star Detection to Shift Analysis and color calibration.
*   **Differential Photometry**: Computing formal AAVSO-ready magnitudes ($B$, $V$, $B-V$) using reference stars.
*   **Time-Series & Light Curves**: Ensemble comparison star photometry with formal uncertainty propagation.
*   **GUI Guide**: The v4.0 Interactive FITS Viewer, FITS File Manager, unified Analysis tab layout, and Plate Solving.

---

## Directory Structure
- `calibra.py`: The main entry point to launch the GUI.
- `gui.py`: The v4.0 Configuration GUI (tkinter).
- `photometry/fits_viewer.py`: The interactive FITS Viewer with role-based marking, radial profiles, and aperture controls.
- `photometry/`: Python modules for photometry, calibration, color terms, differential, time series, and plate solving.
- `photometry_refstars/`: Reference catalogs and online query cache.
- `photometry_output/`: Auto-generated results, logfiles, reports, and light curves.
- `photometry_plots/`: Optional diagnostic PSF plots.
- `calibra_session.json`: Persistent session settings (auto-generated).
