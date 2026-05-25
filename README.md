# Calibra: Automated Photometric Analysis & Calibration Toolkit

**Version:** 4.12 — 2026-05-25
**Description:** 
An automated, highly robust Python toolkit for extracting scientific-grade photometry, calibrating zero points, obtaining color transformation coefficients for V and B filters, and estimating formal CCD/CMOS errors from astronomical monochrome FITS images. Calibra features **Ensemble Differential Photometry** with multiple comparison stars, **Time-Series Light Curve** generation with AAVSO-format reporting, an **Interactive FITS Viewer** with role-based star marking, real-time PSF analysis, and bidirectional synchronization with the photometry pipeline. It integrates **AAVSO VSX** cross-matching to automatically exclude known variable stars from calibration sets.

The code was created with lots of help from Google Antigravity (various agents), guided, tested and debugged by me. I have made several tests to verify the results and ensure the results are reasonable. This code is far from what codes like, e.g., AIJ, ASTAP or Tycho Tracker can do. The purpose is simply to have a playground for understanding the principles of CCD/CMOS based photometry and having a tool to compare to what AIJ, ASTAP or Tycho Tracker provide as fluxes, zero points, etc. However, I have used the code to produce reports for AAVSO (time-series) with my observations as I am quite happy with the results so far.

---

## What's New in v4.12

- **Data Mining Centroid Accuracy**: Implemented a hard distance threshold (max 4.0 pixel shift) for the 2D Gaussian centroiding algorithm across both Data Mining and Time-Series modules. This effectively prevents the tracking pipeline from incorrectly "snapping" to nearby bright neighbor stars when analyzing very faint targets, greatly reducing the false-positive "noisy" star count.
- **Interactive Scatter Plot Context Menus**: Enhanced the Stellar Variability scatter plot. Users can now right-click any star directly on the plot to summon the context menu (with options to "Send to Light Curves" or "Open in FITS Viewer"), fully supporting hidden non-suspect stars.
- **Advanced Light Curve Customization**: The Time-Series module now features interactive inline visibility toggles and a dedicated "Advanced Plot Customizer" window. Users can dynamically toggle the target star, check star, and individual reference stars, set custom plot titles, axis labels, manual Y-axis limits, edit legend labels, and change colors/marker symbols. The plot redraws instantly, and customized plots can be exported to high-resolution images.
- **Reference Star Statistics**: The Time-Series Analysis Summary now includes a scrollable detailed breakdown of statistical metrics (bias, standard deviation, and precision) for every individual reference star in the ensemble, alongside the check star.
- **VSX Name Resolution Fallback**: If standard name resolution (Simbad) fails for a Target or Reference Star, Calibra now automatically falls back to searching the AAVSO Variable Star Index (VSX) database by name to resolve coordinates (e.g., for `NSVS 4892894`).
- **Data Mining Context Integration**: Sent suspects from the Data Mining module to the Time-Series tab now automatically extract their official VSX names or Catalog IDs, resorting to `DM_xxxx` only if entirely unknown.
- **Context Viewer Configuration Inheritance**: Fixed a bug where launching the FITS Viewer from the Suspects context menu failed to inherit GUI settings, causing the viewer to fallback to hardcoded default aperture sizes (8/15/20).
- **Robust Time-Series Outlier Rejection**: Replaced the static global 3-sigma rejection algorithm with a Rolling Median and MAD-based robust outlier detector. This correctly preserves true stellar variability (like large amplitude eclipses) while still flagging sudden anomalous spikes.
- **Interactive Flag Toggling**: Added a right-click context menu to the Time-Series Numerical Results table, empowering users to manually toggle the `SUSPECT` / `OK` status of any data point, instantly updating the plot.
- **AAVSO Sequence Persistence & Viewer Toggles**: Fixed a session loading bug where AUIDs were overwritten by Gaia IDs upon restart by fully persisting the AAVSO Sequence Cache. Added interactive visibility toggles in the FITS Viewer for both AAVSO Sequence Stars and User-Selected Stars (Target/Ref/Check) to allow an unobstructed view of raw image data.
---

## What's New in v4.1

- **Data Mining Robustness**: Relaxed the VSX cross-match radius from 5 arcsec to 10 arcsec and added adaptive bounding box calculations to prevent `Max stars` warnings and WCS boundary clipping issues. Implemented advanced outlier rejection using Median Absolute Deviation (MAD) to correctly classify stars with extreme frame-to-frame magnitude jumps as `NOISY` rather than false-positive variables.
- **Interactive Data Mining UI**: Enhanced the Data Mining tab with an interactive variability diagram (clicking points highlights rows, selecting rows draws a yellow halo), a new "Valid Frames" column, and a fast "Preview Light Curve" pop-up for rapid candidate review.
- **Interactive FITS Header Editor**: Added a new graphical editor accessible via the File Manager to inspect, modify, add, or delete FITS header keywords and values, providing a quick way to fix missing data like filters or exposure times.
  
---

## What's New in v4.0

- **Data Mining & Variable Star Discovery**: Added an automated data mining tab to scan FITS sequences. Combines DAOStarFinder star tracking, global ensemble detrending, and statistical excess RMS scatter calculations (Poisson noise floor modeling) to identify candidate variable stars. Automatically cross-matches candidates against the AAVSO International Variable Star Index (VSX) to label known variable types and highlight new discoveries.
- **One-Shot Color (OSC) Demultiplexing**: Added a dedicated pre-processing tab to split raw color Bayer FITS files into Red, Green, and Blue channels. Supports `Superpixel` (2x2 -> 1px, preserving physical flux with automatic WCS scaling) and `Bilinear` interpolation methods. Automatically imports split frames back into the File Manager.
- **Interactive FITS Viewer Keyboard Shortcuts**: Hover the mouse cursor over a star in the FITS Viewer and press single-key hotkeys to instantly assign stellar roles (`V` for Variable, `R` for Reference, `C` for Check) or press `Delete`/`Backspace` to remove a marker, bypassing right-click menus entirely for lightning-fast sequencing.
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
*   **FITS Pre-processing & OSC Split**: Plate solving via ASTAP, flat/bias calibration, and demultiplexing raw One-Shot Color Bayer FITS files (RGGB, BGGR, etc.) via Superpixel extraction with scaled WCS parameters.
*   **The Processing Pipeline**: From Star Detection to Shift Analysis and color calibration.
*   **Differential Photometry**: Computing formal AAVSO-ready magnitudes ($B$, $V$, $B-V$) using reference stars.
*   **Time-Series & Light Curves**: Ensemble comparison star photometry with formal uncertainty propagation.
*   **Data Mining**: Statistical excess scatter calculations, global ensemble detrending, and VizieR AAVSO VSX cross-matching.
*   **GUI Guide**: The v4.0 Interactive FITS Viewer, FITS File Manager, sidebar tabs, and Settings configuration.

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
