# Calibra: Automated Photometric Analysis & Calibration Toolkit

**Version:** 4.44 — 2026-11-14  
**Developed by:** Stephan Pomp, Google DeepMind Antigravity and Claude (Anthropic)  
**Description:** 
An automated, highly robust Python toolkit for extracting scientific-grade photometry, calibrating zero points, obtaining color transformation coefficients for V and B filters, and estimating formal CCD/CMOS errors from astronomical monochrome FITS images. Calibra features **Ensemble Differential Photometry** with multiple comparison stars, **Time-Series Light Curve** generation with AAVSO-format reporting, an **Interactive FITS Viewer** with role-based star marking, real-time PSF analysis, and bidirectional synchronization with the photometry pipeline. It integrates **AAVSO VSX** cross-matching to automatically exclude known variable stars from calibration sets.

The code was created with lots of help from Google Antigravity (various agents) and, from v4.14 on, Claude (Anthropic), guided, tested and debugged by me. I have made several tests to verify the results and ensure the results are reasonable. This code is far from what codes like, e.g., AIJ, ASTAP or Tycho Tracker can do. The purpose is simply to have a playground for understanding the principles of CCD/CMOS based photometry and having a tool to compare to what AIJ, ASTAP or Tycho Tracker provide as fluxes, zero points, etc. However, I have used the code to produce reports for AAVSO (time-series) with my observations as I am quite happy with the results so far.

---

## What's New in v4.44 (2026-11-14) — faint targets

- **Faint Target** option in Light Curves: forced photometry at the fixed sky position (WCS corrected by the reference stars), same aperture in every frame; frames combined afterwards in groups of X or by marking rows in the table; upper limits below a chosen S/N. Keeps the time resolution of short exposures while reaching fainter magnitudes (e.g. HU Aqr in minimum).

---

## What's New in v4.38 (2026-11-08) — night plan for NINA

- **Night Plan for NINA…**: from the planner's targets Calibra chooses exposure times (no saturation, enough S/N), durations (eclipse times, one cycle for short periods, snapshots for long periods) and the order, and writes a NINA target set for the simple sequencer.

---

## What's New in v4.37 (2026-11-08) — observation planner

- **Observation Planner**: which variables can be observed tonight from the site of an instrument setup - from the local GCVS list, your priority list or a list of open clusters, with time window, altitude, azimuth sector, Moon, magnitude, type and period filters, altitude chart. Works offline.

---

## What's New in v4.36 (2026-11-08) — instrument setups

- **Instrument setups** for each telescope / camera / site (e.g. refractor, MWT, Vespera 2): instrument data, site, filters, and all instrument-dependent settings; switch in the sidebar, Calibra recognises which setup loaded frames belong to.

---

## What's New in v4.35 (2026-11-08) — AAVSO sequences stored

- AAVSO comparison-star sequences are **stored per variable** and reused without asking AAVSO again; *Refresh from AAVSO* gets a new one when wanted.

---

## What's New in v4.34 (2026-11-07) — fast FITS Viewer

- The **FITS Viewer opens at once**; catalogue data are loaded in the background, from the local copies where possible (no per-star online queries), and zooming is much faster.
- Servers that do not answer are skipped for 10 minutes instead of blocking Calibra again and again.

---

## What's New in v4.32 (2026-11-07) — APASS DR10 download

- **Download APASS DR10** for the checked images, a star name or coordinates directly from Calibra, or **automatically** for fields that are not yet local (option, default off). Each field is downloaded once and used offline afterwards.

---

## What's New in v4.31 (2026-11-07) — APASS DR10

- CSV files from the **AAVSO APASS DR10 download page** can be saved unchanged in the local APASS folder and are used for comparison stars (manual 3.6a).

---

## What's New in v4.30 (2026-11-07) — star names offline

- **Star names (e.g. `HU Aqr`) are looked up in the local GCVS list first**, then online (SIMBAD, AAVSO VSX) - name resolution works without internet for all GCVS stars.

---

## What's New in v4.29 (2026-11-07) — bias and flat folders

- **Give a bias folder and a flat folder** (Pre-processing → Master Frames): every FITS file in them is used - no file names or header types needed. The masters are written into the same folders; the flats are bias-corrected when they are built, so nothing is subtracted twice.

---

## What's New in v4.27 (2026-11-07) — calibration frames

- **Master bias / flats (and optional darks) built by Calibra** from a folder of calibration frames; stored next to them and reused when nothing changed.
- **Automatic choice of the matching masters** per light frame (filter V/B/L/other, size, binning, gain, offset).
- **Calibration on the fly** while measuring (no extra files) or as `cal_` files next to the raw frames, created only once. L frames are now calibrated too.

---

## What's New in v4.26 (2026-11-07) — local catalogues

- **Local copies of ATLAS-RefCat2, Gaia DR3, APASS, VSX and a GCVS list**: select the folder that contains them once (Settings → Astro & Catalog); Calibra finds the subfolders, remembers them, and goes online only for what is not available locally.
- **Field Variables** adds the variables of the local GCVS list (e.g. `GCVS_selected.csv`) in the field of the checked images automatically.

---

## What's New in v4.25 (2026-11-07) — phase plots

- Period Analysis: plot selector (phase plot / residuals / O–C) directly above the plots; selecting a frequency returns to the phase plot.
- **Phase Plot Window**: edit title, labels, ranges, symbols, curve and phase zero (maximum light), then save as PNG/PDF/SVG or the folded data as CSV.

---

## What's New in v4.24 (2026-11-06) — periods in depth

- **Period Analysis**: fit-quality check (residual rms vs. photometric errors, χ², night offsets), residuals-vs-time plot, **times of maximum and O–C diagram** (corrected period, period change), period-ratio hints for double-mode pulsators.
- **[period_analysis_guide.md](period_analysis_guide.md)**: a practical guide to measuring periods and finding additional ones (worked HADS example, checklists, other variable types). Practice file `example_HADS_simulated.csv` (simulated) in the data folder.

---

## What's New in v4.23 (2026-11-06) — L filter, GCVS lists, phase plots

- **L (luminance) light curves** in Light Curves and Field Variables (calibrated magnitudes on the V scale, AAVSO filter code CV; relative flux / raw ADU as usual).
- **Target lists exported from GCVS searches** (e.g. SAAF variable-star section, Swedish column names) are read directly; name, RA and Dec are required - Calibra says what is missing.
- **Period Analysis**: the curve in the phase plot can be the fit, a model-free binned mean, or switched off; harmonics of the selected frequency stay in its phase plot (true shape, e.g. sawtooth).
- Fixed: AAVSO comparison-star import used V magnitudes for B light curves. Details: [CHANGELOG.md](CHANGELOG.md).

---

## What's New in v4.22 (2026-11-05) — plate-solving guess

- **Plate solving with a position guess** for images without coordinates in the header: object name (looked up online) or RA/Dec typed in. In a series, each solved image is the guess for the next one; blind solving only as the last resort (optional). Details: [CHANGELOG.md](CHANGELOG.md), manual section 3.1.

---

## What's New in v4.21 (2026-11-05) — Period Analysis

- **New tab *Period Analysis***: Lomb–Scargle (amplitude spectrum, harmonics), PDM and BLS periodograms for one or several light curves (e.g. several nights or seasons); **spectral window** to recognise aliases; **prewhitening** to find further periods on top of a main period, with a joint non-linear fit, uncertainties, S/N and notes on harmonics/combinations/aliases; phase plots; results saved as CSV and PNG.
- Light curves come from files, from the last Light Curves run or from a Field Variables target. Details: [CHANGELOG.md](CHANGELOG.md), manual section 7b.

---

## What's New in v4.20 (2026-11-05) — several nights

- **Field-of-view check against every image**: targets, comparison and check stars are checked on all selected images (different pointings on different nights). Field Variables shows on how many images each target lies and skips targets on none; the Light Curves tab warns before running.
- Automatic comparison stars are chosen among stars present on all images of the target; relative flux stays consistent when a comparison star is missing on some images.
- Output for data from several nights is named `first_to_last` date. Details: [CHANGELOG.md](CHANGELOG.md).

---

## What's New in v4.19 (2026-11-05) — Field Variables

- **New tab *Field Variables***: light curves for many known variables in the field in one run - e.g. long-period variables over many nights. Targets from a **list file** (name + RA/Dec) or from an **online query of AAVSO VSX (incl. GCVS) or the GCVS** for the image field, with filters for brightness, amplitude, period and type.
- **Comparison and check stars are chosen automatically** for each target from the reference catalogue → calibrated magnitudes (or relative flux / raw ADU) for all targets; each image is read only once.
- Output per target (CSV + plot), a **summary table** with variability statistics and an **overview plot** with all light curves; any star can be sent to the Light Curves tab for a detailed analysis.
- **`.fts` files** are accepted by the File Manager.
- Fixed: catalogue search radius error from v4.18. Details: [CHANGELOG.md](CHANGELOG.md).

---

## What's New in v4.18 (2026-11-04) — user-friendliness

- **Light Curves tab**: fixed action bar at the top (Generate, Viewer, Save/Open Setup, Open Output Folder, progress, status); the coefficients/AAVSO section can be collapsed.
- **Named setups** per target (`setups/<target>.json`): target, comparison/check stars, mode, coefficients, chart ID, AAVSO sequence.
- **Dated output files** (UT date of the first observation) — different nights no longer overwrite each other.
- **Data Mining → Light Curves with automatic comparison and check stars** (constant stars of similar brightness close to the target; Relative-flux mode).
- **One place for each setting**: extinction and observing site from Settings, one detection FWHM; numbers in entry fields rounded for display.
- Open ideas are listed in [TODO.md](TODO.md). Details: [CHANGELOG.md](CHANGELOG.md).

---

## What's New in v4.17 (2026-11-04) — speed

- **Data Mining ~100× faster per star** (69 → 0.7 ms per star and frame on a 26-Mpixel frame; 1871 stars in 1.3 s per frame) and **Light Curves ~2× faster**: each frame is converted to float once (photutils otherwise converted the whole image for every star), stars are measured in one vectorised call, and a fast windowed centroid replaces the 2-D Gaussian fit (same accuracy). New module `photometry/fastphot.py`.
- **Data Mining remembers its measurements** (`photometry_output/dm_cache/`): re-running the analysis with other thresholds takes seconds instead of re-reading all frames. Options *Reuse stored measurements* and *Clear Measurement Cache* in the Data Mining tab.
- Results unchanged within noise (light curve AE UMa: 11.3281 → 11.3280). Details: [CHANGELOG.md](CHANGELOG.md).

---

## What's New in v4.16 (2026-11-04) — science revision (code review step 3)

These changes **alter numerical results** (on purpose). Compare with v4.15 using `backup_v4.15/`; details and measured effects in [CHANGELOG.md](CHANGELOG.md).

- **Corrected noise model** (`photometry/noise.py`, used everywhere): Poisson term from the net flux, effective gain = gain × STACKCNT for averaged stacks, sky term from the clipped annulus, optional correlated-noise *Sky noise factor*. Each pipeline run measures that factor on empty sky (AE UMa stacks: 1.40 V, 1.33 B). Errors on the 49-frame V stack were ~5× too large before.
- **Colour transformation**: only well-measured, unsaturated, unblended stars; weighted robust fits; coefficient errors; consistency check Tbv = 1/(1 − Tb_bv + Tv_bv); residuals vs magnitude. AE UMa field: Tbv 0.970 ± 0.011, Tb_bv 0.014 ± 0.008, Tv_bv 0.033 ± 0.007 (v4.15: Tb_bv 0.149, inconsistent).
- **Differential photometry**: ensemble zero points from all suitable comparison stars (V offset vs ATLAS +0.083 → +0.011 mag), B = V + (B−V) (consistent colours), full error propagation (B_err, V_err, B_V_err), optional check star.
- **Zero-point calibration**: ZP error = error of the median; peak limit (default 30 000 ADU) and blend exclusion for calibration stars; colour-dependence diagnostic; automatic catalogue radius (0 = image half-diagonal).
- **Quality flags** in the results CSV (peak_raw, saturated, blended, sky statistics); edge stars skipped.
- **Data Mining**: per-star noise ratio (RMS / own photometric error), quadrature excess, Fast Test Mode off by default, thread-safe status updates, plot crash with very few stars fixed.
- **FITS Viewer**: same aperture photometry as the pipeline, VSX search over the whole frame, float32 image.
- **OSC split**: consistent WCS (old CD/SIP removed), float data preserved, BAYERPAT check.
- **Detection FWHM** setting; **test suite** `tests/test_calibra.py` (`python -m pytest tests -q`).

---

## What's New in v4.15 (2026-11-04) — bug-fix release

Fixes from a full code review (`Calibra_code_review_2026-11-04.md`, steps 1 and 2 of its fix plan). The photometric methods are unchanged; the only effect on numbers is that more known variables are now excluded from the zero-point stars (on the AE UMa V stack: ZP 24.037 → 24.036). See [CHANGELOG.md](CHANGELOG.md) for every item.

- **One B/V filter rule everywhere** (`photometry/filters.py`): "Bessell V"/"Baader V" are no longer treated as B, and a B zero point can no longer be stored as the V default.
- **FITS Viewer no longer overwrites the Light Curves setup on close** (names → "Target", magnitudes → NaN, apertures). Export only on request; existing names/magnitudes are kept.
- **Bias/flat calibration**: output named after the input file (stacks were silently left uncalibrated), saved as 32-bit float (no clipping), optional bias subtraction from the flat, clear errors on size mismatch.
- **File Manager**: ticks survive refreshes and sessions; "Remove Highlighted" no longer empties the list.
- **Errors are visible**: error dialogs in background tasks work again, invalid/empty number fields give a clear message, pipeline errors are reported and one bad file no longer stops a batch.
- **Header editor**: automatic `.bak` backup, value types preserved, COMMENT/HISTORY editable.
- **Also fixed**: RA/Dec region filter, plate-solve false successes (`.FIT`, stale `.wcs`, timeout), Data Mining frame misalignment, duplicate "Reset Ensemble", VSX distance cos δ, ATLAS rows without g or r, shift-analysis search radius, PSF FWHM sign, report table, thread-safe console output.

---

## What's New in v4.14 (2026-11-04)

- **Light Curve Modes**: A new *Light Curve Mode* box in the Light Curves tab offers three ways to build a light curve:
  - **Calibrated magnitudes** (default, unchanged): ensemble differential photometry with catalogue magnitudes; AAVSO report.
  - **Relative flux**: uses only the *measured ADU* of the reference stars — catalogue magnitudes, B-V, colour term and extinction are ignored. Result: $F_\text{target}/\sum F_\text{ref}$ with propagated errors (also as differential instrumental magnitude). Each reference star is shown relative to the sum of the others as a stability diagnostic.
  - **Raw ADU**: the target's background-subtracted net flux in ADU (optionally ADU/s) with CCD-equation uncertainties. **No reference or check stars needed.**
- **Automatic Raw ADU**: If no reference star is selected, Calibra switches to Raw ADU automatically (checkbox, on by default).
- **Flux or magnitude plots** for the uncalibrated modes (checkbox), with correct axis labels/orientation in both the inline preview and the Customize Plot window. The results table and Analysis Summary adapt to the mode (flux columns; scatter in % and mag).
- **Separate output files** (`*_rawADU.csv/.png`, `*_relflux.csv/.png`) so uncalibrated runs never overwrite calibrated results.
- **Fixes**: the light-curve plot PNG is now actually saved (previously the preview wrote a stray `.png` file into the working folder instead); HJD is now computed correctly on cached re-runs of FITS files without `SITELAT`/`SITELONG` keywords (previously fell back to JD silently).

See [CHANGELOG.md](CHANGELOG.md) for details and Section 6.6 of the user manual for the mathematics.

---

## What's New in v4.13 (2026-05-28)

- **VSX Magnitude Limit Filter**: Added a `Max Mag (mag)` spinbox filter in the FITS Viewer sidebar (defaulting to `15.0`) to dynamically hide/show variable stars based on their peak magnitude limit.
- **VSX Amplitude Filter**: Added a `Min Amp (mag)` spinbox filter to show only variables with a variability amplitude above a specified threshold (e.g. `0.2` mag), decluttering the FITS canvas.
- **Immediate WCS Projection in FITS Viewer**: Variable stars are now plotted immediately using their WCS projected coordinates upon loading a FITS image (without requiring previous pipeline photometry results). They automatically refine to centroid coordinates if matching pipeline detections are present.

---

## What's New in v4.12 (2026-05-25)

- **Configurable Centroid Tracking**: Added user-configurable settings under a new "Centroid Tracking" block in the Settings tab, allowing customization of the Initial Search Radius (default 15.0 px) and Max Tracking Shift (default 4.0 px) to handle wide fields or distorted optics at FOV boundaries.
- **AAVSO Chart ID & Check Star Magnitudes**: Added an "AAVSO Chart ID" input field in the Light Curve tab to include the CHART keyword (Sequence ID) in AAVSO reports. Fixed the check star magnitude in AAVSO reports (`KMAG`) to output the dynamic frame-by-frame measured magnitude instead of repeating a static catalog magnitude.
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
*   **Time-Series & Light Curves**: Ensemble comparison star photometry with formal uncertainty propagation; Calibrated, Relative-flux and Raw-ADU light-curve modes (v4.14).
*   **Field Variables** (v4.19): light curves of many known variables at once (list file or VSX/GCVS query).
*   **Period Analysis** (v4.21): periodograms, spectral window, prewhitening, phase plots.
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
- `CHANGELOG.md`: Version history of code changes.
- `TODO.md`: Ideas and open items.
- `setups/`: Saved Light-Curve setups (v4.18).
- `photometry/filters.py`: Central B/V filter classification (v4.15).
- `photometry/noise.py`: Shared noise model (v4.16).
- `photometry/fastphot.py`: Fast frame conversion, centroid and sky statistics (v4.17).
- `tools/catalog_overview.py`: Which files of the local catalogue copies can be deleted or reduced (v4.28).
- `photometry/nina_export.py`: Night plan and NINA target-set export (v4.38).
- `photometry/planner.py`, `photometry/planner_tab.py`: Observation planner (v4.37).
- `photometry/instruments.py`: Instrument setups (telescope, camera, site, filters, settings) (v4.36).
- `photometry/aavso_sequences.py`: Stored AAVSO comparison-star sequences, one file per variable (v4.35).
- `photometry/online.py`: Availability of online services; unreachable servers are skipped for a while (v4.34).
- `photometry/apass_dr10.py`: APASS DR10 downloads from the AAVSO archive (v4.32).
- `photometry/names.py`: Star names -> coordinates (local GCVS list, then SIMBAD) (v4.30).
- `photometry/master_frames.py`: Master bias/dark/flat frames, automatic selection, calibration on the fly (v4.27).
- `calibration_config.json`: Calibration settings (auto-generated).
- `photometry/local_catalogs.py`: Local catalogue copies (detection, index, queries) (v4.26).
- `photometry/period_analysis.py`: Periodograms, spectral window, multi-frequency fit (v4.21), times of maximum and O–C (v4.24).
- `period_analysis_guide.md`: Practical guide to period analysis (v4.24).
- `photometry_output/period_analysis/`: Saved period analyses (v4.21).
- `photometry/field_variables.py`: Target lists, VSX/GCVS field queries, automatic comparison stars (v4.19).
- `photometry_output/field_variables/`: Field Variables results (v4.19).
- `tests/test_calibra.py`: Automated regression tests (v4.16).
