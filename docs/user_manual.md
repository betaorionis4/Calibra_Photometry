# Photometry with Calibra: Comprehensive User Manual

*Calibra version 4.46 (2026-11-17) — developed by Stephan Pomp, Google DeepMind Antigravity and Claude (Anthropic)*

Welcome to **Calibra**: an automated photometric analysis & calibration toolkit.

This short manual provides some background on the software's architecture, mathematical principles, and operational workflow.

The code can identify stars, perform PSF fitting to extract fluxes (using aperture photometry), and compares instrumental magnitudes with references magnitudes from catalogues available online (e.g. APASS DR9). Note that the provided fits file(s) need to have a WCS. If your files are not already plate solved, Calibra can automatically solve them using the integrated ASTAP Plate Solving tab.

Make sure that the code uses the right filter. I use Johnson V and B filters, that are labled 'V_mag' and 'B_mag' by my imaging software - I use N.I.N.A. - in the FITS header.

Since v1.5, Calibra supports automated color transformation calibration using paired B/V images. 
In v2.0, ensemble time-series photometry with multiple comparison stars and AAVSO-format light curve reporting were added. 
Version 3.0 introduces a centralized FITS File Manager, integrated Plate Solving via ASTAP, and a unified, logically flowing Analysis & Calibration tab. 
Version 3.1 adds the **Interactive FITS Viewer** with role-based star marking, real-time radial profile analysis, in-viewer aperture tuning, and full bidirectional synchronization between the viewer and the Light Curves pipeline. 
Version 3.2 implements full AAVSO-compliant Sequence retrieval (AUID caching) and automated Extended Format report generation, ensuring compatibility with AAVSO submission systems. 
Version 4.0 has a reworked GUI which integrates dedicated **"Open Selected FITS in Viewer"** launcher buttons, and extensive interactive quick-help educational popup panels across all processing tabs. 
Version 4.12 (May 25, 2026) has improved data mining accuracy by implementing centroid shift constraints for faint stars, adds right-click context menus to scatter plots, and properly passes configuration to the FITS viewer from suspects, and introduces dynamic target naming when sending suspects to the Light Curves tab. 
Version 4.13 (May 28, 2026) implements immediate WCS projected variable star markings in the FITS Viewer, adds dynamic Max Mag magnitude limits and Min Amp filters to FITS Viewer variable markings.
Version 4.14 (November 4, 2026) adds selectable **Light Curve Modes**: besides calibrated magnitudes, light curves can now be produced as **Raw ADU** (target only, no reference or check stars needed) or as **Relative flux** (reference stars used only through their measured ADU, catalogue magnitudes ignored), all with propagated uncertainties (see Section 6.6).
Version 4.27 (November 7, 2026) builds master bias/flat (and optional dark) frames from a folder of calibration frames, selects the matching masters automatically and calibrates on the fly or into files (Section 3.3).
Version 5.0.1 (November 18, 2026): no more `FITSFixedWarning` messages; the original Landolt CDS tables are read from the local catalogue folder; known variables come from the local GCVS list before VizieR when *Prefer local catalogues* is on; local catalogue copies that cannot be read are skipped (Section 3.6a); local GCVS lists load in about 1 s instead of about 30 s. Version 4.46 (November 17, 2026) puts the offline catalogues first: local copy, cache of earlier queries, then the local ATLAS-RefCat2 copy in place of a catalogue that is not available locally, and only then VizieR (Section 3.6a). Version 4.45 (November 15, 2026) reads local ATLAS-RefCat2 tiles completely (before, tiles whose first star had hexadecimal flags were skipped). Version 4.26 (November 7, 2026) uses local copies of the catalogues (ATLAS-RefCat2, Gaia DR3, APASS, VSX, a GCVS list) and goes online only for what is not available locally (Section 3.6a); Field Variables adds the variables of a local GCVS list automatically (Section 7a).
Version 4.25 (November 7, 2026) adds an editable, savable phase-plot window and moves the plot selector above the plots (Section 7b.6).
Version 4.24 (November 6, 2026) extends the Period Analysis with a fit-quality check, residuals vs time, times of maximum and the O–C diagram (Section 7b.6); a detailed practical guide is in `docs/period_analysis_guide.md`.
Version 4.23 (November 6, 2026) adds L (luminance) light curves (Section 3.4), reads target lists exported from GCVS searches of the SAAF variable-star section (Section 7a) and lets you choose the curve in the phase plot (Section 7b.5).
Version 4.22 (November 5, 2026) lets plate solving start from a position guess (object name or RA/Dec) and uses each solution as the guess for the next image (Section 3.1).
Version 4.21 (November 5, 2026) adds the **Period Analysis** tab: periodograms, spectral window, prewhitening of several frequencies and phase plots (Section 7b).
Version 4.20 (November 5, 2026) checks the field of view of every image and handles light curves from several nights with different pointings (Section 7a, "Several nights").
Version 4.19 (November 5, 2026) adds the **Field Variables** tab: light curves for many known variables in the field in one run, with targets from a list file or from AAVSO VSX / GCVS and automatically chosen comparison stars (Section 7a); `.fts` files are accepted.
Version 4.18 (November 4, 2026) improves usability: a fixed action bar and named setups in the Light Curves tab, dated output files, automatic comparison stars when sending a Data-Mining star to Light Curves, and one place for each setting (Section 6.7).
Version 4.17 (November 4, 2026) makes Light Curves (~2×) and especially Data Mining (~100× per star) much faster and lets Data Mining reuse its stored measurements (Sections 6.3 and 7.1a).
Version 4.16 (November 4, 2026) revises the science core: a corrected noise model (net flux, effective gain of averaged stacks, measured correlated-noise factor), a colour-transformation fit restricted to well-measured stars with weights, errors and a consistency check, ensemble zero points with full error propagation and a check star in Differential Photometry, the zero-point error of the mean, quality flags in all result files, plus several FITS Viewer, Data Mining and OSC improvements (Sections 2.3, 4, 5, 7 and `CHANGELOG.md`).
Version 4.15 (November 4, 2026) is a bug-fix release from a code review: one consistent B/V filter classification everywhere (Section 3.4), reliable bias/flat calibration output (32-bit float, named after the input file; Section 3.5), File Manager ticks that survive refreshes, a FITS Viewer that no longer overwrites the Light Curves setup when closed, a safer FITS header editor (backup + type preservation), working RA/Dec region filter, visible error messages, and more robust plate solving and Data Mining. See `CHANGELOG.md`.

Useful background information is provided by the AAVSO Guide to CCD/CMOS Photometry (available for free via aavso.org).

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
Since v4.16 one noise model (`python/noise.py`) is used by the pipeline, the Light Curves, Data Mining and (for comparison) the FITS Viewer:
$$\sigma_{F}^2 = \frac{F_{net}}{g_{eff}} + k^2\, A\, \sigma_{sky}^2 \left(1 + \frac{A}{n_{sky}}\right)$$
- $F_{net}$: background-subtracted flux in the aperture (ADU). *Before v4.16 the raw aperture sum (star + sky + pedestal) was used here, so the sky was counted twice.*
- $g_{eff}$: effective gain. For an **average** of $N$ frames (FITS keyword `STACKCNT`, `NCOMBINE`, …) $g_{eff} = N \cdot g$; for a **sum** or a single frame $g_{eff} = g$ (Settings → *Stacked images*: `average` / `sum` / `none`). *Before v4.16 the single-frame gain was always used — for your 49-frame V stack the Poisson term was 49× too large.*
- $A$: aperture area; $\sigma_{sky}$, $n_{sky}$: sigma-clipped scatter and number of annulus pixels actually used. $\sigma_{sky}$ already contains sky photon noise, read noise and dark noise, so these are not added separately.
- $k$: **sky noise factor** (Settings, default 1). Registering and interpolating frames for a stack correlates neighbouring pixels, so the noise of a sum of pixels is larger than $\sqrt{A}\,\sigma_{sky}$. Every pipeline run measures this on ~1500 empty-sky apertures and prints e.g. *"Blank-sky noise check … factor 1.40"*; enter that value to obtain realistic errors for stacks (single, unregistered frames: keep 1).
- **SNR** = $F_{net}/\sigma_F$; magnitude error $= 1.0857/SNR$. Detections with SNR < 3 are discarded.

Effect on your AE UMa stacks (v4.15 → v4.16, factor 1): median magnitude errors decreased by about 4–5× in V (49 frames) and 2× in B (8 frames); e.g. 0.087 → 0.019 mag at $m_{inst}$ = −9…−8 in V. A synthetic test in `tests/` confirms that the quoted errors match the actual scatter.

### 2.4 Session Logging
Calibra automatically generates a permanent record of every analysis run. 
- **Console Window**: A separate "Process Console" window opens to show real-time progress.
- **Log Files**: All console output is simultaneously saved to a `.txt` file in the `output/logs/` folder of the user folder (section 2.6).
- **Naming Convention**: Logs are named uniquely to prevent overwriting, e.g., `log_[FileName]_[Filter]_[Catalog].txt`. This allows you to track results across different calibration attempts or reference catalogs.

### 2.5 Instrument setups (v4.36)
Calibra keeps one **instrument setup** per telescope / camera / reducer / site, e.g. the 4" refractor in Uppsala, the MWT in Raspay and the Vespera 2. The active setup is chosen at the bottom of the sidebar (*Instrument setup*); a short line below shows focal length, pixel scale, field of view and site.

**What a setup contains**
- *Instrument*: telescope, aperture, focal length, reducer/corrector, camera, pixel size, sensor size, binning, colour (OSC) or mono, capture gain and offset. Pixel scale, field of view and focal ratio are calculated from these.
- *Site*: name, latitude, longitude, elevation, time zone (used for airmass and HJD, and later for observation planning).
- *Filters*: the filter name written in the FITS header for V, B, L, R, G, I, and the filter-wheel position (for NINA sequences).
- *Planning*: standard exposure times (e.g. 30, 60, 120 s) and the bright/faint magnitude limits of the setup.
- *Analysis settings*: all Calibra settings that depend on the instrument - CCD gain, read noise, dark current, saturation limit, detection and aperture settings, default zero points, colour and extinction coefficients, catalogue search radius, plate-solving field, pixel region, calibration folders and master files.

**How it works.** The analysis settings are the normal Calibra settings (Settings, Pre-processing, Light Curves tabs). They are stored in the active setup automatically - when you switch to another setup and when Calibra is closed. Choosing another setup in the sidebar first stores the current settings in the old setup, then applies the settings of the new one, so switching between e.g. refractor and MWT changes gain, saturation, apertures, zero points, calibration folders etc. in one step. Site coordinates and filter names edited in the setup dialog and in the Settings tab are kept in step.

**Manage Setups…** opens the list of setups:
- *New (current settings)* creates a setup that starts from the current settings; *Duplicate* copies the selected one (e.g. same telescope with reducer); *Delete* removes a setup (not the active one).
- *Fill from FITS Header (checked frame)* reads telescope, camera, focal length, aperture, pixel size, sensor size, binning, gain, offset, site coordinates and filter from the header of a checked frame (NINA, MaxIm, SharpCap ... write these); only empty fields are filled. Without a focal length the pixel scale is taken from the WCS of a plate-solved frame.
- *Save*, *Save and Activate*. Renaming: change the name and press *Save*.

**Which setup do these frames belong to?** When frames are loaded, Calibra compares their header (sensor size, focal length or pixel scale, pixel size, telescope and camera names) with all setups. If they fit another setup clearly better than the active one, a yellow note appears in the sidebar ("These frames look like 'MWT' - click to switch").

At the first start of v4.36 a setup called *Standard* is created from the settings Calibra had so far - rename it (e.g. to your refractor) and add the others. Setups are stored as JSON files in `config/instruments/` of the user folder (section 2.6). Not to be confused with the *target setups* of the Light Curves tab (one per variable star, section 6.7).

### 2.6 Where Calibra keeps its files (v5.0)

The program folder (where `calibra.py` is) contains only the program and its documentation. Everything Calibra writes while running goes into one **user folder**, by default `Calibra` in your home folder (`~/Calibra`, on Windows `C:\Users\<name>\Calibra`):

| Subfolder | Contents |
|---|---|
| `config/` | `calibra_session.json` (all settings, saved on exit), `calibration_config.json`, `instruments/` (instrument setups), `setups/` (target setups of the Light Curves tab), `local_catalogs.json`, `priority_targets.csv`, `open_clusters.csv`, `reference_stars.csv` (your own reference-star list; an example is copied there at the first start) |
| `cache/` | `catalogs/` (results of online catalogue queries, index of the local catalogue copies), `aavso_sequences/`, `apass_dr10_downloads.json`, `time_series_cache_v2.json`, `dm_cache/` (Data Mining measurements) - can be deleted, Calibra rebuilds it |
| `output/` | all results: pipeline CSVs and reports, light curves and AAVSO reports, `period_analysis/`, `field_variables/`, `logs/`, `plots/`, `nina/` |
| `catalogs/` | `APASS_DR10/` - APASS DR10 downloads when no local catalogue folder is set |

*Settings → Operator & Session → User Folder* shows the folder; **Open User Folder** opens it. **Change User Folder…** chooses another one (stored in `calibra_home.txt` in the program folder, used from the next start; existing files are not moved). The environment variable `CALIBRA_HOME` overrides both. To keep everything inside the program folder as before v5.0, write a single `.` into `calibra_home.txt`.

**Coming from v4.x.** Older versions wrote `calibra_session.json`, `calibration_config.json`, `instruments/`, `setups/`, `photometry_output/`, `photometry_plots/` and `photometry_refstars/` into the folder Calibra was started from. At the first start of v5.0 Calibra offers to copy them into the user folder (it looks in the program folder and the current folder, and lets you choose another folder). Later: **Import from older Calibra…** in the same settings box, or on the command line

```
python tools/migrate_user_data.py <old Calibra folder>
```

Nothing is deleted or moved; files already in the user folder are kept. Paths inside the settings that pointed to the old folders (e.g. recently used result files) are changed to the new locations.

## 3. FITS Pre-processing (Calibration)
Calibra can perform basic instrumental calibration (Bias and Flat-fielding) for raw FITS files directly from the camera.

## 3.1 Plate Solving (via ASTAP)
To perform photometric analysis, your FITS files must be plate solved (i.e., contain valid WCS headers like RA and DEC). If your raw files are not already plate solved, you can use Calibra's integrated **Plate Solving** tab. This feature utilizes ASTAP—an external command-line engine that must be installed separately, with its executable path provided in the GUI—to automatically add WCS coordinates to your files prior to analysis.

**Position guess (v4.22).** ASTAP solves an image within seconds if it knows roughly where the telescope pointed; without any position it has to search the whole sky ("blind solve"), which can take minutes or fail. In *Position Guess* choose:
- *From the FITS header* - the capture software (N.I.N.A., SGP, MaxIm, …) usually writes `RA`/`DEC` or `OBJCTRA`/`OBJCTDEC`; if they are missing, a blind solve is made;
- *Object name* - e.g. `AE UMa`, `M 67`, `NGC 2420`; *Look up* shows the coordinates (local GCVS list, then SIMBAD, then AAVSO VSX - see 3.6a);
- *RA / Dec* - typed in as `09:36:53 +44:04:00` (RA in hours) or `144.22 44.07` (degrees).

The guess overrides the header and is searched within the *Search Radius* (Solver Settings; a few degrees is fine). With *Use the solution of each image as the guess for the next image* (default on), only the first image of a series needs a guess - every solved image guides the next one, which also follows a slow drift of the pointing. If an attempt fails, Calibra tries, in this order, your guess, the header position and finally - if *Allow blind solving* is ticked - a blind solve. *Field height* tells ASTAP the image height in degrees when it cannot work it out from the header (focal length/pixel size keywords); 0 = automatic. The console shows which guess was used for every image.

## 3.2 One-Shot Color (OSC) Demultiplexing
When using a One-Shot Color (OSC) camera, each exposure contains a mosaic of red, green, and blue sensitive pixels arranged in a Bayer pattern (e.g. RGGB, BGGR). For scientific-grade photometry, analyzing raw color frames directly can introduce systematic errors due to color-channel cross-talk and interpolation. 

Calibra provides a dedicated **OSC Channel Split** tab under Pre-processing to demultiplex these files before ingestion into the photometry pipeline:
*   **Bayer Pattern Options**: Supports common patterns including `RGGB`, `BGGR`, `GRBG`, and `GBRG`.
*   **Extraction Methods**:
    *   **Superpixel (2x2 -> 1px)** *(Highly Recommended)*: Combines each 2x2 grid of Bayer pixels into a single monochrome pixel. The red, green, and blue intensities are mapped to independent monochrome FITS files. Green values are averaged. Crucially, **no interpolation** is performed, preventing spatial leakage/bleeding between channels and maintaining physical measurement purity. WCS headers are automatically updated (coordinates `CRPIX` are scaled by 0.5, and pixel scales `CDELT`/`CD` are doubled) to preserve sub-pixel accuracy.
    *   **Bilinear (Interpolated)**: Interpolates missing values for each color channel at every pixel coordinate to retain the original frame dimensions. Note that this correlates noise and is generally not recommended for high-precision science.
*   **Workflow Integration**: Channels are written with suffixes `_SP_R.fits`, `_SP_G.fits`, and `_SP_B.fits` and automatically re-imported into the File Manager. Since v4.16: integer R and B stay 16-bit; the averaged G channel (half-integer values) and any float input (e.g. 0–1 normalised stacks, previously destroyed by the 16-bit cast) are written as 32-bit float. All old WCS scale/distortion keywords are replaced by the rescaled solution (before, old `CDi_j` remained next to the new `PC`/`CDELT`, i.e. two conflicting pixel scales); SIP distortion terms are dropped (re-solve the split frames if needed). A warning is printed if the chosen Bayer pattern differs from the header `BAYERPAT`.
*   **Photometric Calibration**: The split green channel behaves as a standard Johnson $V$ filter mock-up. By mapping `V-Filter Keyword` to `G` and `B-Filter Keyword` to `B` in the **Settings** tab, Calibra will calibrate the split frames directly against standard catalogues.

## 3.3 Calibration: master frames and light frames (v4.27, folders per type v4.29)

For quick looks raw frames can be measured directly. For accurate photometry the frames are corrected with a **master bias** (or master dark) and a **master flat** per filter. Everything is in *Pre-processing → FITS Calibration*.

**1. Master frames.** Two ways to tell Calibra where the raw calibration frames are:
- **One folder per type (v4.29, recommended):** *Browse Bias Folder…* and *Browse Flat Folder…* (and *Browse Dark Folder…* only if you use darks). **Every FITS file** in the bias folder is used as a bias, every file in the flat folder as a flat - the file names do not matter and no `IMAGETYP` is needed. A file whose header says it is something else (e.g. `IMAGETYP = Light Frame`, as some capture programs write for every frame) is still used as the type of its folder - the folder you chose decides - and a warning is shown (since v4.33; v4.29–4.32 left such files out). The flat folder may contain flats of several filters: the filter comes from the `FILTER` keyword or, if that is missing, from the file or folder name (`flat_V_001`, `Flat-Lum-3`, `FLAT_Bmag`, `…/B/…`). Subfolders one level down are included.
- **One mixed folder (v4.27):** *Browse Calibration Folder…*: the type comes from the header (`IMAGETYP`/`FRAME`, e.g. "Bias Frame", "Flat Field", "Dark Frame") or from the words bias/offset, flat, dark, flat-dark in the file or folder name.

*Build / Update Masters* then (bias first, then darks, then flats):
- groups them by image size, binning, gain and offset, flats additionally by filter, darks by exposure time and temperature (±1 °C);
- combines each group (default **sigma-clipped mean**, rejecting outliers such as cosmic rays beyond 3 σ (robust median/MAD); or **median**) in strips, so that even 50 frames of 26 Mpixel need only ~200 MB memory (about 1 s per 26-Mpixel frame);
- corrects every flat with the master flat-dark of the same exposure if there is one, otherwise with the master bias, normalises each flat to its median and stores the master flat normalised to 1;
- writes the masters as 32-bit float FITS **into the folder of the raw frames**, e.g. `master_bias_6248x4176_bin1_g0_o50.fits`, `master_flat_L_6248x4176_bin1_g0_o50.fits`, `master_dark_120s_m10C_….fits`. The header records the number of frames (`NCOMBINE`), the method, the file names (`HISTORY`) and a fingerprint of the input files (`CALHASH`).
- **Reuse**: when the button is pressed again, a master whose input files are unchanged is not rebuilt; if frames were added or changed, it is rebuilt. A master flat also remembers which master bias was subtracted: if the bias is rebuilt, the flats are rebuilt as well (v4.29). Groups with fewer than *min. frames* (default 3) are skipped.

**Bias and flats without double counting.** Calibrated light = (light − master bias) / normalised master flat. The bias is removed from every flat *when the master flat is built* (flat − bias, then normalised to median 1), and removed once from every light. A Calibra master flat therefore contains no bias - it only describes the relative sensitivity of each pixel (vignetting, dust). Both masters are still needed for the lights: the bias removes the additive offset of the light, the flat the multiplicative response. Using a bias-subtracted flat is the standard way; Calibra never subtracts the bias from its own master flats a second time (they are marked `FLATNORM`). For master flats from other software see 3.5 (checkbox "already bias/dark-subtracted").

The list shows all masters in the calibration folders - including masters made by other software (e.g. SIRIL) if their name starts with `master` or their `IMAGETYP` contains "master"; masters stored normalised to 0–1 are rescaled.

**2. Which masters are used.** *Automatic* (default): for every light frame Calibra picks the masters with the same image size and binning (required), preferably the same gain and offset (otherwise with a note), and the master flat of the light's filter (V, B, L or any other name). *Manual*: fixed files for bias, flats V/B/L and an optional dark. *Show Masters for Checked Frames* lists the choice for every checked frame.

**Darks (optional).** Only used if *Use darks* is ticked: then the master dark with the light's exposure time (±5 %) and temperature (±3 °C) replaces the bias (a dark contains the bias); without an exact match a dark is scaled with the exposure time, `bias + (dark − bias)·t_light/t_dark`. Cameras without amp glow and with low dark current usually do not need darks for short exposures.

**3. Applying the calibration** - two ways:
- **On the fly** (*Calibrate on the fly when measuring*): Light Curves, Field Variables, Data Mining and the Analysis pipeline correct every frame in memory (masters loaded once; ~0.2 s per 26-Mpixel frame). No calibrated files, no extra disk space; the raw files stay as they are. Measurements with and without calibration are cached separately.
- **Files** (*Write Calibrated Files for Checked Frames*): `cal_<name>` (32-bit float, about twice the size of a 16-bit raw frame) in a `calibrated/` folder **next to the raw files**; they replace the raw frames in the File Manager. A file that exists and was made with the same masters is reused, not recalculated.

**Safety checks.** Frames that are already calibrated (`cal_` files, `BIASFILE`/`FLATFILE`/`DARKFILE`/`CALSTAT` keywords, calibration entries in `HISTORY`) are never calibrated twice. For **stacks** (`STACKCNT`/`NCOMBINE` > 1) a note is printed: calibrating after stacking is only an approximation, because registration shifts the flat-field pattern between the sub-frames - better calibrate the sub-frames before stacking (e.g. in SIRIL, or write calibrated files with Calibra and stack those). Pixels saturated in the raw frame (≥ *Saturation Limit*, Settings) keep a value at or above that limit, so saturated stars are still rejected. If no matching master exists, the frame is measured uncalibrated and a message is printed (once per folder).

## 3.4 Automatic Filter Detection
Calibra reads the `FILTER` keyword from the FITS header and classifies it as **V**, **B** or *other*. Since v4.15 one rule is used everywhere (File Manager counts, flat selection, catalogue band in the zero-point calibration, default zero points, Light Curves filter list, FITS Viewer); before, five different rules were in use and e.g. `Bessell V` was treated as B. The rules, in order (case-insensitive), are:

1. The filter equals the **V-/B-Filter Keyword** set in *Settings* (defaults `Vmag` / `Bmag`).
2. The filter is just `V` or `B`.
3. The keyword (≥ 2 characters) is contained in the filter name (e.g. `Vmag_2`).
4. The filter name contains `V` or `B` as a separate word (e.g. `Bessell V`, `Baader V`, `Johnson-B`).

Anything else (e.g. `L`, `R`, `Master_L`) is *other*: such files are skipped by the batch calibration (no matching flat), their zero point is not stored as a default, and the pipeline calibrates them against catalogue V magnitudes with a warning.

**L (luminance) light curves (v4.23).** For light curves (Light Curves and Field Variables tabs) a third band **L** is recognised: the *L-Filter Keyword* in *Settings → Filter Keyword Mapping* (default `L`) and names such as `L`, `Lum`, `Luminance`, `Lmag`, `Baader L`, `UV/IR-cut`. Choose `L` as the light-curve filter.
- *Relative flux* and *Raw ADU* light curves work exactly as for V or B - often the best choice for L data (periods, times of minimum, amplitudes in the instrumental system).
- *Calibrated magnitudes* use the **V** magnitudes of the comparison stars (catalogue or AAVSO sequence), i.e. the result is a magnitude on the V scale measured through a broad filter. In AAVSO reports the filter code is **CV** (clear/unfiltered, V zero point). Because the L band is much wider than V, target and comparison stars of different colour give an offset (up to several 0.1 mag for red stars such as Miras); choose comparison stars of similar colour and do not mix such data with real V data. No colour transformation is applied for L (coefficient 0), extinction uses k_V.

## 3.5 Output of Calibrated Files
Calibra performs the following operation (D = master bias, or master dark if darks are used):
`Calibrated = (Raw - D) / (Flat / median(Flat))`
I.e. by default it is assumed that the master flat is already corrected for bias. Since v4.15 the checkbox **"Master flats from other software (e.g. Siril) are already bias/dark-subtracted"** (ticked by default) controls this; untick it to use `(Raw - Bias) / norm(Flat - Bias)` for master flats that still contain the bias (since v4.29 also in automatic mode). Calibra's own master flats are always bias-subtracted and never corrected again. Flats stored normalised (median ≈ 1) are used as they are. Flat pixels below 5 % of the median are not corrected.

The resulting calibrated images are saved as new FITS files called `cal_<input file name>` in a `calibrated/` subfolder next to the raw files (since v4.27; before: `fitsfiles/calibrated/` in the Calibra folder) and replace the originals in the File Manager; the header records `BIASFILE`, `DARKFILE`, `FLATFILE`, `CALSRC`, `CALTAG` and `SATURATE`. Since v4.15:
- The file is named after the **input file** (previously after the `FILENAME` header keyword, which stacking software copies from a sub-frame, so the calibrated stack was not found and the *uncalibrated* file was silently used).
- Data are written as **32-bit float** (previously clipped at 0 and truncated to 16-bit integers, which biased low sky levels). `BZERO`/`BSCALE` are removed and `CALSRC` records the source file.
- A size mismatch between light, bias and flat stops the calibration of that frame with a clear message instead of crashing; a file only counts as calibrated if it was actually written in this run.

## 3.6 Online Catalog Transformations
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

**Local Landolt copy (v5.0.1).** The original CDS tables can be put into the local catalogue folder unchanged (see 3.6a). Only tables with V and B−V are used (e.g. `table2`; position-only tables such as `table5`, filter curves and notes are ignored). A star listed in several publications is used once, with the values of the newest one (2013/2009 before 1992); the 1992 positions are rounded and can be up to ~30″ off, so stars are matched by name (`BD`/`HD` prefixes ignored) within 60″, by V within 0.03 mag within 30″ (stars renamed later, e.g. `92 311` → `92 312`), or by position within 2″. A field without Landolt stars counts as not covered: the local ATLAS-RefCat2 copy is used instead (offline first).

#### APASS DR9
APASS provides high-quality, Johnson-band calibrated magnitudes for the entire visible sky, derived from the AAVSO Photometric All-Sky Survey.

#### Gaia DR3
Gaia is a space observatory mission led by the European Space Agency (ESA), providing astrometric, photometric, and spectrophotometric data for celestial objects. 

Transformations to Johnson V/B:
Based on **GAIA DR3 Documentation,Table 5.9**, which provides coefficients for V in the color range ($-0.5 < G_{BP} - G_{RP} < 5.0$) and for B in the color range ($-0.5 < G_{BP} - G_{RP} < 4.0$):
- $V = G + 0.02704 - 0.01424 \cdot C + 0.2156 \cdot C^2 - 0.01426 \cdot C^3$
- $B = G - 0.01448 + 0.6874 \cdot C + 0.3604 \cdot C^2 - 0.06718 \cdot C^3 + 0.006061 \cdot C^4$
- *where $C = G_{BP} - G_{RP}$*

## 3.6a Local copies of the catalogues (v4.26)

Catalogue queries over the internet take time and need a connection. Calibra can use local copies instead. Put them into **one folder with one subfolder per catalogue** and select that folder once: *Settings → Astro & Catalog → Reference Catalog Selection → Local catalogue folder → Browse Folder…*. Calibra finds the subfolders itself and remembers the choice (`config/local_catalogs.json` in the user folder), also for the next start. *Rescan* repeats the search (e.g. after adding a catalogue). The box below shows what was found.

| Catalogue | Recognised by | Expected files |
|---|---|---|
| ATLAS-RefCat2 | folder name containing `atlas` or `refcat`, or tile file names | the 1°×1° tile files as distributed by MAST/STScI (`144+44.rc2`, `144+44.csv`, `144+44`, also `.gz`), usually in magnitude folders `00_m_16`, `16_m_17`, `17_m_18`, … (folders from `18_m_19` on are skipped - not needed for calibration). Without header in the RefCat2 column order, with integer-scaled values (RA/Dec in 10⁻⁸ deg, magnitudes in mmag) or plain values; a header line with column names is also understood. Hexadecimal flag columns such as `2d` (RefCat2 *contrib* columns) are handled; before v4.45 a tile starting with such a star was skipped |
| Gaia DR3 | folder name containing `gaia`, or columns `phot_g_mean_mag`, `phot_bp_mean_mag` | CSV/TSV tables (also `.gz`) with header, e.g. exports from the Gaia archive or VizieR (`source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag`; VizieR names `RA_ICRS, DE_ICRS, Gmag, BPmag, RPmag` also work); `#` comment lines are skipped |
| APASS DR9 / DR10 | folder name containing `apass`, or columns `Johnson_V`/`Johnson_B` | tables with RA/Dec and V, B (`radeg, decdeg, Johnson_V, Johnson_B` or `RAJ2000, DEJ2000, Vmag, Bmag`); the CSV files of the APASS DR10 download page (`Johnson_V (V)` …, missing values `NA`) since v4.31 |
| AAVSO VSX (optional) | folder name containing `vsx` | table with name, RA, Dec, type, max, min, period |
| Landolt (optional) | folder name containing `landolt` | the original CDS tables as downloaded (v5.0.1): raw `table2.dat` with its `ReadMe` in the same folder, or VizieR/CDS downloads starting with `#Table:` (columns separated by `|`); one subfolder per publication (e.g. `1992`, `2009`, `2013`) - or a table with RA, Dec, V, B−V |
| GCVS list (optional) | file name containing `gcvs` (in the folder itself or in a `GCVS` subfolder) | same format as the target lists of Field Variables, e.g. the GCVS export of the SAAF variable-star section (`GCVS_selected.csv`) |

RA/Dec columns are recognised under the usual names (`ra`, `RA_ICRS`, `RAJ2000`, `radeg`, …; `dec`, `DE_ICRS`, `DEJ2000`, `decdeg`, …) and must be in degrees. Folders with other names are recognised from the column names of their files.

**How it is used (offline first, v4.46).** Every catalogue query (pipeline, colour transformation, Differential Photometry, Light Curves, Field Variables, Data Mining, FITS Viewer, VSX marking, Gaia cross-match) uses the sources in this order:

1. the **local copy** of the chosen reference catalogue (*Settings → Ref Catalog*);
2. the **cache** of an earlier online query of that catalogue for the same field (`cache/catalogs` in the user folder; only online results are cached - before v4.46 an old cached VizieR result was used even when a local copy existed);
3. for APASS only: the automatic **DR10 download** from the AAVSO archive, if switched on (see below);
4. the **local ATLAS-RefCat2 copy in place of the chosen catalogue** when the chosen one (Gaia DR3, APASS, Landolt) is not available locally for the field - option *Prefer local catalogues …* (default on). The console says so ("Gaia DR3 is not available locally for this field - using the local ATLAS-RefCat2 copy instead"); RefCat2 contains the Gaia G/BP/RP photometry and gives V/B from Pan-STARRS g/r;
5. **VizieR**, only if *Go online …* is ticked and VizieR answered recently;
6. with *Prefer local catalogues* switched off, the local RefCat2 copy is still used as a last resort when VizieR fails.

The same B/V conversions as for VizieR data are applied (ATLAS g, r → B, V; Gaia G, BP, RP → B, V). With *Prefer local catalogues* on, **Gaia source IDs** (used only as labels in reports, RefCat2 has none) are not fetched online. **Known variables** (VSX marking): local VSX copy → cache → with *Prefer local catalogues* on (since v5.0.1) the variables of the **local GCVS list**, without asking VizieR → VizieR → if VizieR is switched off or not reachable, the local GCVS list (fewer stars than VSX - a local VSX copy is better if you have one; without a GCVS list VizieR is asked as before).

**Catalogue copies that cannot be read (v5.0.1).** If none of the files of a local catalogue gives a single star - e.g. the binary Gaia catalogue of Cartes du Ciel (`.dat` files with a `gaia.hdr` starting with `CDCSTAR1`) - Calibra says so once in the console, marks the catalogue *FORMAT NOT READABLE, not used* in the list under *Settings → Local catalogue folder*, and from then on skips it (before, every look-up checked all files again: about 30 s for 73000 Gaia files). The local ATLAS-RefCat2 copy is used in its place. Choosing or rescanning the catalogue folder checks again. A server that does not answer is skipped for 10 min, after repeated failures for 20, 40 and at most 60 min, so a VizieR outage costs at most one 20 s timeout per hour. *Use the local copies* can be switched off to compare with the online data.

**Large tables.** For table-type catalogues Calibra records the sky area of every file once (index in `cache/catalogs/local_index_<catalogue>.json` in the user folder; updated automatically when files change) and later reads only the files that can contain the field. Many smaller files (e.g. Dec zones or HEALPix regions) are therefore much faster than one huge file. The first scan of large catalogues can take a while; it runs in the background.

**Which files can be deleted? (v4.28)** Calibra needs only stars brighter than about 18 mag, and you may not need the far southern sky. *Settings → Which files can be deleted?…* (or the command-line tool `tools/catalog_overview.py`) shows what each file contains - it never deletes anything. In the settings dialog enter the limits and press *Scan*; on the command line:

```
cd <Calibra folder>
python tools/catalog_overview.py /path/to/catalogs --dec-min -30 --mag-max 18
```

For every file it lists size, number of stars, Dec range, magnitude range and a verdict: **keep**, **DELETE: all Dec < −30**, **DELETE: all fainter than 18**, or **partly useful: x % of the stars inside the limits**. A summary per catalogue gives the disk space that deleting would free; the full list is written to `catalog_overview.csv` (from the settings dialog: in `output/` of the user folder; open it in a spreadsheet, sort by verdict).
- **ATLAS-RefCat2** is decided from the names alone (fast, no file is read): magnitude folders `18_m_19`, `19_m_20` (and fainter) can go; a tile `RRR-DD` covers Dec −DD … −DD+1, so for a limit of −30° all tiles from `-31` to `-90` can go (`-30` covers −30 … −29 and is kept).
- **Gaia / APASS** files are read once (large files take time). Files sorted by sky area (e.g. Dec zones) can often be deleted completely; files that mix all magnitudes (e.g. the Gaia bulk download by HEALPix region) are "partly useful". For those, `--reduce /path/to/reduced` writes **reduced copies** (only the rows inside the limits, same columns and header, same subfolders) into another folder - point Calibra to the reduced folder (and keep the ATLAS folder there or move it over) once you are happy with it.
- After deleting files press *Rescan* in the settings.

**APASS DR10 from the AAVSO download page (v4.31).** On archive.aavso.org/download-apass-data enter RA and Dec of the field centre (sexagesimal `21:07:58.2`, `-05:17:40`, or degrees) and a radius in degrees (at most 2; the result is a square of 2 × radius), choose *CSV output* and save the file (named like `apass_316.99_-5.29_0.5_20_none.csv`) **unchanged** into the APASS subfolder of the catalogue folder, then press *Rescan*. Calibra reads these files directly (columns `radeg`, `decdeg`, `Johnson_V (V)`, `Johnson_B (B)`, …; stars without V are skipped). Choose a radius a bit larger than half the field diagonal (e.g. 0.5° for a 40′ field) and centre it on your target: a field is only taken from the local files if its centre lies inside the area covered, and stars outside the downloaded square are missing. Several downloads may overlap - duplicate stars are removed. DR10 has more stars and better calibration than DR9 (VizieR, used online). Please use the download page with moderation (AAVSO asks to contact them for large areas).

*Download from within Calibra (v4.32).* The button **Download APASS DR10…** (Settings → local catalogues; also in the Light Curves tab and in Field Variables) does the same without the web page: choose *Field(s) of the checked images* (plate-solved; Calibra computes centre and radius from the image size + 10 %, one download per pointing), a *star name* or *RA/Dec* with a radius. The CSV is saved unchanged in the APASS folder of the catalogue folder (created as `APASS` if there is none; without a catalogue folder in `catalogs/APASS_DR10` of the user folder). A field that was downloaded before is not downloaded again (record in `cache/apass_dr10_downloads.json`).

*Automatically.* With **APASS: download DR10 automatically …** ticked (Settings; default off) and APASS chosen as reference catalogue, Calibra downloads DR10 by itself whenever a field is not in the local files, and uses it - then and for all later runs. If the AAVSO server does not answer or returns an error, VizieR (APASS DR9) is used and no further automatic download is tried for 30 minutes; automatic downloads are at least 10 s apart. Rich Milky Way fields can take a minute or more.

**Star names from the local GCVS list (v4.30).** Wherever a star name can be entered (target and comparison stars, Light Curves, Field Variables target lists, the position guess for plate solving), Calibra first looks the name up in the local GCVS list and goes online only if it is not there: 1. local GCVS list, 2. SIMBAD, 3. AAVSO VSX. Spelling details do not matter: `HU Aqr`, `hu aqr`, `HU  AQR`, `HU_Aqr` are the same, as are `V1500 Cyg` and `V 1500 Cyg`. The *Check* buttons show where the name was found, e.g. "Found (local GCVS list): 21h07m58.2s, −5d17m40.0s". This works offline; it is on whenever a GCVS list is in the catalogue folder and *Use the local copies* is ticked. Names that are not in the list (e.g. `M 67`, HD numbers, non-variable stars) are still resolved online; with *Go online* switched off they give a clear message instead. The coordinates are those of the list (GCVS, J2000).

## 3.7 AAVSO VSX Integration (Variable Star Exclusion)
To ensure the highest photometric rigor, Calibra automatically cross-matches all detected stars and reference catalogs against the **AAVSO International Variable Star Index (VSX)** via VizieR (`B/vsx/vsx`).

- **Calibration Rigor**: Any reference star found within 2 arcseconds of a known VSX variable is automatically excluded from the Zero-Point derivation.
- **Statistical Purity**: Known variables are excluded from the Gaussian fits in Accuracy Evaluation plots to prevent their intrinsic fluctuations from artificially inflating the reported scatter ($\sigma$).
- **Identification**: All output CSVs and reports include an `is_variable` flag ("Yes"/"No") to help you identify known variable sources in your field at a glance.
- **Local Caching**: VSX data is cached in `cache/catalogs/` of the user folder to minimize network overhead during repeat analysis.

---

## 4. The Processing Pipeline (A-G)

The pipeline processes each FITS file through seven sequential modules:

### Stage A: Star Detection (`star_detection.py`)
Uses `DAOStarFinder` to scan for density peaks matching a stellar profile. It enforces strict morphological constraints (Sharpness and Roundness) to prevent hot pixels or satellite trails from being flagged as stars.

### Stage B: Coordinate Refinement (`psf_fitting.py`)
Isolates every detected star and fits the 2D Gaussian. 
- **Diagnostics**: Generates radial profile plots and residual images to verify the quality of the fit.
- **Saturation Check**: Flags stars exceeding the hardware linearity limit.

### Stage C: Aperture Photometry (`aperture_phot.py`)
Applies the "Aperture vs. Annulus" math described in Section 2.1. It reports the background-subtracted ADU counts and applies the noise model of Section 2.3. Since v4.16 it also:
- skips stars whose aperture is not completely on the image (flag *edge*; previously they got strongly negative fluxes);
- records the highest raw pixel in the aperture (`peak_raw`), the annulus statistics (`bkg_median`, `bkg_std`, `n_sky`) and a **blend** flag (another detection within the outer annulus radius);
- runs the blank-sky noise check described in Section 2.3.

### Stage D: Data Filtering (`main.py`)
Purges bad data. Any star with negative net flux or an $SNR < 3.0$ is discarded. The pipeline reports the instrumental magnitude of the faintest surviving star as the **Detection Limit**.

### Stage E: Calibration & Zero Points (`calibration.py`)
Matches detected stars against a reference catalog using a spatial **KD-Tree**.
- **Catalog Sources**: Supports local CSV files or **Automated Online Retrieval** from **ATLAS-RefCat2**, **APASS DR9**, or **Gaia DR3** via VizieR.
- **Local Caching**: Online query results are cached locally in `cache/catalogs/` of the user folder to minimize network usage and allow offline repeat analysis.
- **Filtering**: Only stars with an SNR above the user-defined threshold (default 10.0) are used to calculate the zero point. Since v4.16 also excluded: stars with a raw peak above **Max Peak for Calib Stars** (default 30 000 ADU — in registered stacks non-linearity sets in well below the saturation limit; on the AE UMa V stack the ZP residual rose by +0.04 mag at 30–40k ADU), blends (optional, default on), edge stars and known variables.
- **Zero Point (ZP)**: Derives a **Median Zero Point (ZP)** and applies it to all targets to find their **Calibrated Magnitudes**. Since v4.16 the quoted ZP uncertainty is the **error of the median**, $1.2533\,\sigma/\sqrt{N}$ (previously the star-to-star scatter σ, which was also added to every star's error); the scatter is reported separately. With a single calibration star 0.05 mag is assumed.
- **Colour diagnostic (v4.16)**: the slope of the ZP against catalogue $B-V$ is fitted and printed/reported. The pipeline magnitudes are **untransformed** (no colour term); the slope tells you how large the colour error is for a star whose colour differs from the median comparison colour (AE UMa stacks: −0.003 mag/mag in V, +0.10 mag/mag in B). Use Differential Photometry or the Light Curves colour term for transformed magnitudes.
- **Catalogue radius**: *Catalog Search Radius* = 0 sizes the query automatically to the image half-diagonal (87.7′ for the AE UMa stacks); a radius smaller than the field gives a note in the log.
- **Detection FWHM** (Settings → Detection, default 3.5 px) is now configurable; the log notes when the measured median FWHM differs by more than 30 %.

### Stage F: Shift Analysis (`shift_analysis.py`)
Computes the difference between where the FITS header (WCS) *thinks* the star is and where the pixel math *proves* it is. 
- **Console Summary**: Provides a quick median shift report (dX, dY, dRA, dDec) on screen.
- **Detailed Report**: Generates a full markdown report detailing individual tracking errors and rotational shifts.

### Stage G: Color Transformation Calibration (`color_calibration.py`)
A specialized post-processing tool for deriving instrumental **Color Terms**. 
- **Requirements**: Requires a B-filter result CSV and a V-filter result CSV of the same field.
- **Airmass & Extinction**: Automatically handles airmass extraction from results and applies atmospheric extinction correction ($k_B, k_V$).
- **Star selection (v4.16)**: only stars with SNR ≥ *Min SNR (colour fit)* (default 50) in **both** filters, raw peak ≤ *Max Peak for Calib Stars*, not saturated, not blended, not variable, optionally brighter than *Faintest catalogue V*. Before v4.16 all matched stars were used; the faint ones biased the fit (AE UMa field: $T_{b,bv}$ = 0.148 with all stars, −0.007 for V < 12).
- **Weighted robust fitting (v4.16)**: weighted least squares (instrumental errors from the CSV plus 0.02 mag catalogue error), iterative 3σ rejection on normalised residuals with a MAD-based scale (previously one 2σ pass with the non-robust std). $T_{bv}$ is obtained by fitting $(b-v)$ against the catalogue $(B-V)$ and inverting, because the catalogue colour is the less noisy variable (as in the AAVSO Transform Generator).
- **Uncertainties & consistency (v4.16)**: each coefficient is reported with its standard error and $\chi^2_\nu$ (errors are inflated when $\chi^2_\nu > 1$). The three coefficients must satisfy $T_{bv} = 1/(1 - T_{b,bv} + T_{v,bv})$; the report states whether they agree within 3σ, and a table of V-term residuals versus magnitude reveals faint-end bias or non-linearity. The JSON file now also stores the errors, the consistency flag and the selection. Result for your AE UMa stacks: $T_{bv}$ = 0.970 ± 0.011, $T_{b,bv}$ = 0.014 ± 0.008, $T_{v,bv}$ = 0.033 ± 0.007 (consistent; implied $T_{bv}$ = 0.981).
- **Visual Diagnostics**:
    - **Red 'X'**: Outliers that were statistically rejected.
    - **Red Dotted Lines**: The 2-sigma boundary of the first (unclipped) fit.
    - **Grey Shaded Corridor**: The final, refined 2-sigma confidence window.
- **Coefficients Derived**:
    - **$\mu$ (Color Scale)**: Transformation from $(b-v)$ to $(B-V)_{std}$.
    - **$\psi$ & $\epsilon$**: Color terms for B and V filters respectively.
- **Reporting**: Generates a regression report and three diagnostic plots showing color residuals and the statistical cleaning process.

---

## 5. Differential Photometry

The Differential Photometry module computes standard magnitudes $B$, $V$, $B-V$ with uncertainties for every star in a field. Since v4.16 the zero points are by default derived from an **ensemble** of all suitable comparison stars; a single comparison star (chosen automatically, by name or by coordinates) remains available.

### 5.1 Methodology
- **Target Matching**: The module automatically cross-matches stars from your B-filter and V-filter results (using a 2-arcsecond search radius) to form reliable $(b-v)$ pairs.
- **Reference Catalog Query**: The matched coordinates are queried against your chosen standard catalog (e.g., ATLAS, APASS, Landolt). The catalog search covers a field of view that is defined by the "Catalog Search Radius" in the GUI.
- **Comparison Star Selection**:
  - **Automatic ensemble (default, v4.16)**: all catalogue-matched stars that are not known variables, have SNR ≥ *Min SNR (comp stars)* (default 30) in B and V, a raw peak ≤ *Max Peak for Calib Stars*, no neighbour in the annulus and $-0.2 \le B-V \le 1.6$. Each gives $Z_{BV}, Z_B, Z_V$; the zero points are the 3σ-clipped means and their uncertainties the errors of the mean. *Before v4.16 the single **brightest** star with $0.4 \le B-V \le 0.8$ was used — the one most at risk of non-linearity; on the AE UMa field this left a +0.08 mag offset in V against ATLAS (v4.16 ensemble: +0.011).*
  - **Automatic single star**: one star closest to the median brightness and colour of the candidates (instead of the brightest).
  - **Search by Name Mode**: The user inputs a specific star name (e.g., "AE UMa"). The pipeline resolves the exact RA and Dec coordinates from the local GCVS list (v4.30, see 3.6a) or via the SIMBAD astronomical database and anchors the photometry to this object. You can instantly verify the resolved coordinates using the Check button in the GUI before executing.
  - **Manual Mode**: The user inputs specific RA and Dec coordinates ($h, m, s$ and $d, m, s$). The pipeline finds the matched detection within a 4-arcsecond tolerance of those coordinates, verifies it has catalog data, and strictly forces it to be the reference anchor.
- **Zero Point Calculation**: Using the Color Transformation Coefficients ($T_{bv}, T_{b\_bv}, T_{v\_bv}$) and atmospheric extinction ($k_B, k_V$), the pipeline derives the instrumental zero points ($Z_{BV}, Z_B, Z_V$) from the comparison star(s). The coefficients (and, since v4.16, their uncertainties) are loaded with *Load Last Coefficients* from the Color Transformation module (Stage G) or entered manually.
- **Transformation (v4.16)**: $(B-V) = T_{bv}(b-v)_0 + Z_{BV}$, $V = v_0 + T_{v,bv}(B-V) + Z_V$ and **$B = V + (B-V)$**, so $B$, $V$ and $B-V$ are always consistent. (Before, $B$ came from its own equation; with inconsistent coefficients $B - V$ differed from the reported $B-V$ by up to 0.2 mag.)
- **Uncertainties (v4.16)**: `B_err`, `V_err`, `B_V_err` combine the instrumental errors of $b$ and $v$ (propagated through the equations), the zero-point errors and the coefficient errors multiplied by the colour difference to the mean comparison-star colour (correlations neglected).
- **Check star (v4.16)**: enter a name or "RA Dec" in *Check star*; its measured $B, V, B-V$ are compared with the catalogue in the report.

### 5.2 Target Selection Modes
In the Differential Photometry tab, you can define which stars to process:
- **Analyze All Stars (Default)**: The pipeline computes standard magnitudes for every common B/V pair in the image. This is useful for survey work or verifying field-wide accuracy.
- **Analyze a specific Target Pair**: The pipeline isolates a single star (via Name/SIMBAD or Manual Coordinates) and computes standard magnitudes for only that object. This mode skips the population-wide Accuracy Evaluation plotting as it is not statistically valid for a single target.

### 5.3 Standard Magnitude Output
These zero points are instantly applied to all other stars in the field to get standard magnitudes ($B, V$) and color index ($B-V$). The final standard magnitudes and color index are saved in a Markdown table (`differential_photometry_results.md`) alongside a more detailed CSV file (with all the instrumental data, errors, and variability flags).

### 5.4 Accuracy Evaluation
To evaluate the calibration quality, the pipeline automatically compares the internally computed standard magnitudes against the actual catalog magnitudes for all matching stars (excluding known VSX variables, and the reference star in single-star mode; in ensemble mode the comparison stars are included, i.e. the statistics are partly in-sample — use the check star for an independent test).
- **Statistical Fitting**: It calculates the deviations ($\Delta B$, $\Delta V$, $\Delta(B-V)$) and fits a Gaussian distribution to determine the mean offset ($\mu$) and standard deviation/scatter ($\sigma$).
- **Plotting**: It generates a 3-panel histogram plot with the Gaussian fits overlaid (`output/plots/diff_photometry_deviations.png`), allowing you to rapidly identify any systematic errors or estimate your measurement uncertainties. The statistical fits are appended to the `differential_photometry_report.md`.

---

## 6. Time-Series Photometry & Light Curves

The Light Curves module performs **ensemble differential time-series photometry** on a sequence of FITS images to produce calibrated light curves of variable stars.

Since v4.14 the module offers three **Light Curve Modes** (selected in the *Light Curve Mode* box at the top of the tab). Sections 6.1–6.3 describe the default **Calibrated magnitudes** mode; the **Relative flux** and **Raw ADU** modes are described in Section 6.6.

### 6.1 Ensemble Comparison Stars

Rather than relying on a single comparison star (which may itself vary slightly, have a bad pixel, or land on an image artifact in some frames), Calibra supports up to **5 comparison stars** measured simultaneously. Each star's name can be resolved via SIMBAD, and its standard magnitude and $B-V$ color can be fetched automatically from the selected reference catalog.

For each comparison star $i$ in each frame, the pipeline independently derives a zero point:

$$ZP_i = V_{\text{std},i} - \left( m_{\text{inst},i} - k \cdot X + T_{V_{bv}} \cdot (B-V)_i \right)$$

where $m_{\text{inst},i}$ is the instrumental magnitude, $k$ is the extinction coefficient, $X$ is the airmass, and $T_{V_{bv}}$ is the color term.

The ensemble zero point is the mean of the individual values:

$$\overline{ZP} = \frac{1}{N} \sum_{i=1}^{N} ZP_i$$

### 6.2 Calibration of the Target

The target star's calibrated magnitude in each frame is:

$$V_\text{target} = m_{\text{inst,target}} - k \cdot X + T_{V_{bv}} \cdot (B-V)_\text{target} + \overline{ZP}$$

Note that $(B-V)_\text{target}$ is a user-supplied assumed value. Since it is held constant across all frames, any error in this assumption shifts the entire light curve by a fixed offset but does **not** affect the measured amplitude or period of variability.

### 6.3 Uncertainty Propagation

*(v4.17: each frame is converted to floating point once and star positions are refined with a fast windowed centroid instead of a 2-D Gaussian fit — same accuracy, about half the time per frame; see `python/fastphot.py`.)*

*(v4.16: $\sigma_\text{phot}$ uses the noise model of Section 2.3 — effective gain for averaged stacks and the sky noise factor. The photometry cache is now `time_series_cache_v2.json`, keyed also by the noise settings, so measurements from older versions or with other noise settings are re-measured.)*


The total uncertainty on each data point combines two independent sources in quadrature:

$$\sigma_V = \sqrt{\sigma_\text{phot}^2 + \sigma_{\overline{ZP}}^2}$$

where:

- $\sigma_\text{phot}$ is the target star's aperture photometry error (Poisson + background noise, see Section 2.3).
- $\sigma_{\overline{ZP}} = \sigma_{ZP} / \sqrt{N}$ is the standard error of the mean zero point from the ensemble, using Bessel's correction ($N-1$ denominator) for the sample standard deviation.

When only a single comparison star is used ($N=1$), a floor value of $\sigma_{\overline{ZP}} = 0.01$ mag is applied to prevent unrealistically small error bars that would hide the inherent uncertainty of using a single calibrator.

### 6.4 Advanced Plot Customization
The Light Curve module provides highly interactive visualization tools:
- **Inline Visibility Toggles**: Checkboxes beneath the preview instantly toggle the visibility of the Target, Check Star, and individual Reference Stars without requiring a re-run of the pipeline.
- **Advanced Plot Customizer**: Clicking the `[ ⚙️ Customize Plot ]` button opens a dedicated window for creating publication-ready plots.
  - Set manual Y-Axis bounds to zoom in on specific variations.
  - Provide custom legend labels (e.g., `Ref 1` instead of `Ref: Gaia DR3 12345`).
  - Override default colors (e.g., `darkblue`, `green`, `red`) and marker symbols (`o`, `x`, `s`, `*`, etc.) for every single data series.
  - Automatically exports customized views to high-resolution PNG, PDF, or SVG files.

### 6.5 Output

The module produces the following files in `output/` of the user folder. Times per frame (since v4.42/4.43): `date_ut` = UT date of mid-exposure (YYYY-MM-DD), `ut_mid` = UT of mid-exposure as HH:MM:SS (together they replace the JD column) and `hjd` = heliocentric Julian date (UTC) of mid-exposure. The same holds for the result table of the Light Curves tab (*UT (mid)* with date and time, *HJD*) and the Field Variables light curves.

| File | Mode | Contents |
|---|---|---|
| `light_curve_[StarName].csv` | Calibrated | UT date and time of mid-exposure (`date_ut` YYYY-MM-DD, `ut_mid` HH:MM:SS), HJD, magnitude, error, SNR, zero-point, number of ensemble stars per frame |
| `aavso_[StarName].txt` | Calibrated | AAVSO Extended Format report (ready for submission) |
| `plot_[StarName].png` | Calibrated | Light curve plot with error bars and inverted magnitude axis |
| `light_curve_[StarName]_rawADU.csv` | Raw ADU | Per-frame net flux and error in ADU (or ADU/s), instrumental magnitude, check/reference fluxes |
| `plot_[StarName]_rawADU.png` | Raw ADU | Flux (or instrumental magnitude) light curve |
| `light_curve_[StarName]_relflux.csv` | Relative flux | Per-frame flux ratio and error, ensemble flux, transparency-corrected ADU, check/reference ratios |
| `plot_[StarName]_relflux.png` | Relative flux | Relative-flux (or differential instrumental magnitude) light curve |

The `_rawADU` / `_relflux` suffixes mean that uncalibrated runs never overwrite the calibrated results of the same target. No AAVSO report is written in the uncalibrated modes (an AAVSO `STD` report needs standard magnitudes).

### 6.7 Working with the Light Curves tab (v4.18)
- **Action bar**: *Generate Light Curve*, *Open Selected FITS in Viewer*, *Save Target Setup…*, *Open Target Setup…* (called *Save/Open Setup…* before v4.36) and *Open Output Folder*, plus progress bar and status line, stay at the top while the tab scrolls.
- **Stored AAVSO sequences (v4.35)**: every sequence fetched with *Get AAVSO Ref Stars* is stored as one file per variable in `cache/aavso_sequences/` of the user folder (e.g. `huaqr.json`: comparison stars with AUID, AAVSO V/B, the catalogue cross-match, field size and download date). Next time - also after switching to other targets or restarting Calibra - the stored sequence is used without asking AAVSO; the status line says "stored sequence, fetched <date>". Names are matched loosely (`HU Aqr` = `hu aqr` = `HU_Aqr`). A new query is made only if the field (*catalog search radius*) is now larger than the stored one, or with **Refresh from AAVSO** in the sequence window (AAVSO revises sequences now and then; a sequence older than a year is pointed out). If you change the reference catalogue, only the cross-match is redone (local copy first) and stored. The FITS Viewer shows the stored sequence of the current target automatically.
- **Target setups**: *Save Target Setup…* writes the current target, comparison and check stars, light-curve mode, colour term, chart ID and (if fetched) the AAVSO sequence of that target to `config/setups/<target>.json`; *Open Target Setup…* restores it. Use one target setup per variable star and switch between them. The observing site and the instrument-dependent settings belong to the *instrument setup* (section 2.5), not to target setups.
- **Output names** contain the UT date of the first observation (`light_curve_<target>[_relflux|_rawADU][_faint]_<YYYY-MM-DD>.csv`, `plot_…png`, `aavso_<target>_<YYYY-MM-DD>.txt`). Re-running the same night and mode overwrites only that night's files.
- **Settings used from elsewhere**: extinction $k_V$/$k_B$ (Settings → Astro & Catalog, chosen by filter) and the observing site (Settings → Operator & Session; only if the FITS header lacks `SITELAT`/`SITELONG`). The *Coefficients, AAVSO code & chart* section can be collapsed.
- **From Data Mining**: *Send to Light Curves (+ comp stars)* fills the target and automatically 4 comparison stars and 1 check star (see Section 7.1b) in *Relative flux* mode.

### 6.6 Light Curve Modes: Calibrated, Relative Flux and Raw ADU (v4.14)

The *Light Curve Mode* box at the top of the Light Curves tab selects how the target's brightness is expressed:

| Mode | Needs reference stars? | Uses catalogue mags / B-V / k / colour term? | Result | AAVSO report |
|---|---|---|---|---|
| **Calibrated magnitudes** (default) | Yes (≥1) | Yes | Standard magnitude ± error (Sections 6.1–6.3) | Yes |
| **Relative flux** | Yes (≥1), only their ADU | No | $F_\text{target} / \sum F_\text{ref}$ ± error | No |
| **Raw ADU** | No | No | Target net flux in ADU (or ADU/s) ± error | No |

All modes use exactly the same aperture photometry, centroid tracking, flexible-aperture option and photometry cache as before; only the way the measured fluxes are combined differs.

#### 6.6.1 Raw ADU

For each frame the target's background-subtracted net flux $F_t$ (Section 2.1) is reported directly, with the CCD-equation uncertainty (Section 2.3):

$$\sigma_{F_t}^2 = \frac{F_t}{g} + n_\text{pix}\,\sigma_\text{sky}^2 + \frac{n_\text{pix}^2\,\sigma_\text{sky}^2}{n_\text{sky}}$$

where $g$ is the gain (e⁻/ADU, from Settings), $n_\text{pix}$ the aperture area and $n_\text{sky}$ the number of annulus pixels. With **"Raw ADU: normalise to ADU/s"** ticked, $F_t$ and $\sigma_{F_t}$ are divided by the frame's `EXPTIME` — use this if the exposure time changes within the sequence.

The CSV also contains the instrumental magnitude $m_\text{inst} = -2.5\log_{10}F_t$ with $\sigma_m = 1.0857\,\sigma_{F_t}/F_t$ (no extinction correction). Frames with $F_t \le 0$ are kept (and flagged `SUSPECT`), since a faint target near minimum can legitimately give a slightly negative net flux.

**No reference or check star is needed.** If reference or check stars *are* defined, they are measured for display only (their ADU can be toggled on in the plot and appear in the statistics) — they are not used to correct the target. Raw ADU curves therefore contain all transparency and airmass variations; they are ideal for a quick look at a sequence, for timing eclipses/minima on photometric nights, or for fields without usable comparison stars.

**Automatic use:** with **"Automatically use Raw ADU when no reference stars are selected"** ticked (default), pressing *Generate Light Curve* with no reference star marked *Use* switches the mode to Raw ADU (an information box tells you). Untick it to get the old error message instead.

#### 6.6.2 Relative flux (reference-star ADU only)

The reference stars act purely as a flux standard for the atmosphere and instrument; their catalogue magnitudes, $B-V$, the colour term and extinction coefficient are ignored (so you do not need to fetch any catalogue values — the *Mag* and *B-V* fields can contain anything). With the ensemble flux $S = \sum_i F_i$:

$$R = \frac{F_t}{S}, \qquad \sigma_R = \sqrt{\left(\frac{\sigma_{F_t}}{S}\right)^2 + \left(\frac{F_t\,\sigma_S}{S^2}\right)^2}, \qquad \sigma_S = \sqrt{\sum_i \sigma_{F_i}^2}$$

(independent errors assumed). Because target and comparison stars are observed through the same atmosphere, first-order transparency and airmass changes cancel. The CSV additionally contains:

- `mag`, `mag_err`: the differential instrumental magnitude $\Delta m = -2.5\log_{10}R$ and $1.0857\,\sigma_R/R$;
- `ens_flux`, `ens_flux_err`: $S$ and $\sigma_S$ per frame;
- `flux_corr_adu`: $R \cdot \langle S\rangle$ — the target flux in ADU "corrected" to the mean transparency of the run;
- `check_flux`: check star flux divided by $S$ (should be constant);
- `ref_fluxes`: for each reference star, $F_i / (S - F_i)$, i.e. relative to the sum of the *other* references (only with ≥2 references). A reference star whose ratio drifts or scatters more than the others is variable or badly measured and should be dropped.

#### 6.6.3 Plotting, flags and statistics

- **"Raw/Relative: plot flux"** (ticked by default) plots flux (y-axis not inverted). Untick it to plot the instrumental / differential instrumental magnitude instead (inverted axis). The inline plot updates immediately; the *Customize Plot* window inherits the choice and the matching default axis label.
- **Outlier flags** use the same rolling-median + MAD algorithm as the calibrated mode, but in flux space with a minimum jump of 5 % of the median flux (≈ 0.05 mag). Flags can still be toggled by right-clicking the results table.
- The **Numerical Results** table relabels its columns to *Flux [ADU]* / *Rel. Flux* and *Check Flux*.
- The **Analysis Summary** reports the mean, standard deviation, relative scatter (%) and median formal error of the target, and the scatter of the check and reference stars in % and approximate mag ($\sigma_m \approx 1.0857\,\sigma_F/F$).

Tip: comparing the scatter of the check star in Relative-flux mode with the median formal error is a quick test of whether the error bars are realistic (the two should be similar for a constant star).

---

### 6.8 Faint targets: forced photometry and combining frames (v4.44)

**When to use it.** The target is so faint in single frames (S/N below ~5) that it cannot be centroided reliably, but longer exposures would blur rapid changes (e.g. a polar such as HU Aqr dropping into its faint minimum: ingress, egress and the length of the minimum need the time resolution of short exposures). Normal light curves centroid the target in every frame; at low S/N the centroid wanders to noise peaks and the measured flux is biased upwards, and frames with zero or negative flux are dropped.

**How it works** (box *Faint Target* in the Light Curves tab, works with all three light-curve modes):

1. *Position*: the aperture is placed at the target's WCS position in every frame, corrected by the mean offset (centroid − WCS position, weighted by S/N) of the reference stars selected in the Light Curves tab. The reference stars are bright, so the offset is precise; the summary shows the mean offset and its frame-to-frame scatter as a check of the plate solution (MWT, HU Aqr 13 Nov: −0.31/−0.78 px, scatter 0.07/0.04 px). The target position never moves with the noise.
2. *Aperture*: one radius for all frames and all stars: *Aperture radius* × median FWHM of the frames (FWHM from the reference stars; default 1.5 — close to the best S/N for faint stars), or the radius from Settings if the factor is 0. The sky annulus starts at radius + inner gap (Settings). Because target and comparison stars use the same aperture in the same frame, the light outside the aperture cancels in the ratio.
3. *Per frame* Calibra stores the net fluxes (ADU) and errors of target, reference stars and check star — also when the target flux is zero or negative (dropping such frames would make the faint points too bright).
4. *Combining*: a light-curve point of a group of frames uses the **sums** of the fluxes of every star over the frames of the group. This is what an aligned stack of the frames would give, but without shifting/resampling images. Then the magnitude (or relative flux, or mean raw flux per frame / per second) is computed as in Sections 6.2 and 6.6. The error of the sum is $\sigma = \sqrt{\sum \sigma_i^2}$, so S/N grows as √N for N equal frames. The time of the point is the exposure-weighted mean of the frames (HJD, UT date and time); the plot shows the time span as a horizontal bar. A reference star counts in a group only if it was measured in all frames of the group.
5. *Upper limits*: if S/N of a point is below *Upper limit when S/N <* (default 3), the point is given as "fainter than" the magnitude of max(F, 0) + 3σ: table `<18.13`, flag `LIMIT`, plot symbol ▽, AAVSO report `<18.132` with MERR `na`. In flux plots the flux itself is shown (it may be negative).

**Choosing the groups** (no new measurement needed — the frames of the last run are re-combined):

| Control | Result |
|---|---|
| *Combine in groups of X consecutive frames* (set before *Generate Light Curve*, or change and press *Apply Groups*) | e.g. 8 frames, X = 2 → 4 points. A group is closed early at a gap in time (more than 5 × the normal frame spacing, e.g. another night); the last group may be shorter. |
| *Combine Selected Rows* | the rows marked in the result table (Ctrl/Shift + click) become one point — any frames, also combined ones. |
| *Split Selected Rows* | the marked points back to single frames. |
| *Single Frames* | every frame its own point. |

The table column *N* shows the number of frames per point. CSV columns added: `n_frames`, `t_span_min`, `upper_limit`, `mag_limit`, `wcs_dx`, `wcs_dy` (pixels). Output files have `_faint` in their name and are overwritten with every re-combination.

**Practical hints.** Keep the frames single (or in small groups) around the expected changes and combine generously where the star is constant in its faint state; never combine across an ingress or egress. Times of ingress/egress are best taken from the single frames. Example HU Aqr in minimum (MWT, 60 s, mag ~19.2): S/N 1–2 per frame, ~3 for 4 frames, 4.3 for 8 frames. The check star (measured in the same way) shows whether the method works on your data.

## 7. Data Mining (Automated Variable Star Discovery)

Calibra includes a powerful, automated **Data Mining** module designed to discover unknown or undocumented variable stars in your FITS image sequences without requiring any prior knowledge of targets or field geometry.

### 7.1 Mathematical Principles of Detection
Data Mining runs a bulk analysis over a sequence of images to detect variability statistically:
1.  **Bulk Tracking**: Every detected point source in a user-selected reference frame is registered. The algorithm tracks their positions across all loaded frames in the sequence (using sub-pixel centroid refinement).
2.  **Ensemble Detrending**: To remove global systematic variations (such as atmospheric extinction changes, passing thin clouds, or minor tracking errors), Calibra builds a "Global Ensemble" from the brightest, most stable stars in the field. Every star's light curve is detrended against this global ensemble.
3.  **Poisson Noise Floor Modeling**: The pipeline fits a statistical "noise floor" curve describing the expected RMS scatter as a function of stellar magnitude. This noise floor represents the physical limit of the camera and is calculated using Poisson photon noise:
    $$\sigma_{\text{expected}}^2 \approx \sigma_{\text{poisson}}^2 + \sigma_{\text{sky}}^2 + \sigma_{\text{read}}^2$$
4.  **Excess Scatter Flagging**: Any star exhibiting a measured standard deviation (RMS) that lies significantly above the fitted noise floor and exceeds a minimum absolute variability threshold (default: 0.03 mag) is flagged as a **Suspect**. Since v4.16 the excess is computed in quadrature, $\sqrt{RMS^2 - \sigma_{exp}^2}$, and each star's **own** expected noise (median formal error of its measurements with the noise model of Section 2.3) is used as well: a suspect must also have RMS ≥ 2× its own photometric error (CSV columns *Phot. Error (median)* and *RMS / Phot. Error*, which is ≈ 1 for constant stars). This prevents false suspects among bright stars where the empirical noise floor is poorly defined. The pipeline also uses Median Absolute Deviation (MAD) to filter out stars with massive, unphysical frame-to-frame jumps (such as cosmic rays or tracking errors), correctly reclassifying them as **Noisy** rather than suspect variables.

### 7.1a Speed and the measurement cache (v4.17)
- **Measurement**: each frame is read and converted to floating point once; all stars are then measured in a single vectorised aperture-photometry call. Positions are refined with a fast Gaussian-windowed centroid (after a coarse peak search within *Max Tracking Shift*); if a star is not found within that radius, the WCS-predicted position is used, as before. Typical cost on a 26-Mpixel frame: ~0.7 ms per star, i.e. ~1–2 s per frame for 2000 stars (v4.16: ~70 ms per star).
- **Cache**: the measurements of every frame are saved in `cache/dm_cache/` of the user folder (≈100 kB per frame for 2000 stars), and the star list from the reference frame too. When you run Data Mining again on the same frames with the same reference frame and measurement settings (aperture, annulus, tracking shift, saturation limit, edge buffer, gain, stack mode, sky noise factor, FWHM), the stored values are used and only the statistics are recomputed — changing *Min. valid frames*, *Ensemble size*, *Suspect sigma* or *Minimum excess* then takes seconds. If a FITS file or any measurement setting changes, the affected frames are measured again automatically.
- **Options** (Data Mining tab): *Reuse stored measurements* (default on) and *Clear Measurement Cache* (deletes the folder).

### 7.1b From a suspect to a light curve (v4.18)
*Send to Light Curves (+ comp stars)* (button or right-click) transfers the selected star and chooses comparison stars from the Data Mining statistics of the same frames: stars that are constant (not suspect/noisy, not a known VSX variable, RMS ≤ 1.5× their own photometric error, no disturbed background), measured in at least 95 % of the frames, within 1 mag of the target (widened stepwise if fewer than 5 qualify) and as close as possible on the sky (score = Δmag + distance/30′). The best four become comparison stars, the fifth the check star; the mode is set to *Relative flux*. Press *Generate Light Curve* — thanks to the faster photometry this takes seconds. For calibrated magnitudes, fetch catalogue values (*Fetch* / *Get AAVSO Ref Stars*) and switch to *Calibrated magnitudes*.

### 7.2 Catalog Cross-Matching & VSX Integration
Once suspects are identified, Calibra performs a spatial query against the **AAVSO International Variable Star Index (VSX)** via VizieR. 
- A search radius of 5 arcseconds is used to correlate suspects with known variables.
- Suspects are automatically categorized:
  - **Known VSX Variables**: Displays the official VSX Name and Variability Type (e.g. `RRAB`, `EA`, `SR`).
  - **New Discoveries**: Marked explicitly if no VSX match is found within the search radius.
- Real-time statistics are shown in the GUI, showing:
  - **Stars Analyzed**: Total number of tracked stars in the field.
  - **Suspects**: Total candidates exceeding the variability threshold.
  - **Known VSX**: Number of candidates matching known VSX records.
  - **New Discoveries**: Number of candidate variables without matching catalog records.
  - **Median RMS**: Overall quality metric of the photometry sequence.

### 7.3 Interactivity & Data Export
- **Visual Mapping**: The module generates a **Variability Diagram** (RMS Scatter vs. Magnitude) showing all stable stars (blue dots), the fitted noise floor (green line), and suspects (large red stars). The diagram is fully interactive: selecting a row in the table draws a bright yellow halo and ID label around the corresponding point in the diagram, and clicking points on the plot highlights them.
- **Row Coloring & Columns**: The Suspects Table highlights candidates for easy manual verification (Red for full suspects, Amber for borderline candidates with elevated scatter). It includes a **Valid Frames** column to show how many images the star was successfully measured in.
- **Right-Click Context Menu & Preview**:
  - *Copy RA/Dec*: Quick clipboard copying for target identification.
  - *Send to Light Curves*: Instantly exports coordinates to the Light Curves tab for detailed time-series plotting.
  - *Preview Light Curve*: Select a suspect and click this dedicated button to instantly view a pop-up plot of the magnitude across all frames without leaving the tab.
  - *Copy Row as CSV*: Copies a single star's data to the clipboard.
- **Exporting Results**: Click **"Export All Results"** to save a CSV containing coordinates, mean magnitudes, RMS scatter, excess scatter, and VSX metadata for every single star tracked in the sequence.

---

## 7a. Field Variables: light curves of many known variables (v4.19)

Data Mining looks for *unknown* variables and is most sensitive to changes within one night. To follow **known** variables - in particular long-period variables (Miras, semiregulars, irregulars) over weeks or months - use the *Field Variables* tab. It measures light curves for any number of variables in the field in one run.

**Workflow**
1. Load the images of all nights (any number of files; plate-solved, i.e. with WCS) in the File Manager and check them. Choose the filter (V or B) in the tab; only checked images of that filter are used.
2. Build the target list:
   - *Load List…* reads a text or CSV file, one star per line, e.g.
     ```
     # name        RA            Dec
     R Leo         09:47:33.5    +11:25:44
     X UMa, 08 40 49.4, +50 08 12
     RS UMa;183.0213;58.4917
     AE UMa
     ```
     RA in hours (sexagesimal) or degrees, Dec in degrees; separated by blanks, commas, semicolons or tabs. A header line `name,ra,dec[,type,period,max,min,epoch]` is also understood. A line with only a name is resolved from the local GCVS list (if there is one, see 3.6a), else via SIMBAD or AAVSO VSX - this needs an internet connection.

     **CSV exports from GCVS searches** (e.g. from the variable-star section of SAAF) can be loaded directly (v4.23):
     ```
     "Var ID","Namn","Typ","RA","Dekl","Max","Min","Band","Epok","Period","Stig%","Sp Type","Nova"
     "6301090","V1090 Per","DSCT","03 22 42.50","+39 06 34.6","11.600","12.300","CV","2455515.73900","0.1101054100","28","",""
     ```
     Column names are recognised in English and Swedish (name/Namn, RA/Rektascension, Dec/Dekl/Declination, type/Typ, period, max, min, epoch/Epok), in any order; columns that are not needed are ignored, empty fields are allowed, and Windows (Excel) encodings with å, ä, ö are read. A minimum in parentheses, e.g. `(0.30)`, is an amplitude (GCVS convention). **Required are a name, RA and Dec**: a list with a header but without RA or Dec columns is refused with a message listing the columns found; without a name column the stars are called Star_1, Star_2, … and you are reminded to add names; rows with empty RA/Dec are skipped and listed.
   - *Query VSX (incl. GCVS)* or *Query GCVS* lists all catalogued variables in the field of the first checked image, keeping those that pass the filters: brighter at maximum than the given limit, amplitude at least *Min. amplitude*, period inside the given range (stars without a catalogued period are kept if the box is ticked), type containing one of the given words (e.g. `M SR L`; empty = all types; constant stars are always dropped), and at least *Edge margin* pixels from the border. VSX already contains all GCVS stars plus many more and gives the most complete list; the GCVS query is useful for the classical designations.
   - **Local GCVS list (v4.26)**: if the local catalogue folder (Section 3.6a) contains a GCVS list (e.g. `GCVS_selected.csv`), *GCVS (local list / online)* uses it instead of VizieR, and with *Add variables from the local GCVS list automatically* (default on) the variables of that list lying in the field of the checked images are added whenever the Field Variables tab is opened with a new set of checked images - with the filters above (brightness, amplitude, period, type, edge margin). Stars already in the list are not added twice.
   - *Add…* adds a single star by name or coordinates; *Remove Selected*, *Select All/None*, and clicking the *Use* column change the selection; *Save List…* writes the list as CSV that can be loaded again. A star coming from both a list and a catalogue appears only once. Targets outside the image are shown but deselected.
3. Choose the light-curve mode and press *Run Light Curves for Selected Targets*.

**Comparison stars.** For every target Calibra chooses its own comparison stars from the reference catalogue set in *Settings* (ATLAS, APASS, …): catalogue stars on the image that have no neighbour within 20″ (no blends), are not known variables, lie within the chosen radius of the target (default 12′; widened up to 2× if needed) and are close in brightness to the target (default ±1.5 mag around the target's catalogue magnitude or the middle of its catalogued range; the window is widened stepwise), never brighter than 8 mag. The best *N* (default 5, ranked by brightness difference + distance) form the ensemble, the next one is the check star. For a large-amplitude variable the brightness at the time of observation can differ from the catalogue value; comparison stars a magnitude or two fainter or brighter are not a problem for the ensemble method as long as they are well exposed and not saturated.

**Modes.** *Calibrated magnitudes* (ensemble zero point and colour term as in the Light Curves tab; the target's B−V from the catalogue if available, else 0.5), *Relative flux* (same comparison stars, only their measured flux is used) or *Raw ADU* (no comparison stars). The photometry for each target is exactly that of the Light Curves tab (apertures, centroid tracking, noise model, outlier flags); each image is read only once for all targets.

**Output** (`output/field_variables/<date of first image>_<filter>/` in the user folder):
- `lc_<name>.csv` and `plot_<name>.png` per target (suffix `_relflux` / `_rawADU` in the other modes),
- `summary.csv`: number of good points, mean, standard deviation, 5–95 % range, median error, reduced χ² (scatter relative to the errors; ≫ 1 means variability or problems) and time span,
- `overview.png`: one small panel per target,
- optionally AAVSO reports `aavso_<name>.txt` (calibrated mode; check them before submitting - the comparison stars come from the photometric catalogue, not from an AAVSO sequence).

The table in the tab shows the same statistics. *Use Selected in Light Curves Tab* copies the star's name and coordinates to the Light Curves tab, where you can analyse it with hand-picked comparison stars (e.g. the AAVSO sequence).

**Several nights.** Load and check the images of all nights (they may be in different folders - use *Load Directory* several times). Things to know:
- The pointing usually differs a little from night to night. Calibra checks every image: the column *On images* shows on how many of the checked images a target lies; targets on none are skipped. A target is simply not measured on images where it is outside the field.
- Comparison stars are chosen among stars that are on all images containing the target, so that the same ensemble is used every night. If a comparison star is nevertheless missing on some images, calibrated magnitudes use the remaining stars, and relative flux is rescaled (column `ens_complete` = False marks those points).
- Measurements are stored in the session cache (`cache/time_series_cache_v2.json`): when a new night is added and the run is repeated, usually only the new images need to be measured.
- Output for more than one night goes to a folder named `<first date>_to_<last date>_<filter>`; the same naming is used for the files of the Light Curves tab.
- Keep the photometric settings (aperture, gain, stack mode, filter) the same for all nights, and use the same exposure scheme if possible. Transformation to the standard system (colour term) matters more when combining nights with different airmass or with data of other observers.

**Tips.** For long-period variables a few images per night are enough; combining the nights is done simply by checking all images. Mira variables can be several magnitudes fainter at minimum than at maximum: set *Brighter than* with this in mind. Very red stars (M-type) have large B−V; the colour term then matters - determine the transformation coefficient (Analysis & Calibration) and enable it in the Light Curves tab (*Coefficients*), which the Field Variables tab also uses.

---

## 7b. Period Analysis (v4.21)

> **A detailed practical guide** – planning observations, a worked example (HADS, AE UMa-like), how to judge whether one period is enough, how to identify additional periods, times of maximum and O–C, notes for other types of variables – is in **`docs/period_analysis_guide.md`**. A simulated practice light curve, `examples/example_HADS_simulated.csv`, comes with Calibra.

The *Period Analysis* tab determines periods from one or several light curves. Time is in days (HJD), frequency f in cycles per day (c/d), period P = 1/f.

### 7b.1 Data
*Load Light Curve(s)…* reads Calibra light-curve CSV files (Light Curves or Field Variables output) or any CSV with a time column (`hjd`, `bjd`, `jd`, `mjd`, `time`) and a value column; *Load Last Light Curve* takes the last result of the Light Curves tab, and *Period Analysis of Selected* in Field Variables opens a target directly. Several files (e.g. different nights or seasons) are combined; each file gets its own colour. Choose the quantity (`mag` for calibrated curves, `flux` for relative/raw flux; any numeric column), use *only OK points* to drop points flagged SUSPECT, and tick *subtract the mean of each file* when the files have different zero points (relative flux with different comparison stars, data from another observer). Do **not** subtract the means when the star's mean brightness really changes between the files (long-period variables!).

The line below the file list gives the key numbers of the sampling:
- **time span T** and **frequency resolution 1/T**: two frequencies closer than about 1/T cannot be separated; the frequency uncertainty is much smaller than 1/T for good data, but peaks are about 1/T wide;
- **longest useful period ≈ T/2** (better: cover at least two cycles);
- **pseudo-Nyquist frequency 0.5 / (median spacing)**: the highest frequency that can be found (with uneven sampling, higher frequencies are possible but less reliable).

### 7b.2 Periodogram
- **Lomb–Scargle** fits a sinusoid at every trial frequency (least squares; equivalent to a Fourier amplitude spectrum for uneven sampling). Calibra shows the **amplitude** of the best sinusoid in the unit of the data, so peak heights can be read as semi-amplitudes. *harmonics* > 1 fits sinusoid + harmonics (better for non-sinusoidal shapes such as RR Lyrae or eclipsing binaries). The false-alarm probability of the highest peak (Baluev 2008) is reported.
- **PDM** (phase dispersion minimisation, Stellingwerf 1978) folds the data at every trial period, divides the phase into 10 bins and compares the scatter within the bins with the total scatter (θ). Shape-independent; Calibra shows 1 − θ, so good periods are maxima. Slower than Lomb–Scargle.
- **BLS** (box least squares) searches for short periodic dips - suitable for eclipsing binaries with narrow eclipses (and transits). For magnitudes the sign is handled automatically.

Empty period limits mean automatic limits (from about 2 × median spacing to 2 × T). For long-period variables set the limits yourself (e.g. 20 to 1000 d) - this also makes the computation fast. *Oversampling* sets the grid step (1/(oversampling · T)). The x axis can show frequency or (log) period. **Clicking into the periodogram** shows the phase plot at the clicked peak (green line) without changing the fit.

### 7b.3 Aliases and the spectral window
Gaps in the data create false peaks. The **spectral window** is the periodogram of a pure sinusoid sampled like your data; *Show Spectral Window* draws it (orange) on top of the highest peak and lists its strongest side peaks. Every real peak is accompanied by this pattern. Typical cases:
- observations from one site on consecutive nights: aliases at f ± 1 c/d (± 2 c/d, …) - the classic problem when the true and the alias peak have similar height;
- observations every few nights or weeks: aliases at f ± 1/(spacing);
- seasons: aliases at f ± 1/365 c/d.
If two peaks differ by such a value, more data (longer nights, more nights, data from other longitudes) are needed to decide; the phase plot of the correct period usually shows less scatter.

### 7b.4 Prewhitening: further periods on top of a main period
1. Compute the periodogram and press *Prewhiten: Add Highest Peak*. The strongest frequency is fitted (with the chosen number of harmonics) and subtracted; the periodogram now shows the **residuals**.
2. Press it again for the next frequency. After every step **all** frequencies, amplitudes and phases are fitted together (non-linear least squares), which also improves the earlier frequencies.
3. Stop when the highest remaining peak has **S/N < 4** (Breger et al. 1993; S/N = amplitude / mean residual amplitude within ±1 c/d). Rows with S/N < 4 are marked.
4. Look at the notes: `2f1` (harmonic - better: increase the number of harmonics of f1 with *Apply to Selected*), `f1+f2` (combination frequency, typical of non-linear pulsation), `f1+1 c/d alias?` (probably an alias - check the spectral window), `< 2/T: trend?` (a slow drift or zero-point differences between files, not a period).
5. A frequency can also be added by clicking its peak and *Add Preview Frequency*, or by value (*Add Frequency…*: frequency in c/d, or a period followed by `d`, e.g. `0.0860169d`). *Remove Selected/Last* and *Reset* undo steps.

Uncertainties follow Montgomery & O'Donoghue (1999) and assume white noise; with correlated noise (e.g. night-to-night zero-point changes) the real errors are larger. Examples: δ Scuti stars such as AE UMa (radial fundamental and first overtone), Blazhko RR Lyrae stars (side peaks close to the main frequency - need a long time span), semiregular variables with two periods, or a rotation/orbital period superimposed on pulsation.

### 7b.5 Plots and saving
Top: the light curve with the fitted model; middle: periodogram of the residuals with the fitted frequencies (red); bottom: phase plot of the selected frequency with all other frequencies subtracted (two cycles) - frequencies that are harmonics (2f, 3f, …) of the selected one are **kept**, because they describe the shape of its light curve.

*Curve in phase plot* (v4.23): **Fit** draws the fitted model of that frequency (sine plus its harmonics), **Binned mean** the median of the points in phase bins - a model-free curve that shows the real shape - and **None** shows only the data. Light curves of pulsating stars are rarely sinusoidal: δ Scuti and RR Lyrae stars rise steeply and decline slowly (sawtooth); eclipsing binaries have narrow minima. A pure sine (1 harmonic) then fits badly and leaves harmonic peaks (2f1, 3f1) in the residuals. Fit such a frequency with 3–5 harmonics (*Harmonics for a new frequency* before adding it, or *Apply to Selected* afterwards) or use PDM, which makes no assumption about the shape. Use the toolbar to zoom. *Save Results…* writes the frequency table (CSV with all settings in the header), the figure (PNG), the periodogram (`_periodogram.csv`) and the residuals (`_residuals.csv`, e.g. for an O−C analysis or Period04).

### 7b.6 Fit quality, residuals and times of maximum (v4.24)
- **Quality line** (below the frequency table): residual rms, median photometric error, their ratio and the reduced χ², with a short verdict (≈ 1: the frequencies describe the data within the noise; > 1.3: signal left), and the night-to-night scatter of the mean residual.
- **Bottom plot** selector: *Phase plot*, *Residuals vs time* (residuals of all fitted frequencies with the mean of each night – shows zero-point offsets, trends and beating of unfitted modes) or *O-C diagram*.
- **Notes**: harmonics are recognised up to 6f; combinations are formed only from independent frequencies; for an independent frequency the period ratio with f1 is compared with the ratios of radial double-mode pulsators (RRd ≈ 0.745, δ Sct/HADS F/1O 0.756–0.79, Cepheid F/1O ≈ 0.70–0.72, 1O/2O ≈ 0.80).
- **4. Times of maximum and O–C**: one time of maximum per observed cycle of the selected frequency (other non-harmonic frequencies subtracted first), by a *template* fit of the fitted light-curve shape (recommended, works for any shape) or a *parabola*; O–C against the ephemeris from the fit or a given T₀/P (e.g. GCVS); straight-line fit → corrected ephemeris, optional parabola → period change dP/dE, dP/dt; *Save Maxima (CSV)…*.
- Light-curve files with comment lines (`#`) are read.
- **Plot selector (v4.25)**: the buttons *Phase plot / Residuals vs time / O-C diagram* sit directly above the plots; selecting a frequency in the table or clicking into the periodogram always returns to the phase plot.
- **Phase Plot Window (edit / save)… (v4.25)**: opens the phase plot of a chosen frequency in its own window. Editable: frequency, curve (fit / binned mean with number of bins / none), title, axis labels, phase range (e.g. −0.5 to 1.5 to show the maximum in the middle) and y range (empty = automatic), point and font size, error bars, colour per file, inverted y axis, grid, legend, and *phase 0 = maximum light* (T₀ from the fitted light-curve shape; otherwise the reference time of the fit). Text fields take effect with *Apply / Redraw* or Enter. *Save Figure* writes PNG (figure size in inches and dpi selectable), PDF or SVG (vector formats for publications); *Save Phase Data (CSV)* writes HJD, phase, value (data minus all other non-harmonic frequencies, in the original unit), error, file and model, e.g. for plotting in a spreadsheet or another program.

---

## 7c. Observation Planner (v4.37)

*Observation Planner* (sidebar) suggests targets for one night. Everything is computed locally - no internet needed.

**Where and when.**
- *Instrument setup*: the site (latitude, longitude, time zone) and the magnitude limits come from an instrument setup (section 2.5) - the active one by default; choose e.g. *MWT* to get suggestions for Raspay. Enter the site and time zone (e.g. `Europe/Stockholm`, `Europe/Madrid`) in *Manage Setups…*; without a time zone the computer's zone is used.
- *Night of*: the date of the evening (today by default). *from* / *to*: local times (`18:00` to `24:00`, or over midnight `20:00` to `02:00`), or `dusk` / `dawn` for the start/end of darkness.
- *Darkness*: only times when the Sun is below -6, -12 (default) or -18 degrees count; *any* ignores the Sun. The line below shows the dark period, the Moon's phase and how long it is up. (In June there is no nautical darkness in Uppsala - choose *civil* or *any*.)

**Selection.**
- *Altitude ≥* (default 30 degrees), *Azimuth (deg) from … to …*: range in degrees from North through East (0 = N, 90 = E, 180 = S, 270 = W; a range through North e.g. `300` to `60`; 0 to 360 = any). The list next to it fills in common ranges: `N`, `NE`, `E` (45-135), `SE`, `S`, `SW`, `W`, `NW`, `east half` (0-180), `south half`, `west half`, `north half` (since v4.39; before, the sector was typed as text).
- *Moon distance ≥* (degrees; only while the Moon is up), *Observable at least* (hours inside the window and sector).
- *Magnitude from/to*: the star must reach the range at least part of the time (empty = limits of the setup; stars without magnitudes are kept). *Types*: GCVS types separated by blanks, `*` as wildcard (`EA EW DSCT* M`); combined types like `EA/SD` match `EA`. *Period* (days) and *Amplitude ≥* (max-min, mag).

**Sources.**
- *Local GCVS list*: all stars of the GCVS list in the local catalogue folder (e.g. `GCVS_selected.csv`, section 3.6a).
- *Priority list*: your own list (`config/priority_targets.csv` in the user folder, same columns as target lists plus `notes`). Every entry gets a status: *observable in your window*, *above the altitude limit elsewhere* (with best time, altitude and azimuth) or *not observable in this night*. Add stars with **Add to Priority List** (from any result list), remove with **Remove from Priority List**; the file can also be edited by hand (e.g. AAVSO/SAAF campaign targets).
- *Open clusters*: the clusters in `config/open_clusters.csv` (user folder) - created with IC 1805, NGC 1912 (M38), NGC 2355, the Pleiades, NGC 6834, NGC 188, NGC 869/884 (h+chi Per), NGC 457 and NGC 663 (approximate centres and radii; add more lines as `name, alias, ra, dec, radius_arcmin`). The note gives the number of known variables (local GCVS list) within 1.5 x the cluster radius or half the field of the setup; **Variables in Cluster…** lists them.

**Results.** The table gives RA and Dec (J2000), type, magnitude range, period, the time span in which the star is observable (window, darkness, altitude, sector, Moon), the best time (highest altitude inside the window), altitude and azimuth in degrees then, the meridian passage (`*` = outside the window), hours and Moon distance. Click a column header to sort (again to reverse); sorting by *Name* sorts by constellation first and then in the GCVS order of the designations (Bayer letters, R … Z, RR … ZZ, AA … QZ, V335 …). The chart shows the altitude over the night (thick where observable) for the selected rows (or the first six), the altitude limit and the Moon; grey = not dark. With one star selected, a dashed vertical line marks its meridian passage (or the title gives the time if it is outside the window).

All tabs get a horizontal scrollbar when the window is narrower than their content (v4.39), so nothing is cut off on small laptop screens.
- **Set as Light Curves Target**, **Add to Field Variables** (a selected cluster adds its variables), **Save List (CSV)…**.

Accuracy: a few arcminutes (low-precision Sun and Moon positions, refraction ignored) - ample for planning.

### 7c.1 Night plan for NINA (v4.38)

**Night Plan for NINA…** (planner) turns the selected targets (or, without selection, all observable ones) into a plan for the night and writes it as a NINA *simple sequencer* target set (*Sequence → Load Target Set* in NINA). Intended for the refractor; for the MWT the planner only suggests targets.

*Prerequisites in the instrument setup* (Manage Setups…): filter names and filter-wheel positions (button **Filters from NINA File…** takes them from one of your existing NINA sequences, e.g. `L=0, R=1, G=2, B=3, Bmag=5, Vmag=6`), capture gain/offset and binning, aperture, focal length, pixel size, and in the Settings the CCD gain (e-/ADU), saturation limit, read noise and detection FWHM. Planning fields: standard exposure times, *ZP V* (electrons per second for a star of V = 0; empty = estimated from the aperture), sky brightness (mag/arcsec², e.g. 19.5-20 in Uppsala, 21+ in Raspay) and typical seeing.

*What Calibra decides per target*
- **Exposure time**: the longest standard exposure (e.g. 30/60/120 s) that keeps the star at its brightest below 70 % of the saturation limit and is not longer than period/50 (enough points per cycle). If even the shortest standard time saturates, 1-20 s are used (or a note says the star is too bright). The S/N at maximum and minimum light is shown; "faint at minimum" if S/N < 10.
- **How long**: eclipsing binaries with epoch and period in the GCVS list: the predicted primary minimum ± 6 % of the period (at least 45 min, at most 3 h) if it falls in the night, otherwise a few exposures; without epoch a time series. Period ≤ 0.5 d (δ Sct, RRc, EW …): one full cycle + 30 min (at most 5 h). 0.5-1 d: 2 h. Longer periods, Miras, semiregulars and stars without period: a few exposures (*Snapshot exposures*, 3 for periods > 20 d). Clusters: a 2-hour series of the field.
- **When**: eclipses at their time, time series around their best time, snapshots in the gaps - always inside the time the target is observable (window, altitude, sector, Moon). Then the blocks are moved together, because NINA's simple sequencer runs the targets one after the other. A remaining gap (before an eclipse, or until a target is high enough) is written as *Delay* of the next target - the plan holds if the sequence is started at the time shown; the dialog lists such waiting times.

*Dialog*: filters (`V`, `V,B`, `L` … - several filters are taken in turn, mode *Rotate*), overhead per target (slew, centring, guiding; default 3 min) and per exposure (download, default 4 s), number of snapshot exposures, sky brightness; Slew / Center / Start guiding / Dither. The table shows start, end, exposure, frames per filter, S/N and the reason; double-click a row to change exposure or frames or to leave a target out. **Write NINA Target Set…** saves `Plan_<date>_<setup>.ninaTargetSet.xml` (by default in the `NINA` folder next to the Calibra program folder, otherwise in `output/nina/` of the user folder) and a schedule `…_plan.csv`. Check the exposure times with the first frames - the model is simple (Gaussian star, estimated sky), and the zero point is only estimated unless you enter a measured one.

## 8. Running the Pipeline: The Configuration GUI (v4.0)

Launch the pipeline via `python calibra.py` to open the **Configuration GUI**. In v4.0, the interface is split into a persistent top panel and a set of eight processing tabs:

### 8.1 FITS File Manager (Top Panel)
Always visible at the top of the window, this centralized manager allows you to load, preview, and select your FITS files.
- **Add Files/Directory**: Load your images (`.fits`, `.fit` and, since v4.19, `.fts`; any capitalisation).
- **Preview**: View FITS headers directly in the GUI.
- **Edit Header**: Right-click a file or use the dedicated "✏️ Edit FITS Header..." button to open an interactive editor. This allows you to inspect, modify, add, or delete FITS header keywords and values, providing a quick way to fix missing data like filters or exposure times.
- **Selection**: Check the boxes next to files to select them for Plate Solving, Calibration, or Analysis. Since v4.15 the tick state is kept when the table is refreshed (loading files, Refresh Headers, after calibration/plate solving) and is saved in the session.
- **Remove Highlighted**: removes the highlighted (selected) rows. If no row is highlighted, it offers to remove all checked files (with confirmation).
- **Header editor safety (v4.15)**: before the first save a backup `<file>.bak` is written next to the file; unchanged cards keep their exact value type (e.g. the string `'007'` stays a string), `nan`/`inf` are stored as text, and COMMENT/HISTORY cards can be added, edited and deleted.
- **Interactive Viewer**: Double-click any file in the list to open the **Interactive FITS Viewer** (see Section 8.1.1).

#### 8.1.1 Interactive FITS Viewer

The FITS Viewer is a powerful, integrated inspection and configuration tool. It opens as a separate window when you double-click a file in the File Manager.

**Fast start (v4.34).** The window opens at once (under a second for a 26-Mpixel frame); catalogue data are loaded in the background while the image can already be used - the status line says "Loading catalogue data in the background …" and later "Catalogue data loaded (local copy)". In the background:
- the names of the Light Curves target and comparison stars are resolved (local GCVS list first, see 3.6a), then these stars are marked;
- the reference-catalogue stars of the **whole frame** are read once from the **local catalogue copy** (e.g. ATLAS-RefCat2): every star that is marked or clicked is then identified locally, without any online query. Only if there is no local copy for the frame are stars identified online, one by one;
- the variable stars of the frame (red squares) come from the local VSX copy or the VSX cache, plus the stars of the local GCVS list; VSX is asked online only when needed.
- If a server does not answer (VizieR, SIMBAD, AAVSO), it is skipped for 10 minutes instead of waiting again for every request (timeout now 20 s instead of 60 s); a message in the console says so. This applies to all of Calibra.

For large frames the full view shows a 2×2 or 3×3 binned preview (same coordinates; measurements always use the full-resolution data); zoomed in, the full resolution is shown. Zooming and panning are therefore fast. Stars that cannot be fitted during the automatic marking are reported in the console instead of a pop-up.

**Layout**: The viewer consists of a central FITS image canvas flanked by two side panels:
- **Left Panel**: Live Inspection (click data), Radial Profile plot, and Variable Star info.
- **Right Panel**: Check & Reference Stars list, and Aperture Settings controls.

**Core Features**:

1.  **Star Inspection (Left-Click)**: Click any star to perform an instant Gaussian PSF fit. The "Live Inspection" panel displays the fitted coordinates (pixel and celestial), FWHM, instrumental magnitude, calibrated magnitude (using the current zero point), and a catalog cross-match if available.

2.  **Real-Time Radial Profile**: Every click also generates a **radial profile plot** showing physical pixel intensities (scatter points) against the Gaussian fit curve. Vertical lines mark the current **aperture radius** (red dashed) and **inner annulus** (green dotted), letting you visually verify whether the aperture captures the full stellar flux and whether the annulus samples clean sky.

3.  **Role-Based Star Marking (Right-Click & Hotkeys)**: Right-click a star to open a context menu, or hover the mouse cursor over a star and press a single-key hotkey to instantly assign roles:
    - **Mark as Variable Star** (Red) [Hotkey: `V`]: Designates the target variable. Only one variable star can be active at a time.
    - **Mark as Check Star** (Blue) [Hotkey: `C`]: Designates a check/validation star. Only one check star can be active.
    - **Mark as Reference Star** (Green) [Hotkey: `R`]: Adds the star to the reference ensemble (up to 5).
    - **Remove Star** [Hotkey: `Delete` or `Backspace`]: Removes any existing role assignment.

    Marked stars display persistent aperture and annulus overlays in their role color. Roles are mutually exclusive: reassigning a star to a different role automatically removes it from the previous one. Keyboard shortcuts completely bypass context menus, enabling fast comp sequence marking.

4.  **In-Viewer Aperture Controls**: The "Aperture Settings" panel (bottom-right) shows editable fields for **Aperture Radius**, **Annulus Inner**, and **Annulus Outer**. When "Flexible Aperture" is enabled in Settings, these fields auto-update based on the FWHM of each newly marked star. You can also manually edit these values; subsequent markings will use the updated radii.

5.  **Bidirectional Star Synchronization**:
    - **Import (automatic)**: When the viewer opens, any stars already configured in the Light Curves tab (target, check, and reference stars) are automatically pre-marked on the image with their correct role colors.
    - **Export**: Click **"Export Stars to LC Tab"** to push all current viewer markings back into the Light Curves tab. This automatically populates the star names, coordinates, and triggers catalog magnitude lookups (Mag and B-V) without requiring manual "Fetch" clicks.
    - **v4.16 – photometry and VSX markers**: magnitudes in the viewer (clicked star, marked stars, ensemble ZP) now use the same photutils aperture photometry as the pipeline; there is no silent fallback to the Gaussian-fit magnitude (shown separately as "PSF M"); stars whose aperture leaves the image get no magnitude. VSX variables are searched over the whole frame (radius = largest centre-to-corner distance; was capped at 45′). The image is held as 32-bit float (half the memory).
    - **v4.15 – no automatic export on close**: closing the viewer no longer exports anything. If the star selection was changed, you are asked whether to export it. When exporting, a star that is already configured in the Light Curves tab (within 3″) keeps its name, magnitude and B-V (e.g. AAVSO sequence values); placeholder names (`Target`, `Uncataloged…`) never overwrite existing names; NaN magnitudes are never written (the slot is fetched instead). Apertures are only changed with *Export Aps to Settings*.

6.  **Bidirectional Aperture Synchronization**: Click **"Export Aps to Settings"** to write the viewer's current aperture and annulus values back to the main Settings tab, ensuring the pipeline uses the same radii you visually verified.

7.  **Image Display**: The FITS rendering uses `ZScaleInterval` with the display floor set to the image median, producing a dark background with high star contrast. Zoom in/out with the scroll wheel; pan by clicking and dragging.

8.  **Variable Star Marking**: Known variable stars from the VSX catalog are plotted immediately as red squares using WCS coordinates. Clicking near a marker displays selectable details in a non-modal popup window.
    - **Min Amp (mag)** and **Max Mag (mag)**: Dynamic filters in the sidebar allow users to restrict markers based on amplitude or magnitude thresholds.

### 8.2 Processing Tabs
1.  **About**: Version information and a summary of Calibra's capabilities.
2.  **Pre-processing**: Subdivided into:
    - **FITS Calibration**: Standard Bias and Flat field correction.
    - **WCS Plate Solving**: Command-line integration with ASTAP to append celestial coordinates.
    - **OSC Channel Split**: One-Shot Color Bayer demultiplexing (Superpixel & Bilinear).
3.  **🔍 Analysis & Calibration**: A unified top-to-bottom workflow for all primary photometry tasks:
    - **1. Detection & Zero-Point**: Runs the full star detection and zero-point calibration pipeline.
    - **2. Color Coefficients**: Computes transformation coefficients from paired B/V images.
    - **3. Differential Photometry**: Applies transformation coefficients to compute standard magnitudes relative to a reference star.
4.  **Time Series**: Time-series photometry for variable star analysis:
    - **FITS Sequence Selection**: Glob pattern for a sequence of images.
    - **Ensemble Reference Stars**: Up to 5 comparison stars, resolved via SIMBAD.
    - **Target Star**: The variable star to measure.
    - **Check Star**: One ensemble star designated as a check star for independent validation.
    - **Light Curve Mode** (v4.14): Calibrated magnitudes, Relative flux (reference-star ADU only) or Raw ADU (no reference/check stars needed). See Section 6.6.
    - **Generate Light Curve**: Outputs a detailed CSV and a light curve plot, plus an AAVSO Extended Format report in Calibrated mode.
5.  **Data Mining**: Visualizations, statistics, and table results for automated variable star discovery.
6.  **⚙ Settings**: Grid configuration sub-tabs:
    - **Astro & Catalog**: Reference catalog selection, region filtering, zero point limits.
    - **Camera & Photometry**: Gain, read noise, saturation limits, detection thresholds, and aperture sizes.
    - **Operator & Session**: AAVSO observer codes, session directories, and logger options.
7.  **Help**: Quick markdown links to the README and this User Manual.

---

## 9. Understanding the Output

### 9.1 Results CSV (`output/` in the user folder)
The primary output for every image. Key columns include:
- `refined_x` / `refined_y`: The high-precision sub-pixel coordinates.
- `ra_hms` / `dec_dms`: Celestial coordinates from the WCS header.
- `net_flux`: Background-subtracted ADU counts.
- `mag_calibrated`: The final, zero-point corrected true magnitude.
- `mag_calibrated_err`: The ± uncertainty of the final magnitude (instrumental error ⊕ error of the zero point, v4.16).
- `peak_raw`, `saturated`, `blended`: quality flags (v4.16) – highest raw pixel in the aperture, saturation flag from the PSF stage, another detection within the outer annulus radius.
- `n_sky`, `bkg_median`, `bkg_std`: annulus pixels used and the sky level/scatter (v4.16).
- `is_variable`: A "Yes/No" flag indicating if the star matched a record in the AAVSO VSX catalog.
- `airmass`: The atmospheric airmass calculated from the FITS header.

### 9.2 Diagnostic Plots (`output/plots/` in the user folder)
If enabled, the pipeline saves a four-panel graphic for each star showing:
1.  **Raw Data**: The original pixel cutout.
2.  **Gaussian Model**: The idealized mathematical fit.
3.  **Residuals**: The difference (should look like random noise if the fit is good).
4.  **Radial Profile**: A 1D cross-section showing pixel intensity vs. distance from the center.

---

## 10. Troubleshooting & Tips
- **No Stars Found?** Verify your `Detection Sigma`. Lower it (e.g., to 3.0) for faint targets or increase it (e.g., to 10.0) for crowded fields.
- **Calibration Failures?** 
    - Check if the FITS header has valid `RA`/`DEC` keywords for online queries.
    - If using online catalogs, ensure you have an active internet connection for the first run (subsequent runs use the local cache).
    - Verify that your `Min SNR for Calib` is not set so high that no stars are matched.
- **Positional Drift?** Check the **Shift Analysis** console summary. Large consistent shifts indicate a mount tracking issue or an inaccurate WCS header.
- **Slow Performance?** Check if `Print Detailed Calibration` is on; for batch processing, turning this off keeps the console clean and fast.
- **Realistic error bars for stacks (v4.16)**: set *Stacked images* = `average` (default) and enter the *Blank-sky noise check* factor printed in the pipeline log as *Sky noise factor*. Note that results computed with v4.15 or earlier have errors that are too large (up to ~5× for 49-frame stacks); light-curve measurements cached by older versions are not reused.
- **No good comparison stars / just want a quick look?** Use the *Raw ADU* light-curve mode (Section 6.6.1) — or simply leave all reference stars unticked and Calibra switches to it automatically. Switch to *Relative flux* (6.6.2) as soon as you have one or more stable stars in the field to remove transparency changes.
- **HJD on cached re-runs:** before v4.14, re-running a light curve from the photometry cache on FITS files without `SITELAT`/`SITELONG` keywords silently reported JD (mid-exposure) instead of HJD. This is fixed; if you submitted AAVSO data from such a cached re-run, you may want to regenerate the report.
- **HJD of MWT frames (fixed in v4.41):** MaxIm/ACP write the site as text (`SITELAT = '38 27 33'`, `SITELONG = '-01 04 57'`). Before v4.41 Calibra could not read these values, the HJD calculation failed (console: *Could not calculate HJD*) and the plain JD at mid-exposure was written as "HJD". For HU Aqr in November this is 28 s too late (the correction varies over the year by up to ±8.3 min, depending on the star's position relative to the Sun). Light curves of MWT frames made before v4.41 should be generated again. How Calibra computes the time: JD of the exposure start (`JD` or `DATE-OBS`, UTC) + half the exposure time, heliocentric light-travel time to the target (astropy) added - HJD (UTC) at mid-exposure. The MWT headers also contain `HJD-OBS` (ACP: HJD at the *start* of the exposure - agrees with Calibra to 0.01 s after adding half the exposure), `BJD-OBS` and MaxIm's `JD-HELIO`; `JD-HELIO` is about 5.7 min off and should not be used.

## References

**ASTAP (plate solving)**
https://www.hnsky.org/astap.html

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
