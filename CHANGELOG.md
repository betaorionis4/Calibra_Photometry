# Calibra Changelog

All notable code changes are listed here, newest first. Earlier history (v3.0–v4.13) is summarised in the "What's New" sections of `README.md`.

---

## [4.44] — 2026-11-14 — Faint targets: forced photometry and combining frames

Originals of changed files are in `../Calibra_backups/backup_v4.43/`.

### Added
- **Faint Target** box in the Light Curves tab (all three light-curve modes). Option *Faint target*: the target is not centroided; the aperture sits at its WCS position, corrected in every frame by the mean offset (centroid - WCS) of the reference stars of the Light Curves tab. One aperture radius for all frames and all stars (default 1.5 x median FWHM of the frames, from the reference stars; 0 = radius from Settings). The check star is measured the same way as the target.
- **Combining frames** without re-measuring: *Combine in groups of X consecutive frames* (+ *Apply Groups*; a group never spans a gap in time such as another night), *Combine Selected Rows* (any rows marked in the result table), *Split Selected Rows*, *Single Frames*. A combined point = sum of the fluxes of every star over its frames (equivalent to an aligned stack, without resampling); its time is the exposure-weighted mean, the horizontal bar in the plot shows the time span.
- **Upper limits**: below a chosen S/N (default 3) a point becomes "fainter than" (flux + 3 sigma): table `<18.13`, flag `LIMIT`, plot symbol ▽, AAVSO report `<mag` with MERR `na`.
- New module `photometry/faint_target.py` (`measure_frames`, `combine`, `groups_consecutive`, `merge_groups`, `split_groups`).
- CSV columns `n_frames`, `t_span_min`, `upper_limit`, `mag_limit`, `wcs_dx`, `wcs_dy`; table column *N* (frames per point). Output files get `_faint` in the name.

### Changed
- Light Curves: saving/display of the results moved into one function (`ts_publish`) used by normal runs and by faint-target re-combinations.
- AAVSO report: rows without a valid magnitude are skipped; airmass `na` if unknown; combined points get "N frames combined" in NOTES.

### Tests
- `test_faint_target_forced_photometry`, `test_faint_target_limits_and_groups` (synthetic frames with a WCS error; 62 tests). Checked on the HU Aqr frames of 13 Nov (MWT, 8 x 60 s): reference-star offset -0.31/-0.78 px (scatter 0.07/0.04 px), S/N 1-2 per frame, 4.3 for all 8 frames.

---

## [4.43] — 2026-11-14 — UT date in light curves

Originals of changed files are in `../Calibra_backups/backup_v4.42/`.

### Changed
- Light-curve output: new column **`date_ut`** (UT date of mid-exposure, YYYY-MM-DD) directly before `ut_mid` in the Light Curves and Field Variables CSV files. The *UT (mid)* column of the Light Curves table now shows date and time (`2026-11-13 21:20:01`). A mid-exposure time that rounds to 00:00:00 gets the date of the new day.

### Tests
- `test_light_curve_ut_date_column` (60 tests).

---

## [4.42] — 2026-11-14 — UT of mid-exposure in light curves

Originals of changed files are in `../Calibra_backups/backup_v4.41/`.

### Changed
- Light-curve output (Light Curves table and CSV, Field Variables CSV): the JD column is replaced by **UT of mid-exposure as HH:MM:SS** (`ut_mid`, table heading *UT (mid)*); HJD follows as before. JD is still used internally (file names, night labels). Checked on HU Aqr (13 Nov 2026): 21:19:30.69 UTC start + 30 s -> 21:20:01, HJD 2461358.38857.

### Tests
- 1 new test (UT formatting, CSV columns): 59 tests pass.

---

## [4.41] — 2026-11-14 — HJD for MWT frames

Originals of changed files are in `../Calibra_backups/backup_v4.40/`.

### Fixed
- **HJD with sexagesimal site coordinates**: MaxIm/ACP headers (MWT) give `SITELAT = '38 27 33'`, `SITELONG = '-01 04 57'`. `get_hjd` converted them with `float()`, which failed; the heliocentric correction was then dropped and the JD at mid-exposure was written as HJD (HU Aqr, 13 Nov 2026: 28 s too late). Site values are now read as numbers or sexagesimal text, with `LAT-OBS`/`LONG-OBS`/`ALT-OBS` as alternatives; an unreadable site falls back to the geocentre (< 0.03 s effect) instead of dropping the correction.
- Checked against the header of HU Aqr (13 Nov 2026): Calibra's HJD (mid-exposure) = ACP's `HJD-OBS` (start) + 30 s within 0.01 s. MaxIm's `JD-HELIO` in the same header is 5.7 min off (not used by Calibra).

### Tests
- 1 new test (sexagesimal site, comparison with ACP HJD-OBS, size of the correction, unreadable site): 58 tests pass.

---

## [4.40] — 2026-11-08 — azimuth without compass letters

Originals of changed files are in `../Calibra_backups/backup_v4.39/`.

### Changed
- Observation Planner: azimuth in the table (and in the priority-list notes) in degrees only, without compass letters; narrower column.

---

## [4.39] — 2026-11-08 — planner refinements, horizontal scrolling

Originals of changed files are in `../Calibra_backups/backup_v4.38/`.

### Changed
- Observation Planner:
  - Azimuth range entered as *from … to …* degrees (0 = N, 90 = E; through North e.g. 300 to 60); the named sectors fill the fields.
  - Table: new columns RA, Dec (J2000), azimuth at the best time in degrees (with compass direction), meridian passage (`*` outside the window); altitude and azimuth in separate columns; arrow on the sorted column.
  - Sorting by name: constellation first, then GCVS order of the designations (Bayer, R…Z, RR…ZZ, AA…QZ, V numbers); names without constellation last.
  - Altitude chart: dashed vertical line at the meridian passage when one star is shown; hour ticks on full hours.
  - Layout: selection fields in three compact lines, long labels wrapped, buttons in two rows.
- All scrollable tabs: horizontal scrollbar (and Shift + mouse wheel) when the window is narrower than the content - nothing is cut off on small screens.

### Tests
- 1 new test (constellation sort, meridian passage = highest point, time); GUI tests incl. screenshots at 1280x720. 57 tests pass.

---

## [4.38] — 2026-11-08 — night plan for NINA

Originals of changed files are in `../Calibra_backups/backup_v4.37/`.

### Added
- **Night Plan for NINA…** in the Observation Planner (new module `photometry/nina_export.py`): plan for the selected targets and export as NINA simple-sequencer target set (`ArrayOfCaptureSequenceList`, same structure as NINA's own files) plus a schedule CSV.
  - Exposure time per target from the standard exposures of the setup: no saturation at maximum light (70 % of the limit; Gaussian star with the detection FWHM), at most period/50; shorter times for bright stars; S/N at maximum and minimum light. Zero point measured (setup) or estimated from the aperture; sky brightness and seeing from the setup.
  - Duration: predicted primary minimum of eclipsing binaries (GCVS epoch and period), one cycle for short periods, 2 h for 0.5-1 d, a few exposures for long periods / Miras / semiregulars, 2 h for cluster fields.
  - Schedule within each target's observable time; eclipses at their time, series near the best time, snapshots in gaps; blocks moved together for the sequential simple sequencer; remaining gaps written as *Delay*; waiting times reported.
  - Dialog: filters (several in Rotate mode), overheads, snapshot frames, sky; Slew/Center/Guiding/Dither; edit exposure/frames or leave out targets.
- Instrument setups: planning fields *ZP V (e-/s)*, *sky brightness*, *seeing*; button **Filters from NINA File…** (filter names and wheel positions from an existing NINA sequence).

### Tests
- 1 new test (exposure choice incl. saturation and time resolution, measured ZP, eclipse prediction, requests, schedule without overlaps centred on the minimum, XML structure, filters from a NINA file); generated XML compared with Stephan's NINA target set (identical element structure); GUI test (plan of 4 targets, file and CSV written). 56 tests pass.

---

## [4.37] — 2026-11-08 — observation planner

Originals of changed files are in `../Calibra_backups/backup_v4.36/`.

### Added
- **Observation Planner** (new sidebar tab; modules `photometry/planner.py`, `photometry/planner_tab.py`): targets for one night from the local GCVS list, a priority list or a list of open clusters; site, time zone and magnitude limits from an instrument setup (also non-active ones, e.g. MWT); date, time window (`dusk`/`dawn` or local times, over midnight), darkness (Sun below -6/-12/-18 deg), minimum altitude, azimuth sector (named or degrees), Moon distance, minimum observable hours, magnitude range, GCVS types (wildcards), period and amplitude range. Result table (observable span, best time, altitude/direction, hours, Moon distance; sortable), altitude chart for the night with Moon and darkness.
- Priority list `photometry_refstars/priority_targets.csv` (add/remove from the planner, editable) with status per target: observable in the window / above the limit elsewhere / not observable.
- Open-cluster list `photometry_refstars/open_clusters.csv` (IC 1805, NGC 1912, 2355, Pleiades, 6834, 188, 869, 884, 457, 663; editable) with the known variables of each cluster from the local GCVS list.
- Buttons: Set as Light Curves Target, Add to Field Variables, Save List (CSV).
- All computed locally: Sun/Moon low-precision formulas, sidereal time, precession (checked against astropy: altitudes within 0.01 deg).
- About tab: date of the latest update from `APP_DATE` (updated with every version).

### Tests
- 1 new test (time window, sectors, type/magnitude/period filters, observability, altitude vs. astropy, twilight incl. no darkness in June, priority list, cluster file); GUI test of the tab (GCVS list, east sector, priority list, clusters, Light Curves target). 55 tests pass.

---

## [4.36] — 2026-11-08 — instrument setups

Originals of changed files are in `../Calibra_backups/backup_v4.35/`.

### Added
- **Instrument setups** (new module `photometry/instruments.py`, files in `instruments/`): telescope, aperture, focal length, reducer, camera, pixel size, sensor size, binning, OSC, capture gain/offset; site (name, lat, lon, elevation, time zone); filter names in the FITS header and filter-wheel positions (for NINA); planning data (standard exposure times, magnitude limits); a snapshot of all instrument-dependent Calibra settings (60 keys: CCD, detection, apertures, zero points, colour/extinction coefficients, catalogue radius, plate-solving field, pixel region, calibration folders and masters).
- Sidebar: *Instrument setup* selector with summary line (pixel scale, field, site) and *Manage Setups…* (new from current settings, duplicate, delete, rename, fill from FITS header, save and activate).
- Switching stores the current settings in the old setup and applies those of the new one; the active setup is updated on exit as well. Site coordinates and filter names stay in step between the setup dialog and the Settings tab.
- Frames loaded into the File Manager are compared with all setups (sensor size, focal length / pixel scale, pixel size, telescope/camera); a note in the sidebar offers to switch if they fit another setup better.
- First start: setup *Standard* created from the existing settings.

### Changed
- Light Curves: *Save/Open Setup…* renamed *Save/Open Target Setup…* (per variable star; not to be confused with instrument setups).

### Tests
- 1 new test (store/load, snapshot and site/filter sync, pixel scale/field, header values incl. binning and sexagesimal latitude, recognising frames, rename/delete); GUI test: Standard created, new setup MWT filled and activated, saturation limit kept per setup across switches, MWT frame loaded -> switch note -> switched. 54 tests pass.

---

## [4.35] — 2026-11-08 — stored AAVSO sequences

Originals of changed files are in `../Calibra_backups/backup_v4.34/`.

### Added
- New module `photometry/aavso_sequences.py`: every AAVSO comparison-star sequence fetched with *Get AAVSO Ref Stars* is stored per variable in `photometry_refstars/aavso_sequences/<name>.json` (stars, AAVSO V/B, catalogue cross-match, field size, download date). It is reused without a query - also after switching targets or restarting; names matched loosely.
- Sequence window: "Stored sequence, fetched …" and **Refresh from AAVSO**; status line shows the source; note for sequences older than a year.
- New query only if the field is now larger than the stored one; a changed reference catalogue only redoes (and stores) the cross-match.
- FITS Viewer: gets the stored sequence of the current Light Curves target (before: only the last fetched one).

### Changed
- Cross-match of the sequence with the reference catalogue moved into one helper (`aavso_crossmatch`).

### Tests
- 1 new test (store/load, loose names, cross-match update keeps the date); GUI test: HU Aqr -> SS Cyg -> hu aqr needs only two AAVSO queries, Refresh makes a third. 53 tests pass.

---

## [4.34] — 2026-11-07 — fast FITS Viewer, no waiting for unreachable servers

Originals of changed files are in `../Calibra_backups/backup_v4.33/`.

### Why the viewer was slow
- Before the window appeared, the names of the Light Curves target and comparison stars were resolved online (SIMBAD), one after another, in the GUI thread.
- During start-up the VSX variables of the frame were queried online (VizieR), and each automatically marked star triggered further online queries (catalogue, Gaia cross-match, VSX) - all in the GUI thread, each with a 60 s timeout when a server did not answer.
- The 26-Mpixel image was colour-mapped at full resolution for every draw (about 1 s) and drawn twice at start.

### Changed
- **FITS Viewer**: window usable at once (measured 0.8 s for a 6248x4176 frame instead of 3.8 s without any network, plus all online waits). Names, catalogue data and variables are loaded in background threads; the initial stars are marked when the data are there.
- **Local data first, no per-star queries**: the reference-catalogue stars of the whole frame are read once from the local copy; identification of marked/clicked stars is a local look-up. Variables from the local VSX copy / VSX cache plus the local GCVS list; one VSX query for the frame only if needed. Per-star online identification only when no local copy covers the frame (then done in the background for the initial stars).
- **Drawing**: binned preview for the full view of large frames, full resolution when zoomed in; scaling after resampling (`interpolation_stage='data'`); one draw at start. Zoom/pan redraw 0.1-0.2 s instead of 0.6-1 s.
- **Unreachable servers** (new module `photometry/online.py`): after a failed request VizieR / SIMBAD / AAVSO are skipped for 10 min (message once); remote timeout 20 s instead of 60 s. Used by catalogue queries, Gaia cross-match, VSX, name resolution and AUID look-up everywhere in Calibra.
- Automatic marking: stars that cannot be fitted are reported in the console instead of a pop-up.

### Tests
- 1 new test (VizieR failure -> later requests skipped, pause expires); name test extended; per-test reset of the service state. GUI tests: viewer open from Light Curves with SIMBAD simulated offline (window after 0.85 s), local ATLAS identification, preview/full-resolution switching. 52 tests pass.

---

## [4.33] — 2026-11-07 — calibration folders decide the frame type

Originals of changed files are in `../Calibra_backups/backup_v4.32/`.

### Changed
- Bias / flat / dark folder: a file whose header names another frame type (e.g. `IMAGETYP = Light Frame`) is now **used** as the type of its folder, with a warning in the console and in the Master Frames status line (v4.29–4.32 left such files out). The folder chosen by the user decides. The mixed calibration folder still sorts by header/names.

### Tests
- Master-frame test adapted (bias with 'Light Frame' header is combined, warning issued): 51 tests pass.

---

## [4.32] — 2026-11-07 — APASS DR10 download from Calibra

Originals of changed files are in `../Calibra_backups/backup_v4.31/`.

### Added
- New module `photometry/apass_dr10.py`: download of APASS DR10 fields from the AAVSO archive (the form of archive.aavso.org/download-apass-data, CSV output), saved unchanged in the local APASS folder (created if needed) and registered there; record of downloaded fields (no second download of a covered field); field(s) and radius from plate-solved images (one per pointing).
- **Download APASS DR10…** dialog (Settings → local catalogues, Light Curves, Field Variables): checked images, star name or RA/Dec.
- **Automatic DR10 download** (Settings option, default off): `fetch_online_catalog` downloads a missing APASS field before falling back to VizieR DR9; at least 10 s between automatic downloads, 30 min pause after a failure.
- `local_catalogs.register_folder()`, option `apass_auto` in `local_catalogs.json`.

### Tests
- 1 new test (server answer simulated: file name, parameters, reuse, automatic download in the catalogue query, pause after failure, error page, fields from images); GUI test of the dialog. 51 tests pass. The real request to archive.aavso.org could not be made from the development sandbox (the request format was checked with the web form).

---

## [4.31] — 2026-11-07 — APASS DR10 downloads

Originals of changed files are in `../Calibra_backups/backup_v4.30/`.

### Added
- **APASS DR10 CSV files** from archive.aavso.org/download-apass-data can be put unchanged into the APASS folder of the local catalogues. Column names with a band in parentheses (`Johnson_V (V)`, `Johnson_B (B)`, `raerr(")` …) are now recognised (`local_catalogs._norm_col`, also for the catalogue detection by content and `tools/catalog_overview.py`); `NA` values are skipped.
- Overlapping downloads: duplicate stars from several table files are removed when a field is queried.

### Changed
- Settings status shows "APASS (DR9/DR10)".

### Tests
- 1 new test with the real DR10 CSV layout (columns, NA values, overlapping files, detection by content): 50 tests pass.

---

## [4.30] — 2026-11-07 — star names from the local GCVS list

Originals of changed files are in `../Calibra_backups/backup_v4.29/`.

### Added
- **Name look-up in the local GCVS list first** (default when a GCVS list is in the local catalogue folder and *Use the local copies* is on), then SIMBAD, then AAVSO VSX. Used everywhere a name can be entered: target / comparison / check stars, Light Curves, Differential Photometry, Field Variables target lists with names only, plate-solving position guess. Works offline. Spelling variants (case, spaces, underscores) are matched. New module `photometry/names.py` (`from_name`, drop-in for `SkyCoord.from_name`); `local_catalogs.gcvs_lookup()`, `name_key()`.
- *Check* buttons show where a name was found ("local GCVS list", "SIMBAD", "AAVSO VSX").
- With *Go online* switched off, an unknown name gives a clear message instead of a network error.

### Tests
- 1 new test (offline look-up, spelling variants, order local -> online, online switched off, Field Variables names): 49 tests pass.

---

## [4.29] — 2026-11-07 — bias and flat folders

Originals of changed files are in `../Calibra_backups/backup_v4.28/`.

### Added
- *Pre-processing → Master Frames*: **Bias folder**, **Flat folder** and optional **Dark folder** (each with its own Browse button). Every FITS file in such a folder is taken as a frame of that type - file names and `IMAGETYP` are not needed. Files whose header says they are something else (e.g. `IMAGETYP = Light Frame`) are left out and reported. The masters are written into the same folder. The mixed calibration folder (type from header/names, v4.27) remains as an alternative.
- Flat filter from the file or folder name when the header has no `FILTER` (`flat_V_001`, `Flat-Lum-3`, `FLAT_Bmag`, `.../B/...`).
- Help text "What does this do?": how bias and flats are combined without double counting.

### Changed
- The fingerprint (`CALHASH`) of a master flat now includes the fingerprint of the subtracted master bias/dark: a rebuilt bias also rebuilds the flats (v4.27 flats are rebuilt once).
- A master dark with the flats' exposure time can correct the flats (like a flat-dark).
- A master flat made by other software that is *not* bias-subtracted (checkbox unticked) is now corrected in automatic mode too (before only in manual mode). Calibra's own flats are recognised (`FLATNORM`) and never corrected twice.
- Masters are looked for in all configured calibration folders.

### Tests
- 1 new test (type folders without IMAGETYP, stray light frame, filter from name, flat sky after calibration, rebuild chain bias -> flats): 48 tests pass.

---

## [4.28] — 2026-11-07 — catalogue overview tool

### Added
- **`tools/catalog_overview.py`** and *Settings → Which files can be deleted?…* (read-only): lists every file of the local catalogue copies with size, number of stars, Dec and magnitude range and a verdict (keep / DELETE: all Dec < limit / DELETE: all fainter than limit / partly useful), with a summary of the disk space that could be freed and a CSV report. ATLAS-RefCat2 is judged from folder and tile names alone. `--reduce` writes reduced copies of table files (Gaia, APASS …) with only the rows inside the limits. Manual section 3.6a.

### Tests
- 1 new test: 47 tests pass.

---

## [4.27] — 2026-11-07 — master frames, automatic calibration, on the fly

Originals of changed files are in `../Calibra_backups/backup_v4.26/`.

### Review of the previous calibration (findings that are fixed now)
- L frames were not calibrated at all (only V and B flats existed); the pipeline used the V flat for any other filter.
- Master bias and flat were read and normalised again for every light frame.
- Every run wrote all calibrated files again (32-bit float, 2× the raw size), even when they existed with the same masters; output went to `fitsfiles/calibrated/` in the Calibra folder.
- No darks, no check of offset/temperature, saturation not preserved after flat division, no protection against calibrating a calibrated frame twice.

### Added
- **Master frames** (new module `photometry/master_frames.py`, *Pre-processing → Master Frames*): choose the folder of raw calibration frames (subfolders per type or mixed) → *Build / Update Masters*: frames sorted by `IMAGETYP`/`FRAME` or file/folder name (bias/offset, dark, flat, flat-dark), grouped by size, binning, gain, offset, filter (flats), exposure and temperature (darks); sigma-clipped mean (robust MAD clipping) or median, computed in strips (~1 s per 26-Mpixel frame, ~200 MB memory); flats corrected with master flat-dark or bias, normalised per frame, master normalised to 1; written as float32 `master_<type>_…fits` into the folder of the raw frames with `NCOMBINE`, method, file list and input fingerprint (`CALHASH`). Up-to-date masters are reused; changed/added frames trigger a rebuild.
- **Automatic selection** of the matching masters for every light frame (size and binning required; gain/offset preferred; flat by filter V/B/L/any), *Show Masters for Checked Frames*; manual mode with bias, flats V/B/L and optional dark. Masters by other software (name `master*` or `IMAGETYP` "master") are used too; 0–1 normalised masters are rescaled.
- **Optional darks** (*Use darks*): matching exposure (±5 %) and temperature (±3 °C), else scaled from the longest dark with bias.
- **Calibration on the fly** (*Calibrate on the fly when measuring*): Light Curves, Field Variables, Data Mining and the pipeline calibrate in memory (masters cached, ~0.2 s per 26-Mpixel frame); cache keys include the calibration state.
- **Write Calibrated Files**: into `calibrated/` next to the raw files; existing files made with the same masters are reused (`CALTAG`); header `BIASFILE`, `DARKFILE`, `FLATFILE`, `SATURATE`.
- Saturated raw pixels keep a value ≥ the saturation limit after calibration; frames already calibrated (`cal_`, `BIASFILE`/`FLATFILE`/`DARKFILE`/`CALSTAT`, calibration `HISTORY`) are skipped; stacks get a note that calibration after stacking is approximate.
- Settings are stored in `calibration_config.json` (also used by the pipeline without GUI) and in the session.

### Tests
- 2 new tests (building/reusing/rebuilding masters from MaxIm-style frames incl. scaled integer data; on-the-fly and file calibration with flat-field, optional dark, skip of calibrated frames, missing flat): 46 tests pass. GUI test of building masters, showing the selection and writing files.

---

## [4.26] — 2026-11-07 — local catalogue copies, local GCVS list

Originals of changed files are in `../Calibra_backups/backup_v4.25/`.

### Added
- **Local catalogue copies** (new module `photometry/local_catalogs.py`): *Settings → Astro & Catalog → Local catalogue folder* (*Browse Folder…* chooses the folder that holds the catalogues; *Rescan*). The subfolders are recognised automatically by name or by their column names: ATLAS-RefCat2 (MAST tile files `RRR±DD[.rc2|.csv][.gz]` in magnitude folders, integer-scaled or plain, with or without header), Gaia DR3, APASS DR9, VSX, Landolt (CSV/TSV tables with header, `#` comments, `.gz`), and a GCVS list (file name containing `gcvs`). The folder and the detection result are stored in `photometry_refstars/local_catalogs.json` and used at the next start (also by the pipeline without GUI); the folder is also part of the session.
- All catalogue queries go to the local copy first: `fetch_online_catalog` (pipeline, colour transformation, Differential Photometry, Light Curves, Field Variables, Data Mining, FITS Viewer), the Gaia cross-match and `fetch_vsx_catalog`. Same B/V conversions as for VizieR (`atlas_to_vb`, `gaia_to_vb`). VizieR only if the catalogue is not local or the field is not covered; options *Use the local copies* and *Go online for … not available locally*.
- Table catalogues are indexed once by sky area (`photometry_refstars/cache/local_index_<cat>.json`), so a query reads only the relevant files; RefCat2 tiles are opened directly by name.
- **Field Variables + local GCVS list**: *GCVS (local list / online)* uses the local list; new option *Add variables from the local GCVS list automatically* (default on): when the tab is opened with a new set of checked images, the GCVS stars in their field (filters applied) are added.

### Tests
- New test with a synthetic local catalogue folder (RefCat2 tiles incl. a skipped 18_m_19 folder, Gaia table, GCVS list; online access blocked): 44 tests pass. GUI test: settings detection, automatic GCVS additions, Field Variables run with comparison stars from the local ATLAS copy while VizieR was unreachable.

---

## [4.25] — 2026-11-07 — phase plots: navigation, editing, saving

Originals of changed files are in `../Calibra_backups/backup_v4.24/`.

### Changed
- Period Analysis: the **bottom-plot selector** (Phase plot / Residuals vs time / O–C diagram) is now a row of buttons directly above the plots. Selecting a frequency in the table or clicking into the periodogram switches back to the phase plot (before, after the O–C diagram the way back was only via the selector in section 2, easy to miss).

### Added
- **Phase Plot Window (edit / save)…** (button above the plots): a separate phase plot of any fitted frequency (or the clicked preview frequency) with editable title, axis labels, phase range (e.g. −0.5 … 1.5), y range, point size, font size, error bars, one colour or a colour per file, inverted y axis, grid, legend, curve (fit / binned mean with selectable number of bins / none) and **phase 0 at maximum light** (ephemeris from the fitted shape). Values are in the original unit (data minus all other, non-harmonic frequencies). *Save Figure* writes PNG (selectable size and dpi), PDF or SVG; *Save Phase Data (CSV)* writes HJD, phase, value, error, file and model for use in other programs. The matplotlib toolbar (zoom, pan) is included.

### Tests
- 1 new test (harmonic family): 43 tests pass. GUI test: way back from the O–C diagram, phase-plot window with edits, PNG/PDF/CSV export.

---

## [4.24] — 2026-11-06 — period analysis: fit quality, O–C, guide

Originals of changed files are in `../Calibra_backups/backup_v4.23/`.

### Added
- **Fit quality** below the frequency table: residual rms vs. median photometric error, reduced χ², a verdict, and the night-to-night scatter of the mean residual (`period_analysis.fit_quality`).
- **Bottom plot selector**: phase plot, **residuals vs time** (with night means), **O–C diagram**.
- **Times of maximum and O–C** (new section 4 of the tab; `times_of_maximum`, `ephemeris`, `oc_analysis`): template fit of the fitted light-curve shape (or parabola) per covered cycle, after subtracting the other non-harmonic frequencies; O–C against the ephemeris from the fit or a given T₀/P; corrected ephemeris with errors; optional parabola → dP/dE, dP/dt, (1/P) dP/dt; *Save Maxima (CSV)…*.
- **Period-ratio note** for independent frequencies (radial double-mode ratios of RRd, δ Sct/HADS, Cepheids).
- **`period_analysis_guide.md`**: practical guide to measuring periods, judging whether one period suffices, identifying additional periods, times of maximum/O–C, notes for other variable types, pitfalls, reporting, formulas and literature.
- **`example_HADS_simulated.csv`** (data folder): simulated multi-night light curve with AE UMa-like parameters for practising (clearly marked as simulated in its header).

### Changed
- Notes: harmonics recognised up to 6f; combination frequencies are only formed from independent frequencies (no more `f1+f2` for 3f1 when f2 = 2f1).
- Light-curve CSV files may contain comment lines starting with `#`.

### Tests
- 2 new tests (fit quality and period ratio; times of maximum, O–C line and parabola with an injected period change): 42 tests pass. GUI test of the full workflow with the example file.

---

## [4.23] — 2026-11-06 — L filter, GCVS/SAAF lists, phase-plot curve

Originals of changed files are in `backup_v4.22/`.

### Added
- **L (luminance) light curves**: `filters.classify_band()` recognises L (Settings: new *L-Filter Keyword*, default `L`; also `Lum`, `Luminance`, `Lmag`, `Baader L`, `UV/IR-cut`). Light Curves and Field Variables offer the filter `L`; the File Manager status line counts L files. Calibrated L magnitudes use the V magnitudes of the comparison stars, AAVSO reports get the filter code `CV`, no colour transformation (coefficient 0), extinction k_V; a note in the Light Curves tab explains this. `classify_filter()` (B/V only, used by calibration and colour work) is unchanged.
- **Target lists from GCVS searches (SAAF variable-star section)**: Swedish and more English column names (Namn, Typ, Dekl, Epok, Rektascension, Declination, …), columns in any order, extra columns ignored, Windows-1252 encoding, GCVS amplitudes in parentheses, epoch kept (also in *Save List…*).
- **Minimum content check**: a list with a header but without RA or Dec is refused with a clear message (required: name, RA, Dec; found columns listed); without a name column the stars are numbered and the user is reminded; rows with missing coordinates are skipped and reported.
- **Period Analysis - curve in the phase plot** selectable: *Fit (sine + harmonics)*, *Binned mean (no model)* (median per phase bin, shows the real shape, e.g. the sawtooth of δ Scuti stars) or *None*. The legend states how many harmonics the fit has.

### Changed
- Period Analysis: the phase plot of a frequency now keeps the frequencies that are its harmonics (2f, 3f, …, e.g. added by prewhitening) instead of subtracting them, so the true light-curve shape is shown; the fitted curve includes them.

### Fixed
- Light Curves → *Get AAVSO Ref Stars* → import set the **V** magnitudes of the comparison stars also when the light-curve filter was **B**; B light curves now get the B magnitudes.
- Light-curve CSV files of the raw and calibrated modes contained an internal column `_ens_members` (since v4.20); removed.

### Tests
- 3 new tests (SAAF GCVS export incl. amplitude and missing coordinates, required columns / name reminder, L classification): 40 tests pass. GUI tests: Field Variables with L images and the GCVS.csv list; Period Analysis with the binned curve and harmonics kept in the phase plot.

---

## [4.22] — 2026-11-05 — position guess for plate solving

Originals of changed files are in `backup_v4.21/`.

### Added
- **Pre-processing → WCS Plate Solving → Position Guess**: the starting position for ASTAP can come from the FITS header (as before), from an **object name** (SIMBAD, then AAVSO VSX; *Look up* shows the coordinates) or from **RA/Dec typed in** (sexagesimal hours/degrees or decimal degrees). It is passed to ASTAP as `-ra <hours> -spd <Dec+90>` with the search radius from the Solver Settings and overrides the header.
- **Series**: *Use the solution of each image as the guess for the next image* (default on) - the centre of every solved image is the guess for the next one. If an attempt fails, the user's guess, then the header position, then (optional, *Allow blind solving*) a blind whole-sky solve are tried. The console states which guess solved each image; the status line shows progress.
- *Field height (deg)* setting, passed to ASTAP as `-fov` (0 = automatic).
- `plate_solve.header_position()` understands `RA`/`DEC` (degrees or text), `OBJCTRA`/`OBJCTDEC`, `RA_OBJ`/`DEC_OBJ` and `CRVAL1/2`; `plate_solve.solve_series()` implements the chained solving (also used by `plate_solve_files`).

### Tests
- 3 new tests (header positions, ASTAP command line with guess / blind / blind switched off, chained series with fallback to the user's guess): 37 tests pass. GUI test with three images without coordinates and a manual RA/Dec guess (ASTAP simulated).

---

## [4.21] — 2026-11-05 — Period Analysis

Originals of changed files are in `backup_v4.20/`. Photometry results are unchanged.

### Added
- **New sidebar tab *Period Analysis*** (new module `photometry/period_analysis.py`):
  - **Input**: one or several light-curve CSV files (Calibra output of Light Curves / Field Variables, or any CSV with a time column `hjd`/`bjd`/`jd`/`mjd`/`time` and a value column; separators `, ; tab`), choice of the quantity (mag, flux, …), only OK points (flag), optional subtraction of each file's mean (different zero points). *Load Last Light Curve* takes the last result of the Light Curves tab; *Period Analysis of Selected* in Field Variables opens a target's light curve. Data summary: N, time span T, nights, median spacing, frequency resolution 1/T, pseudo-Nyquist frequency.
  - **Periodograms**: Lomb–Scargle (astropy; shown as amplitude spectrum in the data unit; optional harmonics for non-sinusoidal shapes; false-alarm probability of the highest peak), **PDM** (phase dispersion minimisation; 1 − θ shown), **BLS** (box least squares, for eclipses; magnitudes handled). Period range and oversampling selectable; x axis as frequency or (log) period.
  - **Spectral window** overlaid on the highest peak, with the positions of its strongest side peaks (aliases, e.g. ±1 c/d for single-site data).
  - **Prewhitening**: *Add Highest Peak* adds the strongest peak of the residual periodogram and fits all frequencies together (non-linear least squares of frequencies, amplitudes and phases; each frequency with a chosen number of harmonics); frequencies can also be added by clicking into the periodogram (*Add Preview Frequency*) or by value (frequency or period), removed, or refitted with more harmonics. Table with frequency, period, amplitude and uncertainties (Montgomery & O'Donoghue 1999), S/N (amplitude / mean residual amplitude within ±1 c/d; S/N < 4 marked as not significant) and notes on harmonics (2f1), combinations (f1+f2), possible 1 c/d or 1/yr aliases and frequencies below 2/T (trends).
  - **Plots**: light curve with the model (per file in different colours), periodogram of the current residuals with the fitted frequencies, phase plot of the selected frequency with all other frequencies subtracted (or of a clicked preview frequency); zoom/pan toolbar.
  - **Save Results…**: frequency table (CSV with a header describing data and settings), figure (PNG), periodogram (CSV) and residuals (CSV), in `photometry_output/period_analysis/` by default.

### Tests
- 3 new tests (prewhitening of a simulated multi-periodic star incl. harmonic, S/N; harmonic/combination/alias identification and spectral window; PDM, BLS, reading/combining CSV files, saving): 34 tests pass. GUI test of the whole workflow.

---

## [4.20] — 2026-11-05 — several nights and pointings

Originals of changed files are in `backup_v4.19/`. Single-night results with all stars on all images are unchanged.

### Added / changed — field of view
- **Field-of-view check against every image** (new `field_variables.image_footprints` / `in_footprints`; WCS from the headers only, fast). Several nights usually have slightly (or very) different pointings; until now only the first image was used.
  - *Field Variables*: new column **On images** (e.g. `4/6`) = on how many of the checked images a target lies (at least *Edge margin* px from the border). Targets on none of them are deselected and skipped with a note. The VSX/GCVS query covers the area of **all** checked images (centre and radius from all footprints), and a variable is kept if it lies on at least one of them. Images without WCS are reported and ignored.
  - *Light Curves*: before measuring, the target, comparison and check stars are checked against all selected images. If the target is on none of them the run stops with a message; otherwise the console/status line says on how many images a star is outside.
  - Photometry: a target that is not on an image is skipped there without error messages (count reported per target); a target on no image returns "target outside the image in all … frames".
- **Comparison stars across nights**: the automatic choice (Field Variables) only uses stars that are on **every** image containing the target (if too few, on ≥ 80 % of them).
- **Relative flux with an incomplete ensemble**: if a comparison star cannot be measured on some images (e.g. outside the field on one night), the ensemble sum of those images is scaled up by the share of the missing stars in the complete ensemble (median over complete images). Before, the ratio jumped. Column `ens_complete` in the CSV marks such points. (Calibrated mode: each comparison star gives its own zero point, so a missing star only changes the average slightly.)
- **File names for several nights**: light-curve CSV/plot/AAVSO report (Light Curves) and the Field Variables output folder are named `first_to_last` date when the data span more than one night, e.g. `light_curve_R_Leo_2026-03-01_to_2026-04-15.csv`; the overview plot title gives the number of nights.

### Tests
- New test with shifted pointings (target off one image, a comparison star off two): 31 tests pass. GUI test with 6 images from 3 "nights" with different pointings.

---

## [4.19] — 2026-11-05 — Field Variables (light curves of many known variables)

Originals of changed files are in `backup_v4.18/`. Results of the existing tabs are unchanged.

### Added
- **New sidebar tab *Field Variables*** (new module `photometry/field_variables.py`): light curves for many known variables in the field in one run - meant for long-period variables observed over many nights, but works for any period.
  - **Targets from a list file** (*Load List…*): one star per line, `name RA Dec` with RA/Dec sexagesimal (`09:36:53.2 +44:04:00`, `09 36 53.2 +44 04 00`) or in degrees, separated by blanks, commas, semicolons or tabs; or a CSV with header (`name,ra,dec[,type,period,max,min]`). Lines with only a name are resolved via SIMBAD/VSX. Targets outside the image are listed but deselected.
  - **Targets from a catalogue** (*Query VSX (incl. GCVS)* / *Query GCVS*): all catalogued variables in the field of the first checked image (from its WCS), filtered by brightness at maximum, minimum amplitude, period range, type (e.g. `M SR L`) and distance from the image edge. VSX results are cached; old VSX cache files without type/range/period are refreshed automatically.
  - Targets can be added by hand (*Add…*), removed, (de)selected by clicking the *Use* column, and saved as CSV (*Save List…*, re-loadable). The same star from list and catalogue is only listed once.
  - **Comparison stars chosen automatically for every target** from the reference catalogue (Settings): isolated catalogue stars (no neighbour within 20″), not known variables, on the image, within 12′ of the target (widened if necessary) and close in brightness (±1.5 mag, widened stepwise), never brighter than 8 mag; one further star serves as check star. Modes: *Calibrated magnitudes*, *Relative flux* (same comparison stars, no calibration) or *Raw ADU*.
  - **One pass through the images for all targets** (`time_series.run_multi_target_photometry`): each FITS file is read once; per target the same algorithm as the Light Curves tab (identical results for a single target, tested).
  - **Output** in `photometry_output/field_variables/<date>_<filter>/`: one CSV and one plot per target, `summary.csv` (N, mean, standard deviation, 5–95 % range, median error, reduced χ², time span), and `overview.png` with one small panel per target. Optional AAVSO reports (calibrated mode, default off). The table in the tab shows the statistics; buttons open the overview, the plot of the selected star, the summary, the output folder, or send the selected star to the Light Curves tab for a detailed analysis with hand-picked comparison stars.
- **`.fts` files**: the File Manager accepts `*.fts` (as well as `.fits` / `.fit`, any capitalisation) in *Load Files* and *Load Directory*.

### Changed
- `run_time_series_photometry` is now a wrapper around `run_multi_target_photometry` (same signature and results).

### Fixed
- *Catalog Search Radius*: an error introduced in v4.18 made the radius function call itself (endless recursion) - affected the colour transformation, the pipeline catalogue query and AAVSO reference-star fetching whenever the radius was needed.

### Tests
- 4 new tests in `tests/test_calibra.py` (list formats, simulated GCVS answer and filters, VSX cache refresh, multi-target light curves on synthetic frames incl. automatic comparison stars and equality with the single-target function): 30 tests pass.
- The online queries (VSX, GCVS, SIMBAD) could not be run in the development environment (no internet access there); they were tested with simulated catalogue answers.

---

## [4.18] — 2026-11-04 — user-friendliness

Originals of changed files are in `backup_v4.17/`. Photometric results are unchanged. Open ideas are collected in `TODO.md`.

### Changed — one place for each setting
- **Extinction**: the Light Curves tab no longer has its own *Extinction (k)* field (default 0.15); it uses *k_V* / *k_B* from *Settings → Astro & Catalog* according to the filter. (In the differential light curve the extinction term cancels between target and comparison stars, so results do not change.)
- **Observing site**: latitude/longitude moved to *Settings → Operator & Session* (same values as before). They are only used if a FITS header has no `SITELAT`/`SITELONG`.
- **Detection FWHM**: Data Mining and *Settings → Detection* now share one value.

### Changed — readability and layout
- Numbers in entry fields are rounded for display (magnitudes / colours 3 decimals, transformation coefficients 4, zero points 3) after loading a session or setup, importing AAVSO stars, exporting from the FITS Viewer and loading coefficients. Before: e.g. `10.256653035127442`.
- **Light Curves tab**: a fixed action bar at the top (*Generate Light Curve*, *Open Selected FITS in Viewer*, *Save Setup…*, *Open Setup…*, *Open Output Folder*, progress bar, status line) that stays visible while scrolling. The *Coefficients, AAVSO code & chart* section can be collapsed (state is remembered).

### Added
- **Named setups** (Light Curves): *Save Setup…* / *Open Setup…* store/restore target, comparison and check stars, mode, coefficients, chart ID and the AAVSO sequence for that target in `setups/<target>.json`. Site and layout settings are not part of a setup.
- **Dated output files**: light-curve CSV, plot and AAVSO report carry the UT date of the first observation, e.g. `light_curve_AE_UMa_relflux_2026-04-13.csv`, `aavso_AE_UMa_2026-04-13.txt`. Different nights no longer overwrite each other; re-running the same night and mode replaces its own files. *Open AAVSO Report* opens the report of the last run (or the newest one for the target).
- **Open Output Folder** buttons (Light Curves, Data Mining).
- **Data Mining → Light Curves with comparison stars**: *Send to Light Curves (+ comp stars)* (button and right-click menu) also fills 4 comparison stars and 1 check star automatically, chosen from the Data Mining statistics: constant (not suspect/noisy/known variable, RMS ≤ 1.5× own photometric error), measured in ≥ 95 % of the frames, within 1 mag of the target (widened if needed) and close on the sky. The mode is set to *Relative flux* (no catalogue magnitudes needed); for calibrated magnitudes fetch catalogue values and switch mode.

### Fixed
- Data-Mining stars sent to Light Curves were named `DM_DM_0034` (prefix added twice).

### Documentation / credits
- *About Calibra* page: current version and date (was "4.1, 2026-05-22"), updated description of all functions, list of recent changes, credits.
- *User Help*, README and manual: "Developed by Stephan Pomp, Google DeepMind Antigravity and Claude (Anthropic)"; quick tips updated (aperture 1.5–2× FWHM, location of the region filter, sky noise factor, setups).

### Tested
- GUI: Data Mining on a 20-frame synthetic sequence → *Send to Light Curves (+ comp stars)* → 4 comps + 1 check chosen → light curve generated (variable 19 % scatter, check star 1.1 %); setup saved and re-opened (54 values; site not included); dated output files; rounding; collapsible section; fixed action bar (screenshot).
- Regression: 26 automated tests, File Manager / header editor / invalid numbers / calibrated light curve unchanged (10.765).

---

## [4.17] — 2026-11-04 — speed (Light Curves and Data Mining)

Originals of changed files are in `backup_v4.16/`. Results are unchanged within measurement noise (see "Tested").

### Why it was slow
- `photutils.aperture_photometry()` converts the **whole** image to float on every call when it gets integer FITS data: ~116 ms per call on a 6248×4176 frame. Light Curves and Data Mining called it once **per star**.
- The 2-D Gaussian centroid (`centroid_2dg`) costs ~34 ms per star.
- Data Mining computed the global sky scatter on all 26 Mpixel of every frame (~3 s).
- Re-running Data Mining (other thresholds etc.) re-read and re-measured every frame.

### Changed
- New `photometry/fastphot.py`:
  - `as_float_image()` — convert each frame once;
  - `windowed_centroid()` / `track_centroid()` — coarse peak search within the tracking radius + iterative Gaussian-windowed centroid (same or better accuracy than `centroid_2dg`, ~0.3 ms instead of ~30 ms; same "fall back to the predicted position if the star moved too far" protection);
  - `fast_sky_stats()` — numpy implementation of the sigma-clipped annulus statistics, identical results to astropy (tested), ~2× faster; now used by `noise.sky_stats` everywhere.
- **Light Curves** (`time_series.py`): frame converted once; fast centroid with window width from the frame's FWHM.
- **Data Mining** (`data_mining.py`): frame converted once; all stars measured in one vectorised aperture call; fast centroid; global sky scatter from a 1/16 pixel subsample.
- **Data Mining measurement cache**: per-frame measurements stored in `photometry_output/dm_cache/` (compressed `.npz`, ~100 kB per frame for 2000 stars) and the reference-frame detection as JSON. The cache key contains the file (path, size, modification time), the star list and all measurement settings, so changed files or settings are always re-measured. New GUI options: *Reuse stored measurements* (default on) and *Clear Measurement Cache*.

### Measured on the AE UMa V stack (in the test environment; absolute times on a laptop may differ, the ratios should be similar)
| | v4.16 | v4.17 |
|---|---|---|
| Light curve, first run | 0.62 s/frame | 0.33 s/frame |
| Data Mining measurement | 69 ms per star and frame | 0.7 ms per star and frame |
| Data Mining, 1871 stars | ~130 s/frame | 1.3 s/frame |
| Data Mining re-run (cached) | same as first run | 0.006 s/frame; detection 0 s instead of 4.6 s |

Example: 50 frames × 300 stars — about 17 min → about 1 min; a re-analysis with other thresholds → a few seconds.

### Tested
- Light curve AE UMa: 11.3281 (v4.16) vs 11.3280 (v4.17), error unchanged.
- Data Mining, 150 stars on the real frame: median difference 0.0003 mag (MAD 0.0005; max 0.022 mag for one star).
- Synthetic sequence with up to 3.5 px drift and a fixed WCS: 100 % of measurements obtained, constant stars 0.010 mag rms, variable recovered (correlation 0.998) in Data Mining and Light Curves.
- Centroid accuracy vs `centroid_2dg` on simulated stars (FWHM 2.5–5 px, 2·10³–2·10⁵ ADU): equal or better.
- 26 automated tests (4 new: sky statistics = astropy, centroid accuracy, far-jump rejection, cache round trip); GUI Data Mining run twice (second run from cache, identical results); Light Curves regression.

### Not done (by choice)
- Parallel processing and the analysis-pipeline speed-ups (shift analysis, detection) — not needed for a few dozen frames on a laptop.

---

## [4.16] — 2026-11-04 — science revision (code review step 3 + remaining items)

**These changes alter numerical results.** Originals of all changed files are in `backup_v4.15/`. Measured effects refer to `AE_UMa_{V,B}mag_April13_stacked.fits` (V: 49-frame average, B: 8-frame average) with your session settings and the cached ATLAS refcat2 catalogue.

### Changed — noise model (review S1)
- New `photometry/noise.py`, used by the pipeline (`aperture_phot.py`), Light Curves (`time_series.measure_star`), Data Mining and the viewer:
  `var = F_net/g_eff + k² · A · σ_sky² · (1 + A/n_sky)`.
  - Poisson term from the **net** flux (was the raw aperture sum incl. sky → sky counted twice).
  - **Effective gain** `g_eff = gain × N` for averaged stacks (`STACKCNT`/`NCOMBINE`/`NSTACK`), new setting *Stacked images* = average / sum / none (default average). Data Mining previously assumed gain = 1.
  - `n_sky` = annulus pixels after sigma clipping (was all pixels).
  - New setting *Sky noise factor* k (default 1) for correlated noise in registered stacks; each pipeline run prints a **blank-sky noise check** (1500 empty apertures). AE UMa: k = 1.40 (V), 1.33 (B).
- Effect: median magnitude errors 4–5× smaller in V (e.g. 0.087 → 0.019 mag at m_inst −9…−8), 2× smaller in B. A Monte-Carlo test (`tests/`) confirms quoted errors = actual scatter (±15 %).
- Light-curve cache renamed to `time_series_cache_v2.json` and keyed by the noise settings (old measurements are re-measured).

### Changed — zero-point calibration (review M3, H6, saturation)
- ZP uncertainty = error of the median, 1.2533 σ/√N (was the star-to-star scatter σ, also added to every star's error; 0 for a single star → now 0.05). AE UMa V: 24.034 ± 0.0013 (scatter 0.037, 1199 stars); v4.15: 24.036 ± 0.020.
- Calibration stars additionally require raw peak ≤ *Max Peak for Calib Stars* (new setting, default 30 000 ADU; non-linearity in stacks), no neighbour within the annulus (*Exclude blends*, default on) and a complete aperture.
- More stars pass the SNR ≥ 10 cut now that errors are realistic (V: 1286 instead of 789 matches).
- Colour dependence of the ZP is fitted and reported (pipeline magnitudes stay **untransformed**, now stated in the report). AE UMa: dZP/d(B−V) = −0.003 ± 0.003 (V), +0.103 ± 0.009 (B).
- *Catalog Search Radius* = 0 → automatic (image half-diagonal + 1′; AE UMa 87.7′); a note is logged when the radius covers only part of the image. Used consistently by pipeline, shift analysis, colour fit, differential photometry and AAVSO sequence query.
- Results CSV: new columns `peak_raw`, `saturated`, `blended`, `n_sky`, `bkg_median`, `bkg_std`.
- Stars whose aperture is not fully on the image are skipped (flag *edge*) instead of getting negative fluxes (review M1); same in Light Curves and the viewer.
- New setting *Detection FWHM* (was fixed 3.5 px); log note when the measured FWHM differs by > 30 %.

### Changed — colour transformation (review S2)
- Star selection: SNR ≥ *Min SNR (colour fit)* (default 50) in both filters, peak ≤ peak limit, not saturated/blended/variable, optional faintest catalogue V.
- Weighted least squares (CSV errors + 0.02 mag catalogue error), iterative robust 3σ clipping (MAD scale; was one 2σ pass with std). Tbv from fitting (b−v) vs (B−V) and inverting.
- Coefficient errors, χ²_ν, consistency check `Tbv = 1/(1 − Tb_bv + Tv_bv)`, V-term residuals vs magnitude in the report; JSON stores errors, consistency and selection.
- AE UMa field: **Tbv 0.970 ± 0.011, Tb_bv 0.014 ± 0.008, Tv_bv 0.033 ± 0.007, consistent** (implied Tbv 0.981). v4.15: 0.921 / 0.149 / 0.015 (inconsistent: implied Tbv 1.15).

### Changed — differential photometry (review S3, S4)
- Default **ensemble** of all suitable comparison stars (SNR ≥ 30, peak ≤ limit, no blend, not variable, −0.2 ≤ B−V ≤ 1.6), 3σ-clipped mean zero points with errors of the mean. *Automatic single star* now picks the star closest to the field's median brightness/colour (was: the brightest). Name/coordinate mode unchanged.
- **B = V + (B−V)**: B, V, B−V are now consistent (max |B−V − B_V| 0.208 → 0.0001 mag).
- New columns `B_err`, `V_err`, `B_V_err` (instrumental, zero-point and coefficient errors × colour difference to the comparison stars) and `is_comp`.
- Optional **check star** (name or RA Dec) reported against the catalogue.
- AE UMa field: ΔV (calibrated − ATLAS) mean +0.083 → **+0.011** mag (426 comparison stars); check star 000-BJS-557: dV = +0.015, d(B−V) = +0.027.

### Changed — Data Mining
- Per-star photometric error (noise model) → CSV columns *Phot. Error (median)* and *RMS / Phot. Error* (≈ 1 for constant stars); a Suspect must also have RMS ≥ 2× its own error.
- Excess RMS computed in quadrature, √(RMS² − σ_exp²) (was linear difference).
- *Fast Test Mode* off by default and clearly labelled.
- Settings read on the GUI thread; status updates via the GUI thread.
- Fixed: crash in the scatter plot when only a few stars were available (scalar noise-floor function).

### Changed — FITS Viewer
- Magnitudes use the pipeline's photutils aperture photometry (`exact`, clipped annulus); no silent fallback to the Gaussian-fit magnitude; the clicked star's calibrated magnitude now uses the aperture magnitude (shown alongside "PSF M") so it is consistent with the ensemble ZP (review M11).
- VSX search radius = largest centre-to-corner distance (was capped at 45′; review M7).
- Image held as float32 (half the memory); click fit uses `TRFLSQFitter` when available.

### Changed — OSC split (review M9)
- All old WCS scale/distortion keywords (CD, PC, CDELT, CROTA, SIP) replaced by the rescaled solution; SIP dropped with a note.
- Float input stays float32; averaged G channel written as float32; integer R/B as uint16 (rounded).
- Warning if the chosen Bayer pattern differs from header `BAYERPAT`.

### Added
- `tests/test_calibra.py`: 22 automated tests (filters, noise model incl. Monte-Carlo error check, B−V consistency, robust fit, image calibration, Data-Mining alignment). Run with `python -m pytest tests -q`.

### Not changed (deliberately)
- Instrumental magnitudes are not normalised per second (would change the meaning of all stored default zero points).
- No AAVSO "fainter-than" reports; flexible-aperture factor default stays 2.0 (1.5 × FWHM is usually better for faint stars — try it).
- The default *Min SNR for Calib* stays 10; with realistic errors you may raise it (e.g. 20–30) to reduce the ZP scatter.

### Tested
- Pipeline v4.15 vs v4.16 on both AE UMa stacks; colour fit and differential photometry on the new CSVs (numbers above).
- Synthetic: Monte-Carlo error check, light-curve modes (unchanged results for single frames), OSC split with CD+SIP header (positions agree to < 0.01″), Data Mining (variable found, constant stars RMS/error ≈ 1.0).
- Headless GUI: pipeline run with automatic catalogue radius, Data Mining, FITS Viewer, Light Curves (calibrated/relative), File Manager, header editor, invalid-number handling.

---

## [4.15] — 2026-11-04 — bug-fix release (code review, steps 1 + 2)

Based on `Calibra_code_review_2026-11-04.md`. No change to the photometric methods; the science changes of review step 3 (error model, colour fit, multi-star differential ZP) are **not** included yet. All originals of changed files are in `backup_v4.14/`.

### Fixed — wrong inputs reaching the science code
- **One B/V filter classification** (`photometry/filters.py`, `classify_filter()`), used for File Manager counts, flat selection (GUI + pipeline), catalogue band in `match_and_calibrate` (new `band=` argument), default-ZP routing after a pipeline run, Light Curves file selection, the FITS Viewer's catalogue magnitudes and AAVSO labels. Previously five different rules: e.g. `Bessell V` / `Baader V` were calibrated against **B** magnitudes and got the B flat, and with FILTER=`B` the B zero point could be written into `default_zp_v`. Files that are neither B nor V are now reported instead of silently treated as V.
- **FITS Viewer close** (`fits_viewer.py` `on_close`) no longer exports automatically. Before, every close overwrote the Light Curves target name (→ `Target`), comparison-star names (→ `Uncataloged…`), magnitudes/B-V (→ NaN, offline or without catalogue match), replaced AAVSO sequence magnitudes by catalogue values, and replaced the fixed apertures by one star's flexible aperture — all saved to the session. Now: you are asked only if the selection changed; apertures only via *Export Aps*.
- **Export to Light Curves** (`gui.py` `update_light_curve_stars`): a star already configured within 3″ keeps its name, magnitude, B-V and source label; placeholder names never overwrite real ones (new uncatalogued stars get a `Jhhmmss±ddmmss` name); NaN is never written; the target keeps its name if it is the same star.
- **File Manager ticks** are kept across table refreshes (load, Refresh Headers, header save, after calibration/plate solve/OSC split) and saved in the session (`unchecked_file_paths`). Before, every refresh re-ticked all files.
- **"Remove Checked/Selected" → "Remove Highlighted"**: removes the highlighted rows; removing all checked files needs a confirmation. Before, with everything ticked by default, highlighting one row and pressing Remove emptied the list.
- **Bias/flat calibration** (`image_calibration.py`):
  - output named `cal_<input file>` (was: from the `FILENAME` header, which stacks inherit from a sub-frame → calibrated stack not found, GUI silently kept the uncalibrated file; stacks could overwrite each other);
  - saved as 32-bit float without clipping/truncation (was: clipped at 0 and cast to uint16); `BZERO/BSCALE/BLANK` removed, `CALSRC` added;
  - new option *"Master flats are already bias/dark-subtracted"* (default ON = previous behaviour); untick to subtract the bias from raw master flats; normalised flats are detected;
  - size mismatch → frame skipped with a clear message (was: warning, then crash); flat pixels < 5 % of the median are left uncorrected (was: divided by 1e-6);
  - success = file written in this run (was: any existing `cal_*` file, also from older runs); calibration files are closed; Tk state is read before the worker thread starts.
- **Data Mining frame alignment** (`data_mining.py` `calibrate_and_analyze`): magnitudes are stored per frame index (NaN-filled). Before, a star missing in one frame (exception part-way through a frame) shifted all its later values to the wrong frames or raised `IndexError`.

### Fixed — robustness and visibility of errors
- Error dialogs of the Calibration, Plate-Solve, OSC-Split, Data-Mining and ensemble-resolution threads raised `NameError` (`lambda: … {e}` after `e` was deleted) — the dialogs now appear.
- Invalid or empty numeric fields: `read_tk_var()` parses the raw text and reports *which* setting is wrong (was: uncaught `TclError`, console traceback only; `IntVar` silently turned `3.7` into `3`). A global Tk error handler shows any remaining callback error in a dialog.
- Analysis pipeline: errors in the worker thread are shown in a dialog; GUI widgets are updated on the GUI thread; the session log file is closed after each file.
- `calibra.py`: one corrupt/failing file no longer aborts the whole batch (per-file try/except with traceback and summary).
- Console output (`StdoutRedirector`) from worker threads is queued and inserted by the GUI thread (tkinter is not thread-safe).
- **FITS header editor**: `<file>.bak` backup before the first save; unchanged cards keep their original value and type (`'007'` stayed `'007'`, not `7`); edited string cards stay strings; `nan`/`inf` saved as text (previously the whole save failed); COMMENT/HISTORY cards are written (previously silently dropped). Data, `BITPIX` and `BZERO` are untouched (verified).
- **Plate solving** (`plate_solve.py`): solution file is `<name>.wcs` for any extension (`.FIT`/`.fts` previously pointed to the FITS file itself → false "solved"); stale `.wcs` removed before solving; result must contain a celestial WCS; 10-minute timeout; clear message if ASTAP is not found; data read without memory mapping before overwriting the file.
- **RA/Dec region filter** (`star_detection.py`) works (bounds were passed to `SkyCoord` without units → always fell back to *all*); sexagesimal input and regions crossing RA 0/360° supported.
- **Session**: loading a session twice no longer duplicates files.
- **Two `on_reset_ensemble` functions** merged; both buttons now also clear the check-star selection and leave the target unchanged.
- Data Mining → *Open in FITS Viewer* passes the configured catalogue, filter keywords and matching default ZP.

### Fixed — small / low severity
- `calibration.py`: ATLAS stars lacking g **or** r get no B/V (was V := g and B := r); VSX cache-coverage and radius tests include cos δ (RA distances were overestimated by 1/cos δ — at Dec 44° known variables near the edge of the cached circle were missed: on the AE UMa V stack 108 instead of 64 VSX stars are now excluded from the zero-point set; ZP 24.037 → 24.036); duplicated table separator in the calibration report removed; online catalogues no longer labelled "Local File".
- `shift_analysis.py`: uses the configured catalogue search radius (was fixed 15′ → separate catalogue query).
- `psf_fitting.py`: FWHM = mean of |σx|,|σy| (was |mean|, wrong for opposite signs); divergence check on both axes.
- `color_calibration.py`: invalid escape sequences (`\c` SyntaxWarnings) fixed.
- `FITSInspector.py` marked as unused/deprecated (it is not called anywhere).
- Version 4.15 (window title, AAVSO `#SOFTWARE`).

### Tested
- Pipeline on `AE_UMa_Vmag_April13_stacked.fits` (offline catalogue cache), v4.14 vs v4.15: ZP 24.037 ± 0.021 vs 24.036 ± 0.020; difference fully explained by the VSX cos δ fix; shift analysis runs with the configured radius.
- `classify_filter` unit cases (Vmag, Bmag, B, V, Bessell V, Baader V, SBIG V, Johnson-B, JohnsonV, Vmag_2, L, R, Master_L, …).
- Headless GUI: filter counts (`4 files (2× V, 1× B)` with Vmag/Bmag/Bessell V/L), ticks survive Refresh, Remove Highlighted, header-editor round trip (backup, `'007'` kept, `nan` as text, HISTORY added, uint16/BZERO intact), invalid number → clear dialog, FITS Viewer on the real V stack: plain close changes nothing; explicit export (offline placeholders) keeps AE UMa, 000-BJS-55x names and magnitudes, no NaN.
- `calibrate_image` synthetic test (float32 output, flat field correct, size mismatch → not calibrated); Data Mining with a partial frame (no crash, injected variable found); light-curve modes regression (raw/relative) unchanged.

---

## [4.14] — 2026-11-04

### Added — Light Curve Modes (Light Curves tab)
- **New "Light Curve Mode" box** at the top of the Light Curves tab with three modes:
  - **Calibrated magnitudes** (default; identical to v4.13 behaviour): ensemble differential photometry using catalogue magnitudes, B-V, colour term and extinction. Writes the AAVSO report.
  - **Relative flux**: reference stars are used **only through their measured ADU**; catalogue magnitudes, B-V, colour term and extinction are ignored.
    `rel_flux = F_target / Σ F_ref`, with `σ_R = sqrt((σ_Ft/S)² + (F_t·σ_S/S²)²)` and `σ_S = sqrt(Σ σ_Fi²)`.
    Also gives the differential instrumental magnitude `Δm = -2.5 log10(rel_flux)`, the per-frame ensemble flux, a transparency-corrected target flux in ADU (`flux_corr_adu = rel_flux · ⟨S⟩`), the check star relative to the ensemble, and each reference star relative to the sum of the *other* references (stability diagnostic, needs ≥2 refs).
  - **Raw ADU**: background-subtracted net flux of the target in ADU with CCD-equation uncertainties. **No reference or check stars required.** Reference/check stars, if defined, are measured for display and statistics only and are *not* used to correct the target.
- **Automatic Raw ADU**: checkbox *"Automatically use Raw ADU when no reference stars are selected"* (default on). When *Generate Light Curve* is pressed with no reference star ticked *Use*, the mode switches to Raw ADU and an information box explains this. With the box unticked the previous error message is shown.
- **ADU/s option**: checkbox *"Raw ADU: normalise to ADU/s (divide by EXPTIME)"* for sequences with varying exposure times.
- **Flux vs. magnitude plotting** for the uncalibrated modes: checkbox *"Raw/Relative: plot flux"*. Flux plots are not inverted; magnitude plots are. Axis labels adapt automatically. The *Customize Plot* window inherits the choice (default Y label and Y inversion set by mode).
- **Mode-aware results table**: columns are relabelled `Flux [ADU]` / `Flux [ADU/s]` / `Rel. Flux` and `Check Flux` in the uncalibrated modes.
- **Mode-aware Analysis Summary**: mean, standard deviation, relative scatter (%), peak-to-peak and median formal error of the target; scatter of check and reference stars in % and approximate mag.
- **Separate output files** for the uncalibrated modes: `light_curve_<target>_rawADU.csv`, `plot_<target>_rawADU.png`, `light_curve_<target>_relflux.csv`, `plot_<target>_relflux.png`. Calibrated outputs keep their old names, so earlier results are never overwritten by uncalibrated runs. No AAVSO report is written in the uncalibrated modes.
- New CSV columns in uncalibrated modes: `lc_mode`, `flux_unit`, `flux`, `flux_err`, `target_adu`, `target_adu_err`, `exptime`, `check_flux`, `check_flux_err`, `ens_flux`, `ens_flux_err`, `ref_fluxes`, and (relative mode) `flux_corr_adu`, `flux_corr_adu_err`.
- "What does this do?" help for the Light Curves tab describes the three modes.
- New session settings (saved in `calibra_session.json`): `ts_lc_mode`, `ts_auto_raw`, `ts_flux_per_second`, `ts_plot_flux`. Older session files load fine (defaults apply).

### Changed
- `photometry/time_series.py` → `run_time_series_photometry()` has two new keyword arguments, `lc_mode='calibrated' | 'relative' | 'raw'` and `flux_per_second=False`. Default behaviour is unchanged.
- Outlier flagging (rolling median + MAD) works in flux space for the uncalibrated modes (minimum jump 5 % of the median flux ≈ 0.05 mag). Frames with non-finite values (e.g. a negative net flux in Raw ADU mode) are kept and flagged `SUSPECT` instead of being dropped.
- `plot_light_curve()` handles all modes (`plot_options['y_quantity'] = 'flux' | 'mag'`), uses an object-oriented matplotlib `Figure` when saving to file (safe to call from the worker thread), and only skips the legend when there is nothing to show.
- CSV writer uses the union of all result keys (robust against rows with differing columns).
- Version bumped to 4.14 (window title and AAVSO `#SOFTWARE` header).

### Fixed
- **Light-curve plot file was never written**: the "Saved Plot" path shown in the summary (`plot_<target>.png`) was not actually created; instead every preview redraw wrote a stray file named `.png` into the working directory. The plot is now saved to the stated path and the stray file is no longer produced (an old `.png` in the Calibra folder can be deleted).
- **HJD silently became JD on cached re-runs**: when photometry was taken from the session cache for FITS files without `SITELAT`/`SITELONG`/`SITEELEV`, `get_hjd()` received `None` for the site coordinates, failed, and the time fell back to mid-exposure JD without the heliocentric correction (up to ±~8 min). Missing keywords now correctly fall back to the site latitude/longitude from the GUI. AAVSO reports produced from such cached re-runs before v4.14 may be worth regenerating.

### Tested
- Synthetic 30-frame sequence (sinusoidal variable + cloud-like transparency changes, 2 refs + check): Raw ADU recovers the injected flux, Relative flux recovers the injected ratio and removes the transparency variations; scatter matches the propagated errors (≈2.0 % vs. 2.15 %). Calibrated mode unchanged.
- GUI driven end-to-end (all modes, auto-switch, table, summary, plot toggle, Customize Plot, file outputs).
- Real data `AE_UMa_Vmag_April13_stacked.fits` with the session's ensemble: Calibrated V = 11.328 ± 0.007; Relative flux + ensemble catalogue magnitudes = 11.333 ± 0.003 (agree within the zero-point scatter of 0.006); Raw ADU = 121 948 ± 315 ADU.
