# Calibra v4.14 — Code Review (2026-11-04)

> **Status (updated 2026-11-04):** all findings in this review have been addressed.
> Steps 1 + 2 → **v4.15**, step 3 and the remaining medium items → **v4.16** (see `CHANGELOG.md` for what was changed and the measured effect on the AE UMa data).
> Deliberately *not* changed: per-second instrumental magnitudes, AAVSO "fainter-than" reports, the flexible-aperture default factor (see CHANGELOG 4.16 "Not changed").
> Not reproduced/fixed: the `get_cached_catalog` "outdated cache" behaviour is intentional (older cache files lack columns and are refreshed automatically when online).


Scope: the whole code base except the Light-Curve code already reworked in v4.14. Method: the code was read in full. Suspected bugs were then reproduced either on your AE UMa stacks (`AE_UMa_{V,B}mag_April13_stacked.fits`, using the cached ATLAS/Gaia/VSX catalogues) or on small synthetic tests. **No code was changed in this review.**

Status: **Confirmed** = reproduced or checked directly in the code. **Suspected** = plausible, not proven.

The pipeline reproduces your logged V zero point exactly: 24.037 ± 0.020 from 823 stars. WCS pixel conventions (0/1-based), catalogue matching (no duplicate matches), the 1.0857/SNR conversion, the AAVSO sign conventions of the transformation equations, Gaia→V (Riello+2021), and the OSC superpixel CRPIX/CDELT halving all check out.

---

## 1. Priority overview

| # | Severity | Area | Problem (short) |
|---|---|---|---|
| S1 | Critical | Error model | Photometric errors overestimated: sky counted twice, stack count ignored (≈3× for faint stars) |
| S2 | Critical | Colour transformation | Coefficients biased by faint stars; Tb_bv ≈ 0.15 is largely an artefact |
| S3 | Critical | Differential photometry | One "brightest" comparison star sets the ZP → +0.07 mag offset in V; no errors output |
| S4 | High | Differential photometry | Reported `B_V` ≠ `B_mag − V_mag` (up to ±0.2 mag) |
| S5 | Critical | FITS Viewer | Closing the viewer silently overwrites target name, comp mags (→ NaN) and apertures |
| S6 | High | Filter handling | Five different B/V tests; "Bessell V" is treated as B; B ZP can be stored as V default |
| H1 | High | File Manager | Every refresh re-ticks all files; "Remove" deletes all checked files |
| H2 | High | GUI errors | Error pop-ups in 4 worker threads crash with NameError → failures stay silent |
| H3 | High | Settings | Invalid/empty numeric fields fail silently (TclError not caught) |
| H4 | High | Image calibration | Output named from FITS `FILENAME` → stacks silently stay *uncalibrated*; negative values clipped/uint16 |
| H5 | High | Data mining | One failed star in a frame shifts that star's later points to the wrong frames |
| M1–M12 | Medium | various | see §4 |

---

## 2. Science-critical findings

### S1 — Error model overestimates the photometric errors (`aperture_phot.py:64`; same for `time_series.py` `measure_star`) — Confirmed
```python
variance = max(raw_flux,0)/gain + A*bkg_std**2 + A**2*bkg_std**2/n_ann
```
- `raw_flux` is the aperture **sum including sky and pedestal**. The sky is then counted a second time through `A·σ_bkg²`. The Poisson term should use `net_flux`.
- Your stacks are **averages of N subframes** (`STACKCNT` = 49 in V, 8 in B). The effective gain is therefore ≈ N × 1.27 e⁻/ADU, not 1.27.
- **Evidence (V stack):** 3000 apertures on blank sky have a measured scatter of 64 ADU; the code predicts 186 ADU. For stars at instrumental mag −9…−8 the code quotes 0.077 mag, but the actual scatter around the ZP is 0.028 mag (catalogue errors included).
- **Consequences:**
  - The "SNR ≥ 3" cut is really about SNR ≈ 9.
  - The detection limit is quoted too bright.
  - Errors in AAVSO reports are inflated.
- **Fix:**
  ```python
  g_eff = gain * header.get('STACKCNT', 1)          # for averaged stacks
  var   = max(net_flux,0)/g_eff + A*bkg_std**2*(1 + A/n_sky_used)
  ```
  `n_sky_used` = annulus pixels *after* sigma clipping.
- **Optional:** blank-sky scatter is about 1.5× √A·σ_pix, because registration and interpolation correlate neighbouring pixels. An empirical correction factor measured from blank-sky apertures (an option in Settings) would give realistic errors.

### S2 — Colour coefficients are biased by faint stars (`color_calibration.py:9-39, 130-137`) — Confirmed
The fit is unweighted least squares over **all** matched stars. It has no SNR, magnitude or saturation cut, uses a single 2σ clip based on the non-robust std, and in the Tbv fit the noisy instrumental colour sits on the x-axis. The three coefficients are not independent: Tbv = 1/(1 − (Tb_bv − Tv_bv)). Re-fitting your own data:

| Stars used | N | Tbv | Tb_bv | Tv_bv | Tbv implied by Tb, Tv |
|---|---|---|---|---|---|
| V < 11 | 36 | 0.989 ± 0.022 | −0.038 ± 0.026 | −0.052 ± 0.035 | 1.014 |
| V < 12 | 86 | 0.966 ± 0.010 | −0.007 ± 0.011 | −0.013 ± 0.015 | 1.007 |
| V < 13 | 149 | 0.967 ± 0.008 | 0.013 ± 0.008 | 0.040 ± 0.008 | 0.974 |
| all (current) | 939 | 0.926 ± 0.013 | **0.148 ± 0.011** | 0.014 ± 0.004 | **1.154** (inconsistent) |

**Interpretation:**
- For bright stars the coefficients are mutually consistent and your B and V filters look close to standard.
- The large Tb_bv comes from the faint end. That could be noise in the fit, blends, sky-error bias, or colour-dependent catalogue errors for faint stars. The real B colour term should be settled with a cut on bright, unsaturated stars.
- (A separate check of the B zero point against colour over all stars shows a slope of ~0.16. It is driven by the same faint stars.)

**Fix:**
- Select stars with SNR ≥ 50 (or V ≲ 12–13), a peak below ~30–50k ADU, and no neighbour within the annulus.
- Use a weighted fit or ODR (orthogonal distance regression), with iterative 3σ clipping using `mad_std`.
- Fit (b−v) against (B−V) and invert, as the AAVSO Transform Generator does.
- Report the coefficient errors (`linregress().stderr`) and the consistency check above.
- Plot residuals against magnitude.

### S3 — Differential photometry uses a single comparison star with no errors (`diff_photometry.py:193-236`) — Confirmed
- **Single brightest star:** auto mode picks the **brightest** unsaturated star with 0.4 ≤ B−V ≤ 0.8 as the only zero-point star. That is the star most at risk of non-linearity.
- **Evidence:** against ATLAS stars with V < 12 (N = 91), the calibrated V has a median offset of **+0.069 mag** with a MAD scatter of 0.019 mag. This matches the "ΔV μ = +0.082" in your report. B shows no offset (−0.003).
- **No uncertainties at all:** none are propagated (instrumental, comparison-star catalogue, or coefficient errors), and there is no check star.
- **Fix:**
  - Take the ZP as a sigma-clipped weighted mean over all non-variable comparison stars (as in the Light Curves tab).
  - Propagate σ² = σ_t² + σ_ZP² + (Δ(B−V)·σ_T)².
  - Add a check star.
  - When one comparison star *is* used, choose it close in magnitude and colour to the target.

### S4 — `B_V` column is not `B − V` (`diff_photometry.py:35-43, 274-276`) — Confirmed
`BV_t` is computed with Tbv, while B_t and V_t come from Tb_bv and Tv_bv. Because the three coefficients are inconsistent (S2), `B_mag − V_mag − B_V` ranges from −0.16 to +0.20 mag in `differential_photometry_results.csv`. **Fix:** compute V_t and BV_t, then set B_t = V_t + BV_t. Or use only Tb_bv and Tv_bv: Δ(B−V) = Δ(b−v)/(1 − Tb_bv + Tv_bv).

### H6 (relates to S2) — B `mag_calibrated` in the pipeline has no colour term (`calibration.py:630`)
For a star bluer than the comparison stars, such as AE UMa (B−V ≈ 0.25 vs a median of ~0.8), this matters at the level of the true Tb_bv × ΔBV. Either label the column "untransformed" or fit ZP + c·(B−V).

### Suspected: non-linearity well below the saturation limit
- In V, the ZP residual rises with peak value: +0.017 at 10–20k ADU, +0.040 at 30–40k, +0.08 at 40–50k.
- In B, stars with peaks of 30–62k sit about 0.05 mag off.
- In a registered stack, peaks that were saturated in individual subframes get smoothed. The 63 000 ADU limit is therefore too high for stacks; ~25–30k is safer for calibration stars.
- The `saturated` flag is also not written to the CSV.
- **Worth a dedicated test:** measure the same star in short and long exposures.

---

## 3. GUI / workflow bugs (wrong inputs reach the science code)

### S5 — Closing the FITS Viewer overwrites your Light-Curve setup (`fits_viewer.py:617-626` → `gui.py:884-964`) — Confirmed
`on_close()` always calls export, wrapped in a bare `except: pass`. In a headless test:
- Target name became `'Target'`.
- Comparison stars without a catalogue match got placeholder names and **mag/B-V = NaN** (NaN passes the `is not None` checks).
- AAVSO sequence magnitudes were replaced by ATLAS values.
- With flexible aperture on, one star's FWHM-based aperture (7.4/9.4/14.4) replaced your fixed 5/7/13.

All of this is then saved to the session. **Fix:** never export on close (only via the explicit buttons). Never overwrite with placeholders or NaN.

### S6 — Inconsistent B/V filter classification — Confirmed
Five different rules are in use:
- `'B' in FILTER` (`calibration.py:630`, `calibra.py:90`, viewer)
- `b_key in filt`
- `b_key in filt or filt == 'B'`
- a two-way substring test with V first (File Manager)
- `'V' in FILTER` (viewer labels)

**Consequences:**
- "Bessell V", "Baader V" or "SBIG V" are calibrated against B magnitudes and get the B flat.
- With FILTER=`B` and keyword `Bmag`, the File Manager counts the file as B, but the Analysis tab routes it to V and writes the B zero point into `default_zp_v`.

**Fix:** one `classify_filter(filt, v_key, b_key) -> 'V' | 'B' | None` (exact or mapping match), used everywhere.

### H1 — File Manager re-checks all files (`gui.py:1805`) — Confirmed
Every table rebuild (Load, Remove, Refresh, header save, after calibration, plate solve or OSC split) sets all rows to `[X]`. Frames you deliberately excluded return to pipeline, light-curve and data-mining runs. "Remove Checked/Selected" removes **all checked** files first, so highlighting one row and pressing Remove empties the list. **Fix:** store a `checked` flag per file (and in the session); make Remove act on highlighted rows.

### H2 — Error dialogs never appear (`gui.py:2646, 2767, 2886, 4833`) — Confirmed
`root.after(0, lambda: messagebox.showerror(... {e}))` runs after Python 3 has deleted `e`, so it raises `NameError`. Calibration, plate-solve, OSC and Data-Mining failures therefore show no dialog. **Fix:** `lambda err=e: ...` (as already done at line 1298).

### H3 — Bad numbers in settings fail silently (`gui.py:1045, 1193`) — Confirmed
Only `ValueError` is caught. Empty or invalid Tk fields raise `TclError`, so the user sees only a console traceback. `IntVar('3.7')` silently becomes 3. **Fix:** catch `tk.TclError`, validate through a single `build_config()`.

### H4 — Image calibration (`image_calibration.py`) — Confirmed
- **Naming:** the output is named `cal_<FILENAME header>`. Your V stack carries a subframe's FILENAME (`…_0028.fits`), so the GUI does not find `cal_AE_UMa_…` and silently keeps the **uncalibrated** file. Two stacks with the same inherited FILENAME overwrite each other. Name the output from the input path.
- **Clipping:** negative values are clipped and truncated to uint16, which biases faint sky and short/B exposures. Save as float32.
- **Flat not bias-subtracted:** the flat is normalised without subtracting bias/dark from it first.
- **Robustness:** a size mismatch only warns, then crashes; the "max ≤ 1.1 ⇒ × 65535" scaling is a guess; files are not closed.
- **Success check:** the GUI's success test only checks for an existing `cal_*` file, so leftovers from an earlier run count as success.

### H5 — Data mining: frames can be misaligned (`data_mining.py:148-202, 326-333`) — Confirmed (synthetic test)
If an exception is raised partway through a frame, the remaining stars get no entry, but the frame is still appended. Those stars' arrays get shorter, so later values are assigned to the wrong frames, or `IndexError` is raised. **Fix:** a NaN-initialised `(n_frames, n_stars)` array, with a per-star try/except.

---

## 4. Medium / low findings

| # | Where | Finding | Fix |
|---|---|---|---|
| M1 | `aperture_phot.py:23,57` | Stars at the image edge: the aperture sums only on-chip pixels but subtracts sky for the full area, giving strongly negative fluxes | Skip partial overlaps / `exclude_border` with margin ≥ r_out |
| M2 | `star_detection.py:54` | RA/Dec region filter never works (`SkyCoord` without units raises an error → silent fallback to 'all') | `unit=u.deg`, handle RA wrap |
| M3 | `calibration.py:741` | ZP uncertainty = scatter of individual ZPs (0.020) instead of error of the mean (≈0.0009); with N = 1 it is 0 | 1.2533·σ/√N (median) |
| M4 | `calibration.py:106-108` | ATLAS: if r is missing, V := g; if g is missing, B := r | NaN; also limit g−r to the validity range of the transformation |
| M5 | `calibra.py:100` | One corrupt file aborts the whole batch (only FileNotFoundError caught) | per-file try/except + traceback |
| M6 | `gui.py:2298` | Header editor writes in place, no backup; values change type ('007' → 7); NaN breaks the save; COMMENT/HISTORY edits silently dropped (data/BZERO survive) | `.bak` copy, only set changed cards, preserve types |
| M7 | `fits_viewer.py:407` | VSX cone capped at 45′; your frame is 144′×96′ (half-diagonal 87′) → only ~46% of the frame is searched | radius from WCS footprint |
| M8 | `plate_solve.py:56,144,184` | `.FIT`/`.fts` → `.wcs` path equals the FITS file itself; a "solved" result is any WCS object (always truthy); no timeout; stale `.wcs` reused | `splitext`, `has_celestial`, `timeout=`, delete old `.wcs` |
| M9 | `osc_splitter.py:205-219, 238` | Old `CDi_j` and SIP keywords are left next to the new PC/CDELT (2× scale conflict for ASTAP/Siril); float stacks (0–1) destroyed by uint16 cast; BAYERPAT/ROWORDER ignored | drop old WCS keys; keep float32 |
| M10 | `gui.py:3935, 4415` | Two `on_reset_ensemble` functions; the top button's version leaves the check-star index set | keep one |
| M11 | Viewer | Viewer photometry (centre-pixel counting, per-star flexible apertures, Gaussian-fit fallback when flux ≤ 0) is not comparable with pipeline/Light-Curve results | use photutils `exact` and one aperture per frame |
| M12 | Catalogue query | Default 15′ radius (60′ in your session) misses a large part of a 2.4° × 1.6° field | radius from WCS footprint |
| L | various | Duplicate separator row breaks the markdown table in the calibration report; report says "Local File (ATLAS refcat2)"; `psf_fitting` FWHM `abs((fx+fy)/2)` is wrong when the widths have opposite signs; all cached catalogues are treated as outdated and saved under two names (`_15` and `_15.0`); VSX distances computed without cos δ; `shift_analysis` ignores the configured radius; `\c` SyntaxWarnings in `color_calibration.py`; DAOStarFinder uses a fixed FWHM of 3.5 px; `FITSInspector.py` is broken (unused, can be deleted); Load Session duplicates file entries | — |

### Thread safety (structural)
Tkinter is not thread-safe, but worker threads touch the GUI directly in many places:
- `StdoutRedirector.write` inserts every `print` from every thread into Text widgets.
- `on_run` calls `pipeline_terminal.delete` and `run_btn.config`.
- `dm_thread` calls `dm_status_var.set`.
- Worker threads modify `loaded_files` while the GUI reads it.

This explains occasional freezes or odd crashes, if you ever see them. **Pattern:** workers put messages on a `queue.Queue`; a `root.after(50, drain)` loop on the main thread applies them. Read all Tk variables *before* starting a thread.

### Old AAVSO file
`photometry_output/aavso_AE_UMa.txt` was written by Calibra 2.0, with `OBSCODE=XXXX`, no check star and a pre-CHART format. Do not submit it as is; regenerate it with the current version.

---

## 5. Suggested improvements (beyond fixing bugs)

1. **Realistic errors everywhere:**
   - Use the effective gain (× STACKCNT), net flux and clipped-annulus pixel count.
   - Add an optional empirical noise factor from blank-sky apertures.
   - In data mining, judge variability with χ²_red or rms/median(σ_i) instead of the global log-RMS line.
2. **Zero point with colour term:** fit ZP + c·(B−V) with robust clipping; report σ/√N, c and χ².
3. **Instrumental magnitudes per second** (−2.5 log(F/EXPTIME)), so default ZPs are not tied to 120 s.
4. **Quality flags in all CSVs** (saturated, peak, edge, blended neighbour within r_out) instead of silently dropping stars. Add an AAVSO "fainter-than" limit when SNR < 3.
5. **One configuration builder** (`build_config()`) with validation, used by pipeline, Light Curves, Data Mining and Viewer, plus one `classify_filter()`.
6. **Measured FWHM fed back** into DAOStarFinder, and a flexible-aperture default of ~1.5 × FWHM (2 × FWHM adds sky noise).
7. **Faster PSF step:** use `TRFLSQFitter` with bounds (`LevMarLSQFitter` is deprecated), and fit only the stars actually needed.
8. **A small pytest suite** (filter classification, header round-trip, config building, synthetic-star photometry with known flux/noise). It would have caught most of the bugs above.

---

## 6. Proposed fix plan (v4.15)

1. **Quick, safe fixes (no change to science output):** H2, H3, M10, M5, M2, M8, the low-severity items, and a `.bak` copy in the header editor.
2. **Workflow correctness:** S5 (viewer close), S6 (`classify_filter`), H1 (checked state), H4 (calibration naming / float32 / bias-subtracted flat), H5.
3. **Science changes (alter results; to be checked against your data before adoption):** S1 (error model), S2 (colour-fit star selection and weighting), S3/S4 (ensemble ZP and consistent B−V in differential photometry), M1 (edge stars), M3 (ZP error), the saturation limit for stacks.
