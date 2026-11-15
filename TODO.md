# Calibra — Ideas & To-Do List

Items discussed but deliberately **not yet implemented** (status 2026-11-04, v4.18).

## Under consideration
1. **Camera gain from the FITS header** — read `CVF`/`EGAIN` (e⁻/ADU) and `STACKCNT`, show them in the File Manager and warn if they differ from *Settings → Gain*.
   *Note:* the AE UMa stacks (Siril, ZWO ASI2600MM Pro) contain `CVF = 0.768` e⁻/ADU while Settings use 1.27 e⁻/ADU. The gain only enters the photon-noise term of the error bars (bright stars); faint-star errors are dominated by the measured sky noise. To be checked before acting.
8. **Interactive light-curve plot** — click a point to open its frame in the FITS Viewer or to mark it bad.
9. **Time window in Data Mining** — restrict the analysis to a JD range (uses the measurement cache).
10. **Settings help** — tooltips with units and "reset to defaults" for all settings.
11. **Easy start / installation** — `requirements.txt` and a double-click launcher.

## Optional new features
- **Period search** (Lomb–Scargle) and phase-folded plots for Data-Mining suspects and light curves.
- **AAVSO report check** before upload (missing CHART, check star, observer code, …).

## Performance (only if needed)
- Parallel processing of frames (helps only for hundreds of frames).
- Analysis-pipeline speed-ups: vectorised shift analysis (~7 s → <0.5 s per image), faster detection bookkeeping (~7 s → ~3.5 s), PSF fitting only for stars actually used.
- Catalogue cache that reuses any cached file covering the requested area (fewer downloads after small pointing changes).
