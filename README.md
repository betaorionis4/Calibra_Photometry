# Calibra: Photometry of Variable Stars from FITS Images

**Version:** 5.0.1 — 2026-11-18  
**Developed by:** Stephan Pomp, Google DeepMind Antigravity and Claude (Anthropic)  
**Licence:** GNU General Public License v3 (see [LICENSE](LICENSE))

Calibra is a Python program with a graphical interface (tkinter) for photometry of monochrome (and split one-shot-colour) FITS images: zero points against catalogue stars, colour transformation for B and V, ensemble differential photometry, light curves with AAVSO-format reports, period analysis, an interactive FITS viewer, a search for unknown variables, an observation planner and a night-plan export for N.I.N.A.

The code was created with lots of help from Google Antigravity (various agents) and, from v4.14 on, Claude (Anthropic), guided, tested and debugged by me. I have made several tests to verify the results and ensure the results are reasonable. This code is far from what codes like, e.g., AIJ, ASTAP or Tycho Tracker can do. The purpose is simply to have a playground for understanding the principles of CCD/CMOS based photometry and having a tool to compare to what AIJ, ASTAP or Tycho Tracker provide as fluxes, zero points, etc. However, I have used the code to produce reports for AAVSO (time-series) with my observations as I am quite happy with the results so far.

---

## Main features

- **Zero points and colour transformation** from ATLAS-RefCat2, APASS (DR9/DR10), Gaia DR3 or Landolt stars; known variables (VSX) excluded automatically.
- **Light curves** (calibrated magnitudes, relative flux or raw ADU) with ensemble comparison stars, check star, full error propagation and AAVSO extended-format reports; forced photometry and frame combining for **faint targets**.
- **Field Variables**: light curves of many known variables in the same frames at once.
- **Period Analysis**: Lomb–Scargle, PDM, BLS, spectral window, prewhitening, phase plots, times of maximum and O–C.
- **Calibration frames**: master bias/dark/flat built and selected automatically, calibration on the fly.
- **Plate solving** with ASTAP (with a position guess from the object name).
- **Interactive FITS Viewer** with catalogue and VSX overlays, radial profiles and star marking.
- **Data Mining**: search for unknown variables from excess scatter.
- **Local catalogue copies** first, online services (VizieR, SIMBAD, AAVSO) only when needed.
- **Observation planner** and **night plan for N.I.N.A.** (simple sequencer).

## What's new in v5.0.1 (2026-11-18)

- Local catalogues really first: known variables come from the local GCVS list before VizieR (option *Prefer local catalogues*); local catalogue copies Calibra cannot read (e.g. the binary Cartes du Ciel Gaia catalogue) are skipped instead of costing ~30 s per reference-star list; GCVS lists load in ~1 s.
- Landolt standard stars read from the original CDS tables (as downloaded from VizieR/CDS) in the local catalogue folder.
- No more `FITSFixedWarning` messages (`RADECSYS` deprecated, `MJD-OBS` set from `DATE-OBS`) for frames with old-style headers; results unchanged.
- Tests pass on Windows.

## What's new in v5.0 (2026-11-18)

- New repository layout: program modules in `python/`, manuals in `docs/`, example data in `examples/`; GNU GPL v3 licence.
- Settings, caches and results are kept in a user folder (`~/Calibra`) instead of the program folder; settings and results of v4.x can be imported at the first start.
- Version number in one place (`python/version.py`), `pyproject.toml`, `requirements.txt`.
- Photometric methods and results are the same as in v4.46 (offline catalogues first). Full history: [CHANGELOG.md](CHANGELOG.md).

---

## Installation

1. Python 3.10 or newer with tkinter (on Debian/Ubuntu: `sudo apt install python3-tk`).
2. Download Calibra:
   ```bash
   git clone https://github.com/betaorionis4/Calibra_Photometry.git
   ```
   or *Code → Download ZIP* on [the GitHub page](https://github.com/betaorionis4/Calibra_Photometry) and unpack it.
3. Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Optional: [ASTAP](https://www.hnsky.org/astap.htm) with a star database for plate solving.

## Quick start

```bash
cd Calibra_Photometry
python calibra.py
```

Calibra can be started from any folder. Settings, caches and results are kept in a **user folder**, by default `~/Calibra` (on Windows `C:\Users\<name>\Calibra`):

```
~/Calibra/
├── config/     settings (calibra_session.json), instrument and target setups, local-catalogue settings, own lists
├── cache/      catalogue queries, AAVSO sequences, measurement caches (can be deleted)
├── output/     all results: light curves, AAVSO reports, period analyses, logs, plots
└── catalogs/   APASS DR10 downloads (when no local catalogue folder is set)
```

Another location can be chosen in *Settings → Operator & Session → Change User Folder…* or with the environment variable `CALIBRA_HOME`. Coming from v4.x: at the first start Calibra offers to copy settings and results from the old folder (also later via *Import from older Calibra…* or `python tools/migrate_user_data.py <old folder>`). See section 2.6 of the user manual.

## Example data

- [`examples/example_HADS_simulated.csv`](examples/example_HADS_simulated.csv): simulated light curve of a high-amplitude δ Scuti star, used in the period-analysis guide.
- [`examples/reference_stars.csv`](examples/reference_stars.csv): example of an own reference-star list (AAVSO sequence of the AE UMa field); copied to `config/` of the user folder at the first start.
- [`examples/HU_Aqr_MWT_2026-11-01/`](examples/HU_Aqr_MWT_2026-11-01/): three raw 30 s luminance frames of the polar HU Aqr, taken with the 0.42 m Margareta Westlund Telescope (SAAF) — see the README in that folder.

## Documentation

- **[User manual](docs/user_manual.md)** — principles (aperture photometry, PSF, noise model, colour transformation), every tab of the program, file formats.
- **[Period analysis guide](docs/period_analysis_guide.md)** — measuring periods in practice, with a worked example.
- **[CHANGELOG.md](CHANGELOG.md)** — all versions.

## Repository layout

```
Calibra_Photometry/
├── calibra.py                 start the program
├── gui.py                     graphical interface
├── calibra_logo.png
├── python/                    program modules (photometry, calibration, catalogues, user folder,
│                              light curves, period analysis, planner, NINA export, …)
│   └── version.py             version number and date (the only place they are set)
├── tools/
│   ├── catalog_overview.py    which local catalogue files can be deleted or reduced
│   └── migrate_user_data.py   copy settings and results of Calibra 4.x into the user folder
├── tests/test_calibra.py      automated tests (pytest)
├── docs/                      user manual, period analysis guide
├── examples/                  example light curve, FITS frames, reference-star list
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml             project description, dependencies, test settings
└── requirements.txt           dependencies for pip install -r
```

Nothing is written into the program folder while Calibra runs, except `calibra_home.txt` if you choose another user folder.

## Tests

```bash
pip install pytest
python -m pytest
```

## Questions and bug reports

Please use the [issue tracker](https://github.com/betaorionis4/Calibra_Photometry/issues).

## Licence

Calibra is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY. See [LICENSE](LICENSE).
