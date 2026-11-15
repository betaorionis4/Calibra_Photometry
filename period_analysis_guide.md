# Measuring Periods with Calibra – A Practical Guide

*Calibra v4.25 – Period Analysis tab. This guide explains how to determine the period(s) of a pulsating variable star from your light curves, how to judge whether one period describes the star, how to find and identify additional periods, and how to use times of maximum and the O–C diagram. The high-amplitude δ Scuti star (HADS) AE UMa is the running example, but the procedure applies to any periodic variable: δ Scuti and γ Doradus stars, RR Lyrae stars, Cepheids, Miras and semiregulars, and (with BLS/PDM) eclipsing binaries.*

---

## Contents

1. [What the Period Analysis tab does](#1-what-the-period-analysis-tab-does)
2. [The ideas behind it – in brief](#2-the-ideas-behind-it--in-brief)
3. [Planning observations for period work](#3-planning-observations-for-period-work)
4. [Step by step: a worked example (HADS, AE UMa-like)](#4-step-by-step-a-worked-example-hads-ae-uma-like)
5. [Is one period enough? – the checklist](#5-is-one-period-enough--the-checklist)
6. [Identifying additional periods](#6-identifying-additional-periods)
7. [Times of maximum and the O–C diagram](#7-times-of-maximum-and-the-oc-diagram)
8. [Notes for other types of variables](#8-notes-for-other-types-of-variables)
9. [Pitfalls and troubleshooting](#9-pitfalls-and-troubleshooting)
10. [Reporting your results](#10-reporting-your-results)
11. [Reference: controls, output files, formulas](#11-reference-controls-output-files-formulas)
12. [Literature](#12-literature)

---

## 1. What the Period Analysis tab does

| Task | Where in the tab |
|---|---|
| Load one or several light curves (Calibra CSV or any CSV with a time column) | **1. Data** |
| Periodogram: Lomb–Scargle, PDM, BLS; spectral window | **2. Periodogram** |
| Fit several frequencies at once, prewhitening, harmonics, S/N, notes (harmonic, combination, alias, period ratio) | **3. Frequencies** |
| How well do the frequencies describe the data? (rms vs. errors, χ², night offsets) | quality line below the frequency table |
| Phase plot (fit / binned mean / none), residuals vs. time, O–C diagram | bottom plot (selector buttons directly above the plots) |
| Publication-quality phase plot: edit labels, ranges, symbols; save PNG/PDF/SVG and the folded data (CSV) | *Phase Plot Window (edit / save)…* |
| Times of maximum, O–C, corrected ephemeris, period change | **4. Times of maximum and O–C** |
| Save frequency table, figure, periodogram, residuals, times of maximum | *Save Results…*, *Save Maxima (CSV)…* |

Time is always in days (HJD, as written by Calibra's light-curve functions), frequency in cycles per day (c/d), and the period is P = 1/f. For practice, the data folder contains **`example_HADS_simulated.csv`**: a *simulated* light curve (not real data) of a HADS star with AE UMa-like parameters, 9 nights, 2-min cadence. All numbers in Section 4 come from this file, so you can repeat every step.

---

## 2. The ideas behind it – in brief

### 2.1 Every periodic light curve is a sum of sines

Any light curve that repeats exactly with period P can be written as a **Fourier series**:

  m(t) = m₀ + A₁ sin(2π f t + φ₁) + A₂ sin(2π·2f t + φ₂) + A₃ sin(2π·3f t + φ₃) + …

with f = 1/P. The terms at 2f, 3f, … are the **harmonics**. A sinusoidal light curve has only A₁; a δ Scuti or RR Lyrae light curve with its steep rise and slow decline ("sawtooth") has strong harmonics; an eclipsing binary with narrow eclipses has many harmonics of similar size. Consequences:

- The periodogram of a non-sinusoidal star shows peaks at **f, 2f, 3f …** – these are *not* additional periods, they describe the shape.
- In Calibra you either add the harmonics as separate frequencies (prewhitening finds them and marks them `2f1`, `3f1`) or you give a frequency several **harmonics** (*Harmonics for a new frequency* / *Apply to Selected*). Both give the same model.
- The amplitudes A are **semi-amplitudes**. For a pure sine the full (peak-to-peak) range is 2A; for a non-sinusoidal curve read the range from the phase plot.

### 2.2 The periodogram

- **Lomb–Scargle** fits a sine at every trial frequency by least squares; Calibra shows the amplitude of that sine (in mag or flux). It is the standard method for unevenly sampled data and is equivalent to a Fourier amplitude spectrum. It finds the fundamental of almost any shape. *harmonics* > 1 in section 2 fits sine + harmonics at every trial frequency (good for strongly non-sinusoidal shapes).
- **PDM** (phase dispersion minimisation) folds the data with every trial period and measures how tight the folded curve is – no assumption about the shape. Good as an independent check, for eclipsing binaries and for semiregular stars. Slower.
- **BLS** (box least squares) searches for short dips – for eclipsing binaries with narrow eclipses.

### 2.3 Resolution, Nyquist frequency, aliases

- **Frequency resolution ≈ 1/T** (T = total time span). Peaks are about 1/T wide. Two frequencies closer than about **1.5/T** cannot be separated reliably. The *uncertainty* of a well-defined frequency is much smaller than 1/T (Section 11.3).
- **Longest measurable period ≈ T/2** – better: cover at least two cycles, preferably many.
- **Pseudo-Nyquist frequency ≈ 0.5 / (median spacing)**: the highest frequency that can be found reliably.
- **Aliases**: gaps in the data create false peaks at f ± 1/(gap period). With observations from one site every night, every peak has side peaks at **f ± 1 c/d, ± 2 c/d …** (daily aliases); with observations once a week at f ± 1/7 c/d; with seasons at f ± 1/365 c/d. The **spectral window** (*Show Spectral Window*) is the periodogram of a perfect sine sampled like your data – its pattern accompanies every real peak. If the alias peak is almost as high as the true one, the decision can be wrong: more nights, longer nights or data from other longitudes help.

### 2.4 Significance

- A peak is usually accepted as real if its **S/N ≥ 4** (Breger et al. 1993): amplitude divided by the mean amplitude of the *residual* periodogram (after removing all fitted frequencies) within ±1 c/d of the peak. Calibra computes this for every fitted frequency and marks S/N < 4.
- For the highest peak of the first periodogram Calibra also gives the **false-alarm probability** (FAP): the probability that pure noise produces such a high peak somewhere in the searched range.
- Statistical significance is not the same as astrophysical reality: zero-point jumps between nights, a variable comparison star or extinction effects can produce perfectly "significant" peaks (Section 9).

---

## 3. Planning observations for period work

| Item | Recommendation (HADS example, P ≈ 2 h) | General rule |
|---|---|---|
| Exposure time | 30–60 s (plus readout) | ≤ P/50 when harmonics matter (an exposure of length τ reduces the amplitude of frequency f by sinc(π f τ); for τ = P/50 the 4th harmonic loses ~1 %) |
| Cadence | ≤ 2 min, i.e. ≥ 40 points per cycle | ≥ 20 points per cycle for shapes with harmonics; for times of maximum ≥ 8–10 points around each maximum |
| Length of a night | ≥ 1 full cycle, preferably 2–3 | the longer, the better against 1 c/d aliases |
| Number of nights / time span | 3–10 nights over 1–4 weeks for a period and a second mode; months to years for period changes | T ≥ 1.5/Δf to separate two frequencies Δf apart; T ≥ 2–3 P for long periods |
| Photometry | relative flux or calibrated magnitudes with the **same comparison stars** every night | keep aperture, gain, stack mode identical; check the check star |
| Filter | V, B or L; do not mix filters in one analysis | amplitudes and phases differ between filters |

Use the **Light Curves** tab (or **Field Variables** for several stars) with the images of all nights checked; Calibra writes one CSV with HJD, magnitude/flux, error and quality flag. You may also analyse each night separately and combine the CSV files in the Period Analysis tab.

---

## 4. Step by step: a worked example (HADS, AE UMa-like)

**About the star.** AE UMa is a high-amplitude δ Scuti star (HADS), V ≈ 10.9–11.5, with a fundamental-mode period of about 0.0860 d (f₀ ≈ 11.626 c/d, ~2.06 h). It is also known to pulsate in the first radial overtone (f₁ ≈ 15.03 c/d; period ratio ≈ 0.773). The practice file `example_HADS_simulated.csv` was generated with these frequencies, a sawtooth-like shape (three harmonics), the overtone with 0.030 mag, a combination frequency f₀+f₁ with 0.010 mag, photometric noise of 6–8 mmag and small night-to-night zero-point shifts (3 mmag). Your own results for AE UMa (or any other star) will of course differ.

### Step 1 – Load the data (section 1)
*Load Light Curve(s)…* → `example_HADS_simulated.csv`. Quantity `mag`; *only OK points* on; **do not** tick *subtract the mean of each file* (only one file here; for several files with the same comparison stars leave it off as well).

The summary line reads:

> N = 1170   span T = 36.18 d   nights = 9   median spacing = 2.0 min   resolution 1/T = 0.0276 c/d   pseudo-Nyquist = 358 c/d

So frequencies closer than ~0.04 c/d cannot be separated, and everything up to the automatic upper limit of 100 c/d is accessible.

### Step 2 – First periodogram (section 2)
Method *Lomb–Scargle*, harmonics 1, empty period limits, oversampling 10 → *Compute Periodogram*:

> highest peak f = 11.6257 c/d (P = 0.086016 d), false-alarm probability 0

Click on the peak to see a preview phase plot – a clean, asymmetric (sawtooth-like) light curve. Press *Show Spectral Window*: with 9 nights from one site the window has side lobes at ±1 c/d with 95 % and at ±2 c/d with 80 % of the central peak. Every real peak is accompanied by this pattern – an alias only 5 % lower than the true peak is why a wrong choice is possible with noisier data.

### Step 3 – Prewhitening (section 3)
Press *Prewhiten: Add Highest Peak* repeatedly. After each step all frequencies are fitted together and the periodogram shows what is left. The example gives:

| step | new peak (c/d) | amplitude (mag) | note shown by Calibra | rms/error after the fit |
|---|---|---|---|---|
| 1 | 11.62569 | 0.200 | – (main frequency, f1) | 10.1 |
| 2 | 23.2513 | 0.080 | `2f1` | 5.8 |
| 3 | 34.8771 | 0.041 | `3f1` | 4.0 |
| 4 | 15.0314 | 0.030 | `with f1: period ratio 0.7734: typical of F + 1O of delta Sct / HADS` | 2.5 |
| 5 | 46.5027 | 0.020 | `4f1` | 1.46 |
| 6 | 26.6572 | 0.010 | `f1+f4` | **1.06** |

The highest remaining peak is at 0.21 c/d. Adding it gives S/N ≈ 3.8 (marked *not significant*) – remove it again (*Remove Selected/Last*). Its very low frequency (< 10/T) points to the small night-to-night offsets, not to pulsation (see the *Residuals vs time* plot). Stop here.

Interpretation: f1 = 11.6257 c/d is the fundamental mode; 2f1, 3f1, 4f1 describe the sawtooth shape; f4 = 15.031 c/d is an independent mode with the period ratio of the first radial overtone; f1+f4 is a combination frequency, which shows that both modes belong to the same star (and is typical of large-amplitude pulsators).

**Equivalent and tidier:** *Reset*, set *Harmonics for a new frequency* = 4, add the highest peak (f₀ with 4 harmonics), then set harmonics back to 1 and continue: the next peaks are f₁ = 15.0314 and f₀+f₁ = 26.657. The table then has three rows instead of six.

### Step 4 – Check the fit quality
The quality line below the table now reads:

> residual rms = 0.0074 mag   median photometric error = 0.0070 mag   rms/error = 1.06   reduced chi2 = 1.12
> → the fitted frequencies describe the data to within the noise.
> Night-to-night scatter of the mean residual: 0.0025 mag (9 nights)

and the plot button *Residuals vs time* (above the plots) shows residuals without structure, with night means scattered by a few mmag (the simulated zero-point shifts). For comparison: with f₀ alone (pure sine) rms/error was 10.1, with f₀ and 4 harmonics 3.4 – the overtone is clearly needed.

### Step 5 – Phase plots
Select a row in the table to see its phase plot (other frequencies subtracted; harmonics of the selected frequency are kept so that the true shape is visible). *Curve in phase plot*: **Fit** (red: the model with its harmonics), **Binned mean** (model-free median curve), **None**. For f₀ you see the steep rise and slow decline; for f₁ a nearly sinusoidal curve with 0.03 mag semi-amplitude.

For a figure to keep or publish, press *Phase Plot Window (edit / save)…* above the plots: choose the frequency, tick *phase 0 = maximum light*, set the phase range to −0.5 … 1.5 (maximum in the middle), adjust title, labels, y range and symbols, and save as PNG, PDF or SVG; *Save Phase Data (CSV)* exports the folded points and the model.

### Step 6 – Times of maximum and O–C (section 4)
Select f1 (fundamental), method *Template (fitted shape)*, ephemeris *from the fit* → *Times of Maximum / O-C*:

> 18 times of maximum of f1 (template fit), median error 0.11 min
> O-C rms = 0.10 min, reduced chi2 = 0.98. Corrected ephemeris: T0 = HJD 2461155.87708 ± 0.00002, P = 0.08601642 ± 0.00000011 d

O–C scatters around zero within its errors (χ² ≈ 1): the period is constant over the 36 days and agrees with the Fourier result. *Save Maxima (CSV)…* writes the times of maximum, e.g. to compare with published maxima (Section 7).

### Step 7 – Save
*Save Results…* writes the frequency table (with all settings in the header), the figure, the periodogram and the residuals.

---

## 5. Is one period enough? – the checklist

Work through these points after fitting the main frequency **with enough harmonics** (add harmonics until the next one has S/N < 4):

1. **Residual rms versus photometric error** (quality line).
   - rms/error ≈ 1 (0.8–1.3), reduced χ² ≈ 1: the model describes the data within the noise – nothing else is detectable *with these data*.
   - 1.3–2: something is left – look at the residual periodogram and the residual plot.
   - > 2: clearly more signal: further frequencies, missing harmonics, or zero-point problems.
   - < 0.8: the error bars are too large (or you fitted too many frequencies).
   Calibra's errors include photon, sky and read noise but not scintillation or flat-field errors; for bright stars and short exposures scintillation can dominate, so rms/error of 1.2–1.5 may still mean "noise only" – compare with the check star's scatter.
2. **Residual periodogram**: any peak with S/N ≥ 4 that is not a harmonic is a candidate for another period (Section 6). An empty residual periodogram at the level of the noise means one period (+ harmonics) is enough.
3. **Phase plot with *Binned mean***: with one period all nights fall on one thin curve. Extra scatter concentrated at maximum light (heights of successive maxima differ) is the signature of **beating** with a second mode. A whole night shifted up or down points to a zero-point problem.
4. **Residuals vs time**: structure within a night (a wave pattern) = unfitted mode; night means scattered more than the errors = zero-point differences between nights (comparison stars, flat fields, extinction/colour effects) or a genuine slow variation.
5. **O–C**: with a single constant period the O–C values scatter around a straight line with χ² ≈ 1. Systematic wiggles with a period of the beat period 1/|f₁ − f₀| show that a second mode was not subtracted.
6. **Stability**: split the data (e.g. first and second half, or season by season – load them as separate files or save the results separately). Frequency, amplitude and phase of the main mode should agree within the errors.

Only if all six points are passed is "one period" an adequate description – always *at the precision of your data*: a second mode with 2 mmag amplitude is invisible in data with 10 mmag noise and few points.

**Detection limit.** The noise level of the residual amplitude spectrum, roughly σ·√(π/N) (σ = rms, N = number of points), sets what can be found: for S/N ≥ 4 the amplitude must exceed about 4σ√(π/N). Example: σ = 7 mmag, N = 1170 → noise ≈ 0.36 mmag → detection limit ≈ 1.5 mmag. Doubling the data lowers the limit by √2.

---

## 6. Identifying additional periods

After prewhitening, each remaining significant peak falls into one of these classes. Calibra's *Note* column flags the first five automatically.

| Class | How to recognise it | Calibra note | Meaning |
|---|---|---|---|
| Harmonic | f = n·f₁ (within the resolution) | `2f1`, `3f1` … | shape of the light curve, not a new period |
| Combination | f = f₁ + f₂, f₂ − f₁, 2f₁ + f₂ … | `f1+f2`, `-f1+2f2` | non-linear coupling of two modes; confirms that both belong to the same star |
| Daily / yearly alias | f differs from a fitted frequency by 1 or 2 c/d (or 1/365 c/d) | `f1+1 c/d alias?` | probably a remainder of an imperfectly removed frequency; check the spectral window; refit or try the alias as main peak |
| Low frequency | f < 2/T | `< 2/T: trend or night offsets?` | drift, zero-point differences between nights, or a very long period; not determinable from these data |
| Independent mode with known ratio | P_short/P_long in a known range | `period ratio 0.7734: typical of …` | radial double-mode pulsation (table below) |
| Close peak | within a few 1/T of a fitted frequency | – | amplitude or phase modulation (Blazhko effect, beating of close non-radial modes), a slowly changing period, or an unresolved pair – needs a longer time span |
| Other independent frequency | none of the above | – | further (often non-radial) pulsation mode – common in δ Scuti and γ Dor stars |

**Period ratios of radial double-mode pulsators** (shorter / longer period), used by Calibra for the note:

| Ratio | Typical of |
|---|---|
| 0.740–0.750 | RR Lyrae stars pulsating in fundamental + first overtone (RRd) |
| 0.756–0.790 | δ Scuti / HADS: fundamental + first overtone (AE UMa ≈ 0.773) |
| 0.695–0.725 | Cepheids: fundamental + first overtone |
| 0.795–0.815 | first + second overtone (δ Scuti, Cepheids) |

The exact ratio depends on mass and metallicity (the "Petersen diagram"); a ratio inside a range is a strong hint, not a proof.

**Practical rules**

- Add one frequency at a time and look at the residual periodogram after each step – a frequency found early can be an alias of the true one, and the joint fit can then become unstable (a frequency moves by ~1 c/d). Try the alternative and compare rms and phase plots.
- Prefer harmonics over many separate frequencies once you know the main frequency (Section 4, step 3).
- Treat every peak with S/N between 3.5 and 5 as uncertain until confirmed by more data or by an independent data set (e.g. another season).
- A frequency is more credible if it appears in each half of the data separately, with consistent amplitude.
- Check that the comparison and check stars are constant: a variable comparison star appears as an extra frequency in the target (Section 9).

---

## 7. Times of maximum and the O–C diagram

### 7.1 Why times of maximum?
The Fourier fit gives the best *mean* period for your data. Times of maximum (or minimum) of individual cycles let you combine your data with observations from other years and observers, detect slow period changes and light-travel-time effects, and they are the classical way of following pulsators such as HADS, RR Lyrae stars and Cepheids over decades.

### 7.2 How Calibra determines them
For the selected frequency (default f1) Calibra first subtracts all other, non-harmonic frequencies (otherwise a second mode shifts the maxima), then for every cycle with enough points on **both sides** of the expected maximum:

- **Template (recommended)**: the fitted light-curve shape of that frequency (with its harmonics) is shifted in time and brightness to the points within ±*window* (in phase) of the expected maximum. Works for any shape, uses all points of the cycle, and is unbiased for asymmetric (sawtooth) maxima. Fit enough harmonics first.
- **Parabola**: a parabola fitted to the points within ±window/2 around the brightest point – the classical method; needs a well-covered, sharp maximum and is biased for asymmetric light curves (in the example its O–C scatter was ~10× larger than with the template).

Settings: *min. points per maximum* (default 8) and *window* (±0.3 in phase). Errors are formal least-squares errors; check them against the O–C scatter (χ²).

### 7.3 The ephemeris and O–C
O–C = observed time of maximum − calculated time, with the calculated time from the **ephemeris** T(E) = T₀ + E·P (E = cycle number).

- *Ephemeris from the fit*: T₀ = maximum near the middle of the data, P from the Fourier fit.
- *Given*: e.g. from the GCVS/VSX (epoch and period – the GCVS epoch of a pulsator is a time of maximum; for eclipsing binaries it is a minimum) or from the literature. Enter the full HJD (reduced dates like 55515.739 are completed by +2400000). Time systems: Calibra uses HJD (UTC); literature ephemerides are sometimes in BJD(TDB) – the difference is about 1 min and matters for precise O–C work.

### 7.4 Reading the O–C diagram

| O–C pattern | Interpretation |
|---|---|
| scatter around 0, χ² ≈ 1 | ephemeris correct, period constant |
| straight line with slope b | the period is P + b (Calibra reports the corrected T₀ and P) |
| parabola | the period changes: dP/dE = 2c (Calibra reports dP/dE, dP/dt and (1/P)dP/dt) – evolutionary changes in HADS, RR Lyrae stars and Cepheids are typically of order 10⁻⁷–10⁻⁹ per year in (1/P)dP/dt and need decades of data |
| sinusoidal wave over years | light-travel time in a binary system (or a long-period modulation) |
| wave with the beat period of two modes | second mode not subtracted |
| sudden jump | a period change, or a wrong cycle count across a gap |

**Cycle-count ambiguity.** Across a long gap the number of cycles must be known exactly. The prediction error after N cycles is N·σ_P; it must stay well below P/4. In the example σ_P = 1.1·10⁻⁷ d, P = 0.086 d → N < 2·10⁵ cycles (≈ 50 years). With a single night's period (σ_P of ~10⁻⁵ d) a gap of a few hundred cycles (a few weeks) can already be ambiguous – that is why intermediate nights are valuable.

To combine your maxima with published ones, save them (*Save Maxima (CSV)…*), add the literature maxima to the file (same columns), and fit the O–C in a spreadsheet, or enter the literature ephemeris as *given* in Calibra.

---

## 8. Notes for other types of variables

| Type | Typical periods / amplitudes | What to watch |
|---|---|---|
| **δ Scuti (low amplitude)** | 0.02–0.25 d, mmag–0.1 mag | many, often close non-radial frequencies; needs long time spans (weeks–months) and S/N analysis; 1 c/d aliases are the main danger; combination frequencies are rare at low amplitude |
| **HADS** | 0.05–0.25 d, > 0.3 mag | sawtooth → 3–5 harmonics; possible first-overtone mode (ratio ≈ 0.77); combination frequencies; times of maximum ideal for O–C |
| **SX Phe** | like HADS, Population II | ratio F/1O ≈ 0.78; often period changes |
| **γ Doradus** | 0.3–3 d, mmag–0.1 mag | several close low frequencies around 1 c/d → severe daily aliasing from one site; long, dense campaigns |
| **RR Lyrae (RRab)** | 0.45–1 d, 0.5–1.5 mag | very steep rise → many harmonics (5–10); **Blazhko effect**: side peaks at f ± f_B (and nf ± f_B) around the main frequency and its harmonics, amplitude and phase modulation with periods of days to years; only a few cycles per night → many nights needed; times of maximum are standard |
| **RR Lyrae (RRc)** | 0.2–0.45 d | nearly sinusoidal; sometimes additional non-radial modes at P/P₁ ≈ 0.61–0.63 |
| **RRd** | two modes, ratio ≈ 0.745 | identify both and the combinations |
| **Cepheids** | 1–50+ d, 0.3–1 mag | one or a few points per night are enough; the time span must cover several cycles; double-mode Cepheids with ratio ≈ 0.70–0.71 (F/1O) or ≈ 0.80 (1O/2O); periods change slowly (O–C over decades) |
| **Mira and semiregular (SR)** | 80–1000 d, up to several mag | one point every few days, span ≥ 2–3 cycles; use 1 harmonic first; periods and amplitudes vary from cycle to cycle → broad peaks; SR stars often show two periods (and a "long secondary period"); times of maximum from the binned/smoothed light curve, O–C with large scatter; **do not subtract per-file means** |
| **Eclipsing binaries** (not pulsating, but periodic) | hours to days | use BLS or PDM; Lomb–Scargle often finds **P/2** (two similar minima per cycle) – check the phase plot at 2P; times of minimum instead of maximum (fit the inverted light curve, e.g. flux instead of magnitude) |

---

## 9. Pitfalls and troubleshooting

- **Peak at 1 c/d (or 2, 3 c/d) or at f ± 1 c/d**: sampling aliases or a nightly systematic effect (extinction, colour terms, temperature-dependent flats). Compare with the spectral window; check the residuals vs time.
- **Low-frequency peaks (< 2/T)**: zero-point offsets between nights, a slow drift, a variable comparison star. Check the night means in *Residuals vs time*, the check-star light curve, and whether the same comparison stars were used every night. *Subtract the mean of each file* removes offsets between files – but also a real slow variation.
- **Variable comparison star**: produces a "target" frequency that is also present (with opposite sign) in the check star relative to the ensemble. Analyse the check star's light curve as well (Calibra writes `check_mag` / `check_flux` into the CSV: choose it as *Quantity*).
- **Colour and airmass**: a red target and blue comparison stars give brightness changes with airmass, i.e. with the time of night → spurious peaks near 1 c/d. Use comparison stars of similar colour, especially in L.
- **Wrong main peak (alias chosen)**: the joint fit becomes unstable or a frequency jumps by ~1 c/d; the phase plot shows two branches. Try the neighbouring peak (click it, *Add Preview Frequency*) and compare rms and phase plots.
- **Exposure too long**: amplitudes of harmonics are reduced, maxima flattened; times of maximum barely affected.
- **Too few points per cycle**: harmonics above the pseudo-Nyquist frequency fold back (aliasing) and appear at wrong frequencies.
- **Over-fitting**: many frequencies with S/N just above 4 and rms/error < 1 – remove the weakest ones and compare.
- **Time system**: all your files must use the same time (Calibra: HJD mid-exposure). Do not mix JD and HJD (differences up to 8 min).

---

## 10. Reporting your results

Give for each frequency: f (c/d) and σ_f, P (d) and σ_P, semi-amplitude A and σ_A (filter / quantity), S/N, identification (harmonic, combination, mode). Also give: N, time span, number of nights, filter, comparison stars, residual rms, and the method (Calibra Period Analysis, Lomb–Scargle + least-squares fit). The uncertainties of Section 11.3 are lower limits; doubling them is a common conservative practice for correlated noise.

For times of maximum give HJD, error, method and filter. Collections of times of maxima exist for many pulsating stars and eclipsing binaries (e.g. those maintained by variable-star organisations such as the BAV or the GEOS RR Lyrae database); check their submission rules. The AAVSO accepts the photometry itself (Calibra's AAVSO extended-format reports).

---

## 11. Reference: controls, output files, formulas

### 11.1 Controls
- **Data**: *Quantity* (any numeric column, e.g. `mag`, `flux`, `check_mag`); *only OK points*; *subtract the mean of each file*.
- **Periodogram**: method (Lomb–Scargle / PDM / BLS), *harmonics* (Lomb–Scargle with n terms), period range (empty = automatic), oversampling (grid step 1/(oversampling·T)), x-axis as period, *Curve in phase plot* (Fit / Binned mean / None). Click into the periodogram → preview phase plot.
- **Above the plots**: *Phase plot / Residuals vs time / O-C diagram* buttons; *Phase Plot Window (edit / save)…* (frequency, curve, bins, title, labels, phase and y range, point/font size, error bars, colour per file, inverted y, grid, legend, phase 0 = maximum light; save PNG/PDF/SVG and CSV). Selecting a frequency in the table returns to the phase plot.
- **Frequencies**: *Prewhiten: Add Highest Peak*, *Add Preview Frequency*, *Add Frequency…* (value in c/d, or period with `d`, e.g. `0.0860164d`), *Remove Selected/Last*, *Reset*, *Harmonics for a new frequency* / *Apply to Selected*; quality line.
- **Times of maximum and O–C**: method (Template / Parabola), min. points, window, ephemeris (from the fit / given T₀, P), parabola fit (period change); *Save Maxima (CSV)…*.

### 11.2 Output files (default folder `photometry_output/period_analysis/`)
- `periods_<name>.csv`: frequency table; header lines with data and settings; columns id, harmonic, frequency, error, period, error, amplitude, error, phase, error, S/N, note.
- `periods_<name>.png`: the current figure.
- `periods_<name>_periodogram.csv`, `periods_<name>_residuals.csv` (HJD, value, model, residual, file).
- `maxima_<name>.csv`: E, HJD of maximum, error, O–C, number of points, method; header with ephemeris, corrected ephemeris and period change.
- `phase_<name>.png/.pdf/.svg` and `phase_<name>.csv` (Phase Plot Window): figure; HJD, phase, value, error, file, model.

### 11.3 Formulas
- Model: y(t) = c + Σ_k Σ_h A_kh sin(2π (h f_k (t − t₀) + φ_kh)), fitted by non-linear least squares (frequencies, amplitudes, phases), weighted with the photometric errors.
- Uncertainties (Montgomery & O'Donoghue 1999), N points, residual rms σ, time span T:
  σ_f = √(6/N) · σ / (π T A),  σ_A = √(2/N) · σ,  σ_φ = σ_A / A (radians).
  σ_P = σ_f / f².
- S/N = A / ⟨residual amplitude spectrum within ±1 c/d⟩ (Breger et al. 1993); significant if ≥ 4.
- Fit quality: rms/median error; reduced χ² = Σ(res/σᵢ)² / (N − n_par).
- O–C: O−C = T_max − (T₀ + E P); straight line → P' = P + b, T₀' = T₀ + a; parabola O−C = a + bE + cE² → dP/dE = 2c, dP/dt = 2c/P.

---

## 12. Literature

- Breger, M., et al. 1993, A&A 271, 482 – the S/N ≥ 4 criterion.
- Lomb, N. R. 1976, Ap&SS 39, 447; Scargle, J. D. 1982, ApJ 263, 835 – the Lomb–Scargle periodogram.
- VanderPlas, J. T. 2018, ApJS 236, 16 – "Understanding the Lomb–Scargle periodogram" (very readable introduction).
- Stellingwerf, R. F. 1978, ApJ 224, 953 – Phase Dispersion Minimisation.
- Kovács, G., Zucker, S., Mazeh, T. 2002, A&A 391, 369 – Box Least Squares.
- Montgomery, M. H., O'Donoghue, D. 1999, Delta Scuti Star Newsletter 13, 28 – uncertainties of frequency, amplitude and phase.
- Baluev, R. V. 2008, MNRAS 385, 1279 – false-alarm probability.
- Lenz, P., Breger, M. 2005, Comm. Asteroseismology 146, 53 – Period04 (the reference program for prewhitening; Calibra's light-curve and `_residuals.csv` files can be imported there via their time and value columns).
- Sterken, C. (ed.) 2005, ASP Conf. Ser. 335 – "The Light-Time Effect in Astrophysics"; includes a tutorial on the O–C diagram.
- AAVSO: *Guide to CCD/CMOS Photometry* and the *VSTAR* tutorials (free at aavso.org).
