# Color Transformation Calibration Report

Analyzed 1016 common stars.
Applied 2-sigma iterative outlier rejection.
Applied Extinction Correction:
- B-Filter: k=0.35, X=1.055
- V-Filter: k=0.20, X=1.109

## Derived Coefficients (Cleaned)
- **$\mu$ (Color Scale):** 0.9476  (R=0.928)
- **$\psi$ (B-Term):** 0.1588  (R=0.437)
- **$\epsilon$ (V-Term):** 0.0102  (R=0.078)

## Transformation Equations
Using these coefficients, your calibrated magnitudes are:
1. $(B-V)_{std} = 0.948 \cdot (b-v)_{corr} + -0.134$
2. $B_{std} = b_{corr} + 0.159 \cdot (B-V)_{std} + 23.967$
3. $V_{std} = v_{corr} + 0.010 \cdot (B-V)_{std} + 24.249$
*(Note: v_corr and b_corr are the instrumental magnitudes corrected for extinction)*

![Color Diagnostic Plots](color_plots.png)
