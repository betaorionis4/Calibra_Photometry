# Positional Shift Analysis

Here is a breakdown of how the positions of the matched stars deviate from their expected reference catalog coordinates. The shifts are calculated as `Detected Position - Reference Position`.

## Individual Star Deviations

| Reference ID | Detected ID | dRA (arcsec) | dDec (arcsec) | dX (pixels) | dY (pixels) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **online_0** | Auto_350 | 0.01 | 0.15 | 0.08 | -0.10 |
| **online_3** | Auto_854 | 0.03 | -0.06 | 0.01 | 0.04 |
| **online_5** | Auto_749 | 0.02 | 0.15 | 0.02 | -0.06 |
| **online_11** | Auto_420 | 0.19 | 0.17 | 0.18 | -0.02 |
| **online_13** | Auto_025 | -1.08 | -0.85 | -1.00 | 0.12 |
| **online_19** | Auto_235 | 0.39 | 0.29 | 0.31 | 0.03 |
| **online_21** | Auto_122 | -0.05 | 0.04 | -0.06 | -0.04 |
| **online_39** | Auto_665 | 0.37 | 0.31 | 0.34 | -0.03 |
| **online_40** | Auto_585 | 0.57 | 0.39 | 0.28 | 0.01 |
| **online_53** | Auto_824 | 0.25 | -0.19 | 0.11 | 0.15 |
| **online_56** | Auto_148 | 0.54 | 0.28 | 0.42 | 0.02 |
| **online_61** | Auto_100 | 0.25 | 0.33 | 0.26 | -0.10 |
| **online_73** | Auto_868 | -0.11 | 0.08 | 0.13 | -0.17 |
| **online_76** | Auto_145 | -0.31 | -0.20 | -0.27 | -0.00 |
| **online_83** | Auto_718 | 0.34 | -0.09 | 0.18 | -0.09 |
| **online_87** | Auto_880 | 0.11 | 0.22 | 0.01 | -0.06 |
| **online_89** | Auto_346 | -0.02 | 0.10 | -0.03 | -0.01 |
| **online_93** | Auto_138 | -0.07 | 0.14 | 0.01 | -0.11 |
| **online_95** | Auto_843 | -0.62 | -0.41 | -0.61 | -0.01 |
| **online_96** | Auto_504 | -0.05 | 0.27 | 0.05 | -0.17 |
| **online_97** | Auto_050 | 0.02 | 0.10 | 0.04 | -0.07 |
| **online_98** | Auto_050 | 1.14 | 2.20 | 1.53 | -0.91 |
| **online_99** | Auto_745 | -0.14 | 0.28 | -0.03 | -0.21 |
| **online_102** | Auto_215 | 0.07 | -0.04 | -0.01 | 0.06 |
| **online_104** | Auto_549 | 0.34 | 0.14 | 0.28 | 0.03 |
| **online_107** | Auto_056 | -0.01 | -0.06 | -0.04 | 0.00 |
| **online_113** | Auto_364 | 0.29 | 0.19 | 0.23 | -0.00 |
| **online_118** | Auto_146 | 0.31 | 0.18 | 0.23 | 0.02 |
| **online_122** | Auto_847 | 0.38 | 0.40 | 0.35 | -0.29 |
| **online_124** | Auto_163 | 0.35 | 0.35 | 0.33 | -0.06 |
| **online_125** | Auto_338 | 0.26 | 0.38 | 0.25 | -0.11 |
| **online_126** | Auto_632 | 0.38 | 0.33 | 0.37 | -0.07 |
| **online_128** | Auto_007 | 0.26 | 0.35 | 0.29 | -0.09 |
| **online_131** | Auto_172 | 0.33 | 0.36 | 0.35 | -0.10 |
| **online_139** | Auto_179 | -0.00 | 0.21 | 0.05 | -0.10 |
| **online_148** | Auto_480 | 0.24 | 0.24 | 0.26 | -0.11 |
| **online_150** | Auto_447 | -0.07 | 0.12 | -0.04 | -0.06 |
| **online_152** | Auto_225 | 0.22 | 0.26 | 0.23 | -0.07 |
| **online_155** | Auto_747 | 0.39 | 0.34 | 0.37 | -0.07 |
| **online_158** | Auto_239 | 0.00 | 0.17 | -0.01 | -0.08 |
| **online_163** | Auto_536 | 0.14 | 0.08 | 0.11 | -0.00 |
| **online_167** | Auto_503 | 0.09 | 0.23 | 0.15 | -0.11 |
| **online_169** | Auto_750 | 0.50 | -0.12 | 0.36 | -0.16 |
| **online_175** | Auto_343 | 0.11 | 0.21 | 0.15 | -0.08 |
| **online_178** | Auto_039 | 0.17 | 0.47 | 0.27 | -0.20 |
| **online_180** | Auto_251 | 0.13 | 0.24 | 0.17 | -0.10 |
| **online_184** | Auto_354 | 0.34 | 0.26 | 0.29 | -0.03 |
| **online_186** | Auto_791 | 0.37 | 0.50 | 0.40 | -0.06 |
| **online_187** | Auto_305 | 0.41 | 0.37 | 0.37 | -0.05 |

## Overall Statistics

> [!NOTE]
> **Conclusion on Systematic Shifts**
> The detected stars are, on average, offset by **+0.18 pixels** (X) and **-0.06 pixels** (Y).
> In celestial coordinates, this is a shift of **+0.19 arcsec** (RA) and **+0.21 arcsec** (Dec).

| Metric | dRA (arcsec) | dDec (arcsec) | dX (pixels) | dY (pixels) |
| :--- | :--- | :--- | :--- | :--- |
| **Median Shift** | +0.19 | +0.21 | +0.18 | -0.06 |
| **Standard Deviation** | ±0.32 | ±0.37 | ±0.32 | ±0.14 |