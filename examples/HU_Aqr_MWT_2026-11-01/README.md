# Example frames: HU Aqr, 2026-11-01

Three raw (uncalibrated) luminance frames of the polar (AM Her-type cataclysmic variable) **HU Aqr**, taken on 2026-11-01 around 21:37 UT with the 0.42 m **Margareta Westlund Telescope** of the Swedish Amateur Astronomers' Association (SAAF) in Spain.

| | |
|---|---|
| Frames | 3 files `frames/RAW-HU Aqr-1402-20261101@HHMMSS-030S-B4-Lum.fts` (21:36:33, 21:37:16, 21:37:58 UT; 7.7 MB each) |
| Exposure | 30 s each, filter L (luminance, no photometric filter) |
| Camera | ZWO ASI (CMOS), binning 4×4, 2394 × 1597 pixels, gain 100, offset 50, EGAIN 0.247 e⁻/ADU |
| Optics | aperture 420 mm, focal length 2990 mm |
| Header | MaxIm DL; site latitude/longitude in sexagesimal text (SITELAT/SITELONG), JD and JD-HELIO at start of exposure |
| WCS | none, so the frames have to be plate solved first (ASTAP, *Plate Solving* tab; use the object name "HU Aqr" as position guess) |

No bias, dark or flat frames are included, so the frames can only be used uncalibrated.

## What you can try

1. Add the three frames in the File Manager and plate solve them.
2. Open one frame in the FITS Viewer: catalogue stars, known variables (VSX/GCVS), radial profiles.
3. Light Curves: HU Aqr as target, comparison stars from the catalogue. L frames use V comparison magnitudes (AAVSO filter code CV).

Example results made from these frames will be added in `results/`.

The frames are provided for testing Calibra. The software serial number and owner name written by the camera software were removed from the FITS headers; image data are unchanged.
