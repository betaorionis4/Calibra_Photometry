"""Calibra program modules (photometry, calibration, catalogues, light curves,
period analysis, planner, NINA export, user folder). Start the program with
calibra.py in the folder above."""
from .version import __version__, VERSION_DATE  # noqa: F401

# v5.0.1: astropy's FITSFixedWarning only reports header keywords it repaired on
# reading a WCS (e.g. deprecated RADECSYS -> RADESYSa, MJD-OBS set from
# DATE-OBS); the WCS and the times are correct. Silenced once for the whole
# program instead of at each WCS() call (several places were not covered).
import warnings as _warnings
from astropy.wcs import FITSFixedWarning as _FITSFixedWarning
_warnings.simplefilter('ignore', _FITSFixedWarning)
