import numpy as np
from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import mad_std
from photutils.background import Background2D, MedianBackground

fits_filename = 'fitsfiles/AE_UMa_Bmag_corr_00003.fits'
image_data = fits.getdata(fits_filename)

bkg = Background2D(image_data, (50, 50), filter_size=(3, 3), bkg_estimator=MedianBackground())
median_bg = bkg.background
std_bg = bkg.background_rms

daofind_loose = DAOStarFinder(fwhm=3.5, threshold=5.0 * std_bg, sharplo=-2.0, sharphi=2.0, roundlo=-2.0, roundhi=2.0)
sources_loose = daofind_loose(image_data - median_bg)

daofind_strict = DAOStarFinder(fwhm=3.5, threshold=5.0 * std_bg)
sources_strict = daofind_strict(image_data - median_bg)

print(f"Strict DAO (default): {len(sources_strict)} sources")
print(f"Loose DAO (no sharp/round limits): {len(sources_loose)} sources")
