import csv
import warnings
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning

def detect_stars(image_data, header, detect_sigma, 
                 sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0,
                 filter_mode='all', xy_bounds=None, radec_bounds=None, fwhm=3.5):
    print("=================================================================")
    print("--- 1. Automated Star Retrieval (DAOStarFinder) ---")
    print("=================================================================\n")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        wcs = WCS(header)
    has_wcs = wcs.has_celestial

    mean, median_bg, std_bg = sigma_clipped_stats(image_data, sigma=3.0, maxiters=5)
    print(f"Global Image Stats: Median Bkg = {median_bg:.1f} ADU | Noise Sigma = {std_bg:.1f} ADU")
    
    # Initialize DAOStarFinder with user-defined morphology limits
    # We use the older parameter names (sharplo, sharphi, etc.) for compatibility
    # but silence the DeprecationWarning that photutils 2.x+ generates.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=AstropyWarning, message=".*sharplo.*")
        warnings.filterwarnings("ignore", category=AstropyWarning, message=".*roundlo.*")
        try:
            # v4.16: detection kernel FWHM is configurable (was fixed at 3.5 px)
            daofind = DAOStarFinder(fwhm=fwhm, threshold=detect_sigma * std_bg,
                                    sharpness_range=(sharplo, sharphi), 
                                    roundness_range=(roundlo, roundhi))
        except TypeError:
            daofind = DAOStarFinder(fwhm=fwhm, threshold=detect_sigma * std_bg,
                                    sharplo=sharplo, sharphi=sharphi, 
                                    roundlo=roundlo, roundhi=roundhi)
    
    sources = daofind(image_data - median_bg)

    if sources is None:
        print("No targets found above the threshold!")
        return []

    sources.sort('flux')
    sources.reverse()
    print(f"Found {len(sources)} stars before filtering.\n")

    # Pre-parse RA/DEC bounds if needed
    ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg = None, None, None, None
    if filter_mode == 'radec' and radec_bounds and has_wcs:
        # v4.15: bounds from the GUI are plain numbers in degrees. Previously they
        # were passed to SkyCoord without units, which always raised an error, so
        # the RA/Dec region filter silently fell back to 'all'.
        # RA_min > RA_max is interpreted as a region crossing RA = 0/360 deg.
        def _to_deg(val, is_ra):
            try:
                return float(val)
            except (TypeError, ValueError):
                from astropy.coordinates import Angle  # sexagesimal text, e.g. '9:36:53'
                return Angle(str(val), unit=u.hourangle if is_ra else u.deg).deg
        try:
            ra_min_deg = _to_deg(radec_bounds['ra_min'], True) % 360.0
            ra_max_deg = _to_deg(radec_bounds['ra_max'], True)
            ra_max_deg = 360.0 if ra_max_deg >= 360.0 else ra_max_deg % 360.0
            d1 = _to_deg(radec_bounds['dec_min'], False)
            d2 = _to_deg(radec_bounds['dec_max'], False)
            dec_min_deg, dec_max_deg = min(d1, d2), max(d1, d2)
            print(f"RA/Dec region filter: RA {ra_min_deg:.4f}..{ra_max_deg:.4f} deg"
                  f"{' (wraps through 0)' if ra_min_deg > ra_max_deg else ''}, Dec {dec_min_deg:.4f}..{dec_max_deg:.4f} deg")
        except Exception as e:
            print(f"Error parsing RA/DEC bounds: {e}")
            filter_mode = 'all' # fallback

    results = []

    # Detect column names dynamically to handle different photutils versions
    x_col = 'x_centroid' if 'x_centroid' in sources.colnames else 'xcentroid'
    y_col = 'y_centroid' if 'y_centroid' in sources.colnames else 'ycentroid'
    
    valid_count = 0
    for row in sources:
        fits_x = row[x_col] + 1.0
        fits_y = row[y_col] + 1.0
        peak_val = row['peak']
        flux_val = row['flux']

        ra_deg, dec_deg, ra_hms, dec_dms = "", "", "", ""
        if has_wcs:
            ra_conv, dec_conv = wcs.all_pix2world(fits_x, fits_y, 1)
            ra_deg, dec_deg = float(ra_conv), float(dec_conv)
            coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
            ra_hms = coord.ra.to_string(unit=u.hourangle, sep='hms', precision=2, pad=True)
            dec_dms = coord.dec.to_string(unit=u.degree, sep='dms', precision=1, pad=True, alwayssign=True)

        # Apply filtering
        if filter_mode == 'xy' and xy_bounds:
            if not (xy_bounds.get('x_min', 0) <= fits_x <= xy_bounds.get('x_max', 1e9) and 
                    xy_bounds.get('y_min', 0) <= fits_y <= xy_bounds.get('y_max', 1e9)):
                continue
                
        if filter_mode == 'radec' and ra_min_deg is not None:
            if ra_min_deg <= ra_max_deg:
                in_ra = ra_min_deg <= ra_deg <= ra_max_deg
            else:  # wrap-around region
                in_ra = ra_deg >= ra_min_deg or ra_deg <= ra_max_deg
            if not (in_ra and dec_min_deg <= dec_deg <= dec_max_deg):
                continue

        valid_count += 1
        star_id = f"Auto_{valid_count:03d}"

        results.append({
            'id': star_id, 
            'x': fits_x, 
            'y': fits_y,
            'peak_adu': peak_val,
            'dao_flux': flux_val,
            'ra_deg': f"{ra_deg:.5f}" if ra_deg else "",
            'dec_deg': f"{dec_deg:.5f}" if dec_deg else "",
            'ra_hms': ra_hms,
            'dec_dms': dec_dms
        })

    print(f"Keeping {valid_count} stars after filtering.\n")

    return results
