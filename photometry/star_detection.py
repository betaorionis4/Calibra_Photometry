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
                 filter_mode='all', xy_bounds=None, radec_bounds=None):
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
    daofind = DAOStarFinder(fwhm=3.5, threshold=detect_sigma * std_bg,
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
        try:
            coord1 = SkyCoord(ra=radec_bounds['ra_min'], dec=radec_bounds['dec_min'], frame='icrs')
            coord2 = SkyCoord(ra=radec_bounds['ra_max'], dec=radec_bounds['dec_max'], frame='icrs')
            ra_min_deg = min(coord1.ra.deg, coord2.ra.deg)
            ra_max_deg = max(coord1.ra.deg, coord2.ra.deg)
            dec_min_deg = min(coord1.dec.deg, coord2.dec.deg)
            dec_max_deg = max(coord1.dec.deg, coord2.dec.deg)
        except Exception as e:
            print(f"Error parsing RA/DEC bounds: {e}")
            filter_mode = 'all' # fallback

    results = []

    valid_count = 0
    for row in sources:
        fits_x = row['xcentroid'] + 1.0
        fits_y = row['ycentroid'] + 1.0
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
            if not (ra_min_deg <= ra_deg <= ra_max_deg and dec_min_deg <= dec_deg <= dec_max_deg):
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
