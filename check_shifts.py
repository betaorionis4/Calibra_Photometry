import csv
import re
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

fits_filename = 'fitsfiles/AE_UMa_Bmag_corr_00003.fits'
header = fits.getheader(fits_filename)
wcs = WCS(header)

# Load reference stars
ref_file = r'photometry_refstars\reference_stars.csv'
ref_stars = []
with open(ref_file, mode='r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ra_match = re.search(r'\[(.*?)[°]\]', row['RA'])
        dec_match = re.search(r'\[(.*?)[°]\]', row['Dec'])
        if ra_match and dec_match:
            r_ra = float(ra_match.group(1))
            r_dec = float(dec_match.group(1))
            x, y = wcs.all_world2pix(r_ra, r_dec, 1)
            ref_stars.append({
                'id': row['AUID'], 
                'ra': r_ra, 
                'dec': r_dec,
                'x': float(x),
                'y': float(y)
            })

# Load detected stars
det_stars = []
out_csv = r'photometry_output\targets_auto_AE_UMa_Bmag_corr_00003.csv'
with open(out_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['ra_deg'] and row['refined_x']:
            det_stars.append({
                'id': row['id'], 
                'x': float(row['refined_x']), 
                'y': float(row['refined_y']),
                'ra': float(row['ra_deg']),
                'dec': float(row['dec_deg'])
            })

# Cross-match
ref_coords = SkyCoord(ra=[s['ra'] for s in ref_stars]*u.deg, dec=[s['dec'] for s in ref_stars]*u.deg)
det_coords = SkyCoord(ra=[s['ra'] for s in det_stars]*u.deg, dec=[s['dec'] for s in det_stars]*u.deg)

idx, d2d, d3d = ref_coords.match_to_catalog_sky(det_coords)

print(f"{'Ref ID':<15} | {'Det ID':<10} | {'dRA (arcsec)':<12} | {'dDec (arcsec)':<13} | {'dX (px)':<8} | {'dY (px)':<8}")
print("-" * 75)

dx_list = []
dy_list = []
dra_list = []
ddec_list = []

for i, ref in enumerate(ref_stars):
    if d2d[i].arcsec < 8.0:
        det = det_stars[idx[i]]
        
        # Calculate shifts
        c_ref = SkyCoord(ra=ref['ra']*u.deg, dec=ref['dec']*u.deg)
        c_det = SkyCoord(ra=det['ra']*u.deg, dec=det['dec']*u.deg)
        
        dra, ddec = c_ref.spherical_offsets_to(c_det)
        dra_arcsec = dra.arcsec
        ddec_arcsec = ddec.arcsec
        
        # shift = Detected - Reference
        dx = det['x'] - ref['x']
        dy = det['y'] - ref['y']
        
        dx_list.append(dx)
        dy_list.append(dy)
        dra_list.append(dra_arcsec)
        ddec_list.append(ddec_arcsec)
        
        print(f"{ref['id']:<15} | {det['id']:<10} | {dra_arcsec:>12.2f} | {ddec_arcsec:>13.2f} | {dx:>8.2f} | {dy:>8.2f}")

print("-" * 75)
print(f"MEDIAN SHIFTS   | {'':<10} | {np.median(dra_list):>12.2f} | {np.median(ddec_list):>13.2f} | {np.median(dx_list):>8.2f} | {np.median(dy_list):>8.2f}")
print(f"STD DEV         | {'':<10} | {np.std(dra_list):>12.2f} | {np.std(ddec_list):>13.2f} | {np.std(dx_list):>8.2f} | {np.std(dy_list):>8.2f}")
