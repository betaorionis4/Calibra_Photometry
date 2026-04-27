import csv
import re
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

fits_filename = 'fitsfiles/AE_UMa_Bmag_corr_00003.fits'
header = fits.getheader(fits_filename)
wcs = WCS(header)

ref_file = r'photometry_refstars\reference_stars.csv'
ref_stars = []
with open(ref_file, mode='r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ra_match = re.search(r'\[(.*?)[°]\]', row['RA'])
        dec_match = re.search(r'\[(.*?)[°]\]', row['Dec'])
        if ra_match and dec_match:
            ref_stars.append({'id': row['AUID'], 'ra': float(ra_match.group(1)), 'dec': float(dec_match.group(1))})

# Calculate ideal pixel coordinates for reference stars using all_world2pix
for rs in ref_stars:
    # Use 1 for 1-based pixel coordinates to match fits_x, fits_y
    x, y = wcs.all_world2pix(rs['ra'], rs['dec'], 1)
    rs['pixel_x'] = float(x)
    rs['pixel_y'] = float(y)

# Load detected stars
det_stars = []
out_csv = r'photometry_output\targets_auto_AE_UMa_Bmag_corr_00003.csv'
with open(out_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['raw_x']:
            det_stars.append({
                'id': row['id'], 
                'x': float(row['raw_x']), 
                'y': float(row['raw_y']),
                'ra': float(row['ra_deg']) if row['ra_deg'] else 0,
                'dec': float(row['dec_deg']) if row['dec_deg'] else 0
            })

# Cross-match by pixel distance
print(f"{'Ref ID':<15} | {'Det ID':<10} | {'Pixel Dist':<10}")
print("-" * 40)

for rs in ref_stars:
    distances = []
    for ds in det_stars:
        dist = np.sqrt((rs['pixel_x'] - ds['x'])**2 + (rs['pixel_y'] - ds['y'])**2)
        distances.append((dist, ds))
    
    distances.sort(key=lambda x: x[0])
    best_match = distances[0]
    
    print(f"{rs['id']:<15} | {best_match[1]['id']:<10} | {best_match[0]:<10.1f}")
