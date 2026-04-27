import os
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photometry.calibration import read_reference_catalog

def generate_shift_report(results, ref_catalog_file, header, tolerance_arcsec, output_md):
    """
    Cross-matches detected stars with the reference catalog and generates a 
    markdown report detailing the systematic positional shifts (RA, Dec, X, Y).
    """
    ref_stars = read_reference_catalog(ref_catalog_file)
    if not ref_stars:
        return
        
    try:
        wcs = WCS(header)
    except Exception as e:
        print(f"Warning: Could not create WCS for shift analysis: {e}")
        return

    # Calculate ideal pixel coordinates for reference stars
    for rs in ref_stars:
        x, y = wcs.all_world2pix(rs['ra_deg'], rs['dec_deg'], 1)
        rs['pixel_x'] = float(x)
        rs['pixel_y'] = float(y)

    # Filter detected stars
    det_valid = []
    for rs in results:
        if 'ra_deg' in rs and rs['ra_deg'] != "" and 'refined_x' in rs:
            det_valid.append({
                'id': rs['id'],
                'x': float(rs['refined_x']),
                'y': float(rs['refined_y']),
                'ra': float(rs['ra_deg']),
                'dec': float(rs['dec_deg'])
            })
            
    if not det_valid:
        return

    # Cross-match
    ref_coords = SkyCoord(ra=[s['ra_deg'] for s in ref_stars]*u.deg, dec=[s['dec_deg'] for s in ref_stars]*u.deg)
    det_coords = SkyCoord(ra=[s['ra'] for s in det_valid]*u.deg, dec=[s['dec'] for s in det_valid]*u.deg)

    idx, d2d, d3d = ref_coords.match_to_catalog_sky(det_coords)

    dx_list = []
    dy_list = []
    dra_list = []
    ddec_list = []
    
    report_lines = []
    report_lines.append("# Positional Shift Analysis\n")
    report_lines.append("Here is a breakdown of how the positions of the matched stars deviate from their expected reference catalog coordinates. The shifts are calculated as `Detected Position - Reference Position`.\n")
    report_lines.append("## Individual Star Deviations\n")
    report_lines.append("| Reference ID | Detected ID | dRA (arcsec) | dDec (arcsec) | dX (pixels) | dY (pixels) |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for i, ref in enumerate(ref_stars):
        if d2d[i].arcsec <= tolerance_arcsec:
            det = det_valid[idx[i]]
            
            c_ref = SkyCoord(ra=ref['ra_deg']*u.deg, dec=ref['dec_deg']*u.deg)
            c_det = SkyCoord(ra=det['ra']*u.deg, dec=det['dec']*u.deg)
            
            dra, ddec = c_ref.spherical_offsets_to(c_det)
            dra_arcsec = dra.arcsec
            ddec_arcsec = ddec.arcsec
            
            dx = det['x'] - ref['pixel_x']
            dy = det['y'] - ref['pixel_y']
            
            dx_list.append(dx)
            dy_list.append(dy)
            dra_list.append(dra_arcsec)
            ddec_list.append(ddec_arcsec)
            
            report_lines.append(f"| **{ref['id']}** | {det['id']} | {dra_arcsec:.2f} | {ddec_arcsec:.2f} | {dx:.2f} | {dy:.2f} |")

    if not dx_list:
        return # No matches

    med_dra, std_dra = np.median(dra_list), np.std(dra_list)
    med_ddec, std_ddec = np.median(ddec_list), np.std(ddec_list)
    med_dx, std_dx = np.median(dx_list), np.std(dx_list)
    med_dy, std_dy = np.median(dy_list), np.std(dy_list)

    report_lines.append("\n## Overall Statistics\n")
    report_lines.append("> [!NOTE]")
    report_lines.append("> **Conclusion on Systematic Shifts**")
    report_lines.append(f"> The detected stars are, on average, offset by **{med_dx:+.2f} pixels** (X) and **{med_dy:+.2f} pixels** (Y).")
    report_lines.append(f"> In celestial coordinates, this is a shift of **{med_dra:+.2f} arcsec** (RA) and **{med_ddec:+.2f} arcsec** (Dec).\n")
    
    report_lines.append("| Metric | dRA (arcsec) | dDec (arcsec) | dX (pixels) | dY (pixels) |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    report_lines.append(f"| **Median Shift** | {med_dra:+.2f} | {med_ddec:+.2f} | {med_dx:+.2f} | {med_dy:+.2f} |")
    report_lines.append(f"| **Standard Deviation** | ±{std_dra:.2f} | ±{std_ddec:.2f} | ±{std_dx:.2f} | ±{std_dy:.2f} |")

    with open(output_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
