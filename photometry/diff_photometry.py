import os
import csv
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from photometry.calibration import get_ref_stars

def compute_zero_points(Tbv, Tb_bv, Tv_bv, kB, kV, B_ref, V_ref, b_ref, v_ref, XB_ref, XV_ref):
    # Extinction-corrected instrumental magnitudes
    b0 = b_ref - kB * XB_ref
    v0 = v_ref - kV * XV_ref

    # Colors
    BV_ref = B_ref - V_ref
    bv0_ref = b0 - v0

    # Zero point for color equation
    Z_BV = BV_ref - Tbv * bv0_ref

    # Zero point for B equation
    Z_B = B_ref - (b0 + Tb_bv * BV_ref)

    # Zero point for V equation
    Z_V = V_ref - (v0 + Tv_bv * BV_ref)

    return Z_BV, Z_B, Z_V

def compute_target_BV(b_t, v_t, XB_t, XV_t, Tbv, Tb_bv, Tv_bv, Z_BV, Z_B, Z_V, kB, kV):
    # Extinction-corrected instrumental magnitudes
    b0_t = b_t - kB * XB_t
    v0_t = v_t - kV * XV_t

    # Color
    bv0_t = b0_t - v0_t

    # Standard color
    BV_t = Tbv * bv0_t + Z_BV

    # Standard magnitudes
    B_t = b0_t + Tb_bv * BV_t + Z_B
    V_t = v0_t + Tv_bv * BV_t + Z_V

    return B_t, V_t, BV_t

def run_differential_photometry(csv_b, csv_v, ref_catalog, k_b, k_v, Tbv, Tb_bv, Tv_bv):
    """
    Reads two CSVs (B and V), matches the stars, selects a reference star,
    and applies differential photometry to all common stars.
    """
    # 1. Read CSVs
    def read_csv_data(filepath):
        data = []
        if not os.path.exists(filepath): return data
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    data_b = read_csv_data(csv_b)
    data_v = read_csv_data(csv_v)
    
    if not data_b or not data_v:
        return "Error: Could not read B or V CSV files."
        
    # 2. Match B and V stars
    ra_b = [float(r['ra_deg']) for r in data_b if r['ra_deg'] and r['mag_inst']]
    dec_b = [float(r['dec_deg']) for r in data_b if r['dec_deg'] and r['mag_inst']]
    valid_b = [r for r in data_b if r['ra_deg'] and r['dec_deg'] and r['mag_inst']]
    
    ra_v = [float(r['ra_deg']) for r in data_v if r['ra_deg'] and r['mag_inst']]
    dec_v = [float(r['dec_deg']) for r in data_v if r['dec_deg'] and r['mag_inst']]
    valid_v = [r for r in data_v if r['ra_deg'] and r['dec_deg'] and r['mag_inst']]
    
    if not valid_b or not valid_v:
        return "Error: Missing valid coordinates or instrumental magnitudes in input CSVs."
        
    coords_b = SkyCoord(ra=ra_b*u.deg, dec=dec_b*u.deg)
    coords_v = SkyCoord(ra=ra_v*u.deg, dec=dec_v*u.deg)
    
    idx, d2d, _ = coords_b.match_to_catalog_sky(coords_v)
    mask = d2d < 2.0 * u.arcsec
    
    matched_pairs = []
    for i, is_match in enumerate(mask):
        if is_match:
            matched_pairs.append({
                'b': valid_b[i],
                'v': valid_v[idx[i]]
            })
            
    if not matched_pairs:
        return "Error: No matching stars found between B and V results."
        
    # 3. Query Reference Catalog
    # Use the center of the field based on the first matched star
    center_ra = float(matched_pairs[0]['v']['ra_deg'])
    center_dec = float(matched_pairs[0]['v']['dec_deg'])
    
    ref_stars = get_ref_stars(ref_catalog, center_ra, center_dec, radius_arcmin=15, verbose=False)
    if not ref_stars:
        return f"Error: Could not retrieve reference stars from {ref_catalog}."
        
    coords_cat = SkyCoord(ra=[s['ra_deg'] for s in ref_stars]*u.deg, 
                          dec=[s['dec_deg'] for s in ref_stars]*u.deg)
                          
    coords_pairs = SkyCoord(ra=[float(p['v']['ra_deg']) for p in matched_pairs]*u.deg, 
                            dec=[float(p['v']['dec_deg']) for p in matched_pairs]*u.deg)
                            
    idx_cat, d2d_cat, _ = coords_pairs.match_to_catalog_sky(coords_cat)
    mask_cat = d2d_cat < 2.0 * u.arcsec
    
    # 4. Find the best reference star
    # Criteria: 0.4 <= (B-V) <= 0.8, not saturated, brightest
    best_ref_pair = None
    best_ref_cat = None
    min_v_inst = float('inf')
    
    for i, is_match in enumerate(mask_cat):
        if is_match:
            pair = matched_pairs[i]
            cat_star = ref_stars[idx_cat[i]]
            
            # Check for saturation (we consider it saturated if the FITS peak > 60k or if it's explicitly flagged,
            # wait, the CSV doesn't have a 'saturated' boolean, but it has 'peak_adu'. We can check if peak_adu > 55000)
            peak_b = float(pair['b'].get('peak_adu', 0))
            peak_v = float(pair['v'].get('peak_adu', 0))
            if peak_b > 55000 or peak_v > 55000:
                continue
                
            b_cat = cat_star.get('B_mag', np.nan)
            v_cat = cat_star.get('V_mag', np.nan)
            
            if np.isnan(b_cat) or np.isnan(v_cat):
                continue
                
            bv_cat = b_cat - v_cat
            if 0.4 <= bv_cat <= 0.8:
                v_inst = float(pair['v']['mag_inst'])
                if v_inst < min_v_inst:
                    min_v_inst = v_inst
                    best_ref_pair = pair
                    best_ref_cat = cat_star
                    
    if not best_ref_pair:
        return "Error: No suitable reference star found (0.4 <= B-V <= 0.8, unsaturated)."
        
    # 5. Extract Reference Star Data
    B_ref = best_ref_cat['B_mag']
    V_ref = best_ref_cat['V_mag']
    b_ref = float(best_ref_pair['b']['mag_inst'])
    v_ref = float(best_ref_pair['v']['mag_inst'])
    XB_ref = float(best_ref_pair['b'].get('airmass', 1.0))
    XV_ref = float(best_ref_pair['v'].get('airmass', 1.0))
    
    Z_BV, Z_B, Z_V = compute_zero_points(Tbv, Tb_bv, Tv_bv, k_b, k_v, B_ref, V_ref, b_ref, v_ref, XB_ref, XV_ref)
    
    # 6. Apply to all matched targets
    output_rows = []
    
    for pair in matched_pairs:
        b_inst = float(pair['b']['mag_inst'])
        v_inst = float(pair['v']['mag_inst'])
        XB = float(pair['b'].get('airmass', 1.0))
        XV = float(pair['v'].get('airmass', 1.0))
        
        B_t, V_t, BV_t = compute_target_BV(b_inst, v_inst, XB, XV, Tbv, Tb_bv, Tv_bv, Z_BV, Z_B, Z_V, k_b, k_v)
        
        output_rows.append({
            'id_v': pair['v']['id'],
            'id_b': pair['b']['id'],
            'ra_deg': pair['v']['ra_deg'],
            'dec_deg': pair['v']['dec_deg'],
            'B_mag': f"{B_t:.4f}",
            'V_mag': f"{V_t:.4f}",
            'B_V': f"{BV_t:.4f}",
            'v_inst': f"{v_inst:.4f}",
            'b_inst': f"{b_inst:.4f}",
            'airmass_v': f"{XV:.4f}",
            'airmass_b': f"{XB:.4f}"
        })
        
    # 7. Write output
    output_dir = os.path.dirname(csv_v)
    out_csv = os.path.join(output_dir, "differential_photometry_results.csv")
    out_md = os.path.join(output_dir, "differential_photometry_report.md")
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
        
    with open(out_md, 'w') as f:
        f.write("# Differential Photometry Report\n\n")
        f.write("## Reference Star\n")
        f.write(f"- RA/Dec: {best_ref_cat['ra_deg']:.5f}, {best_ref_cat['dec_deg']:.5f}\n")
        f.write(f"- Standard B: {B_ref:.4f}\n")
        f.write(f"- Standard V: {V_ref:.4f}\n")
        f.write(f"- Standard B-V: {B_ref - V_ref:.4f}\n")
        f.write(f"- Instrumental B: {b_ref:.4f}\n")
        f.write(f"- Instrumental V: {v_ref:.4f}\n")
        f.write(f"- Airmass B/V: {XB_ref:.3f} / {XV_ref:.3f}\n\n")
        
        f.write("## Calculated Zero Points\n")
        f.write(f"- $Z_{{BV}}$: {Z_BV:.4f}\n")
        f.write(f"- $Z_B$: {Z_B:.4f}\n")
        f.write(f"- $Z_V$: {Z_V:.4f}\n\n")
        
        f.write(f"Processed {len(output_rows)} stars successfully.\n")
        f.write(f"Results saved to: `{out_csv}`\n")
        
    return f"Success! Used 1 reference star to calibrate {len(output_rows)} stars. Saved to {out_csv}."
