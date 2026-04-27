import csv
import re
import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats

def read_reference_catalog(csv_path):
    """
    Reads the AAVSO reference star CSV.
    Extracts RA (deg), Dec (deg), V mag, and B mag.
    """
    ref_stars = []
    
    if not os.path.exists(csv_path):
        print(f"Error: Reference catalog not found at {csv_path}")
        return ref_stars

    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            auid = row['AUID']
            ra_str = row['RA']
            dec_str = row['Dec']
            v_str = row['V']
            bv_str = row['B-V']
            
            try:
                # Extract decimal degrees inside brackets e.g. [144.38566589°]
                ra_match = re.search(r'\[(.*?)[°]\]', ra_str)
                dec_match = re.search(r'\[(.*?)[°]\]', dec_str)
                
                if not ra_match or not dec_match:
                    continue
                    
                ra_deg = float(ra_match.group(1))
                dec_deg = float(dec_match.group(1))
                
                # Extract magnitudes, ignoring uncertainties in parentheses
                v_mag = float(v_str.split()[0])
                bv_mag = float(bv_str.split()[0])
                
                # Calculate B magnitude: B = (B-V) + V
                b_mag = bv_mag + v_mag
                
                ref_stars.append({
                    'id': auid,
                    'ra_deg': ra_deg,
                    'dec_deg': dec_deg,
                    'V_mag': v_mag,
                    'B_mag': b_mag
                })
            except Exception as e:
                print(f"Failed to parse row {auid}: {e}")
                
    return ref_stars

def match_and_calibrate(results, ref_catalog_file, filter_name, tolerance_arcsec=2.0, 
                        default_zp=23.399, run_new_calibration=True, output_report=None):
    print("\n=================================================================")
    print("--- 4. Zero Point Calibration ---")
    print("=================================================================\n")
    
    # Calculate instrumental magnitudes
    for rs in results:
        net_flux = rs.get('net_flux', 0)
        if net_flux > 0:
            rs['mag_inst'] = -2.5 * np.log10(net_flux)
        else:
            rs['mag_inst'] = np.nan
            
    if not run_new_calibration:
        print(f"Skipping new calibration. Applying default Zero Point: {default_zp:.3f}")
        for rs in results:
            if 'mag_inst' in rs and not np.isnan(rs['mag_inst']):
                rs['mag_calibrated'] = rs['mag_inst'] + default_zp
                rs['mag_calibrated_err'] = rs.get('mag_inst_err', np.nan)
            else:
                rs['mag_calibrated'] = np.nan
                rs['mag_calibrated_err'] = np.nan
        return
        
    ref_stars = read_reference_catalog(ref_catalog_file)
    if not ref_stars:
        print(f"WARNING: No reference stars loaded. Applying default Zero Point: {default_zp:.3f}")
        for rs in results:
            if 'mag_inst' in rs and not np.isnan(rs['mag_inst']):
                rs['mag_calibrated'] = rs['mag_inst'] + default_zp
                rs['mag_calibrated_err'] = rs.get('mag_inst_err', np.nan)
            else:
                rs['mag_calibrated'] = np.nan
                rs['mag_calibrated_err'] = np.nan
        return
        
    mag_key = 'B_mag' if 'B' in filter_name.upper() else 'V_mag'
    print(f"Using {mag_key} from reference catalog for calibration.")
    
    ref_ra = [s['ra_deg'] for s in ref_stars]
    ref_dec = [s['dec_deg'] for s in ref_stars]
    ref_coords = SkyCoord(ra=ref_ra*u.deg, dec=ref_dec*u.deg)
    ref_mags = np.array([s[mag_key] for s in ref_stars])
    
    det_valid = []
    det_ra = []
    det_dec = []
    for rs in results:
        if 'ra_deg' in rs and 'dec_deg' in rs and rs['ra_deg'] != "" and not np.isnan(rs.get('mag_inst', np.nan)):
            det_valid.append(rs)
            det_ra.append(float(rs['ra_deg']))
            det_dec.append(float(rs['dec_deg']))
            
    if not det_valid:
        print("WARNING: No detected stars with valid coordinates/flux to match.")
        for rs in results:
            rs['mag_calibrated'] = np.nan
        return
        
    det_coords = SkyCoord(ra=det_ra*u.deg, dec=det_dec*u.deg)
    
    idx, d2d, d3d = det_coords.match_to_catalog_sky(ref_coords)
    
    match_mask = d2d.arcsec < tolerance_arcsec
    matched_det = np.array(det_valid)[match_mask]
    matched_ref_mags = ref_mags[idx[match_mask]]
    
    if len(matched_det) == 0:
        print(f"WARNING: No matches found within {tolerance_arcsec} arcsec. Applying default Zero Point: {default_zp:.3f}")
        for rs in results:
            if 'mag_inst' in rs and not np.isnan(rs['mag_inst']):
                rs['mag_calibrated'] = rs['mag_inst'] + default_zp
                rs['mag_calibrated_err'] = rs.get('mag_inst_err', np.nan)
            else:
                rs['mag_calibrated'] = np.nan
                rs['mag_calibrated_err'] = np.nan
        return
        
    print(f"Found {len(matched_det)} matches with reference catalog.")
    
    report_lines = []
    report_lines.append(f"# Zero Point Calibration Report")
    report_lines.append(f"**Filter**: {filter_name}")
    report_lines.append(f"**Matches Found**: {len(matched_det)}\n")
    report_lines.append(f"| Match ID | Ref Mag | Inst Mag | Zero Point |")
    report_lines.append(f"| :--- | :--- | :--- | :--- |")
    
    zps = []
    for i, det_rs in enumerate(matched_det):
        mag_inst = det_rs['mag_inst']
        mag_ref = matched_ref_mags[i]
        zp = mag_ref - mag_inst
        zps.append(zp)
        print(f"  Match: {det_rs['id']} -> ZP: {zp:.3f} (Ref: {mag_ref:.3f}, Inst: {mag_inst:.3f})")
        report_lines.append(f"| {det_rs['id']} | {mag_ref:.3f} | {mag_inst:.3f} | {zp:.3f} |")
        
    zps = np.array(zps)
    mean_zp, median_zp, std_zp = sigma_clipped_stats(zps, sigma=3.0, maxiters=5)
    
    print(f"\nCalculated Zero Point: {median_zp:.3f} ± {std_zp:.3f} (Median)")
    
    report_lines.append(f"\n## Results")
    report_lines.append(f"- **Calculated Median Zero Point**: {median_zp:.3f}")
    report_lines.append(f"- **Standard Deviation**: ± {std_zp:.3f}\n")
    
    if output_report:
        with open(output_report, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    
    for rs in results:
        if 'mag_inst' in rs and not np.isnan(rs['mag_inst']):
            rs['mag_calibrated'] = rs['mag_inst'] + median_zp
            mag_inst_err = rs.get('mag_inst_err', np.nan)
            if not np.isnan(mag_inst_err):
                rs['mag_calibrated_err'] = np.sqrt(mag_inst_err**2 + std_zp**2)
            else:
                rs['mag_calibrated_err'] = np.nan
        else:
            rs['mag_calibrated'] = np.nan
            rs['mag_calibrated_err'] = np.nan

if __name__ == '__main__':
    # Test the parser
    test_file = r'c:\Astro\StarID\photometry_refstars\reference_stars.csv'
    print(f"Testing parsing of: {test_file}")
    stars = read_reference_catalog(test_file)
    print(f"Successfully loaded {len(stars)} reference stars:")
    for s in stars:
        print(f"ID: {s['id']}, RA: {s['ra_deg']:.4f}, Dec: {s['dec_deg']:.4f}, V: {s['V_mag']:.3f}, B: {s['B_mag']:.3f}")
