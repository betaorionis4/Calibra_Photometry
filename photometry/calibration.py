import csv
import re
import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astroquery.vizier import Vizier
from astropy.table import Table

def fetch_online_catalog(ra_deg, dec_deg, radius_arcmin=15, catalog_name="ATLAS", verbose=True):
    """
    Queries VizieR for photometric reference stars.
    Supports ATLAS (ATLAS-RefCat2) and APASS.
    """
    if verbose:
        print(f"Fetching {catalog_name} catalog from VizieR for RA={ra_deg:.4f}, Dec={dec_deg:.4f}...")
    
    # Configure Vizier
    # ATLAS-RefCat2: J/ApJ/867/105
    # APASS DR9: II/336
    # Map user-friendly labels to official VizieR identifiers
    viz_map = {
        "ATLAS": "J/ApJ/867/105",
        "ATLAS REFCAT2": "J/ApJ/867/105",
        "APASS": "II/336",
        "APASS DR9": "II/336",
        "GAIA_DR3": "I/355"
    }
    
    cat_upper = catalog_name.upper()
    if cat_upper in viz_map:
        catalog_id = viz_map[cat_upper]
    else:
        # Flexible fallback
        if "ATLAS" in cat_upper: catalog_id = viz_map["ATLAS"]
        elif "GAIA" in cat_upper: catalog_id = viz_map["GAIA_DR3"]
        elif "APASS" in cat_upper: catalog_id = viz_map["APASS"]
        else:
            print(f"Error: Unknown catalog {catalog_name}. Defaulting to ATLAS.")
            catalog_id = viz_map["ATLAS"]
    
    # Internal normalization for logic branches
    if "ATLAS" in cat_upper: catalog_name = "ATLAS"
    elif "GAIA" in cat_upper: catalog_name = "GAIA_DR3"
    elif "APASS" in cat_upper: catalog_name = "APASS"

    # Increase timeout and configure Vizier
    from astropy.utils.data import Conf
    Conf.remote_timeout.set(60) # 60 seconds timeout
    
    v = Vizier(columns=['*'], row_limit=-1)
    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame='icrs')
    
    try:
        if verbose:
            print(f"Querying VizieR {catalog_id}...")
        result = v.query_region(coord, radius=radius_arcmin * u.arcmin, catalog=catalog_id)
    except Exception as e:
        print(f"Error querying VizieR: {e}")
        return []

    if not result or len(result) == 0:
        print(f"VizieR returned no tables for {catalog_name}. Trying fallback ID...")
        # Fallback for ATLAS if specific ID fails
        if catalog_name == "ATLAS":
            try:
                result = v.query_region(coord, radius=radius_arcmin * u.arcmin, catalog="atlas")
                if not result:
                    return []
            except:
                return []
        else:
            return []

    table = result[0]
    print(f"VizieR returned {len(table)} raw rows from {catalog_name}.")
    ref_stars = []
    
    # Check available columns once
    cols = table.colnames
    
    for row in table:
        try:
            if catalog_name == "ATLAS":
                ra = float(row['RA_ICRS'])
                dec = float(row['DE_ICRS'])
                g = float(row['gmag']) if 'gmag' in cols else np.nan
                r = float(row['rmag']) if 'rmag' in cols else np.nan
                
                if not np.isnan(g) and not np.isnan(r):
                    color = g - r
                    # Jester et al. (2005) transformations (Stars, g-r < 2.1)
                    # Often used in software like ASTAP/MuniWin as the default for ATLAS
                    #v_mag = g - 0.011 - 0.487 * color
                    #b_mag = g + 0.202 + 0.473 * color
                    
                    # Alternative: Kostov et al. (2017) - Specific for Pan-STARRS1
                    v_mag = g - 0.020 - 0.498 * color - 0.008 * (color**2)
                    b_mag = g + 0.199 + 0.540 * color + 0.016 * (color**2)
                else:
                    v_mag = g if not np.isnan(g) else np.nan
                    b_mag = r if not np.isnan(r) else np.nan
                
                if 'Vmag' in cols and not np.isnan(float(row['Vmag'])): v_mag = float(row['Vmag'])
                if 'Bmag' in cols and not np.isnan(float(row['Bmag'])): b_mag = float(row['Bmag'])

            elif catalog_name == "GAIA_DR3":
                ra = float(row['RA_ICRS'])
                dec = float(row['DE_ICRS'])
                
                # Handle masked elements from VizieR to avoid UserWarnings
                def get_val(col):
                    val = row[col]
                    if np.ma.is_masked(val): return np.nan
                    try: return float(val)
                    except: return np.nan

                g = get_val('Gmag')
                bp = get_val('BPmag')
                rp = get_val('RPmag')
                
                if not np.isnan(g) and not np.isnan(bp) and not np.isnan(rp):
                    c = bp - rp
                    # Riello et al. (2021) Table C.2 (Wide range: -0.5 < Gbp-Grp < 5.0)
                    # Johnson V:
                    v_mag = g - 0.02704 + 0.01424 * c - 0.1604 * (c**2) + 0.02334 * (c**3)
                    
                    # Johnson B:
                    b_mag = g + 0.2162 + 0.7303 * c - 0.1518 * (c**2) + 0.02447 * (c**3)
                else:
                    v_mag = np.nan
                    b_mag = np.nan

            else: # Default/APASS
                ra = float(row['RAJ2000'])
                dec = float(row['DEJ2000'])
                v_mag = float(row['Vmag']) if 'Vmag' in cols else np.nan
                b_mag = float(row['Bmag']) if 'Bmag' in cols else np.nan

            if not np.isnan(v_mag) and v_mag < 18.0:
                ref_stars.append({
                    'id': f"online_{len(ref_stars)}",
                    'ra_deg': ra,
                    'dec_deg': dec,
                    'V_mag': v_mag,
                    'B_mag': b_mag,
                    'raw_g': g if catalog_name == "ATLAS" else np.nan,
                    'raw_r': r if catalog_name == "ATLAS" else np.nan,
                    'raw_G': g if catalog_name == "GAIA_DR3" else np.nan,
                    'raw_BP': bp if catalog_name == "GAIA_DR3" else np.nan,
                    'raw_RP': rp if catalog_name == "GAIA_DR3" else np.nan
                })
        except:
            continue

    if verbose:
        print(f"Successfully retrieved {len(ref_stars)} stars from {catalog_name}.")
    return ref_stars

def get_cached_catalog(ra, dec, radius, catalog_name, cache_dir="photometry_refstars/cache", verbose=True):
    """
    Checks for a cached version of the catalog query.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    cache_file = os.path.join(cache_dir, f"{catalog_name}_{ra:.3f}_{dec:.3f}_{radius}.csv")
    
    if os.path.exists(cache_file):
        if verbose:
            print(f"Loading cached catalog from {cache_file}")
        ref_stars = []
        with open(cache_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ref_stars.append({
                    'id': row['id'],
                    'ra_deg': float(row['ra_deg']),
                    'dec_deg': float(row['dec_deg']),
                    'V_mag': float(row['V_mag']),
                    'B_mag': float(row['B_mag'])
                })
        return ref_stars
    return None

def save_to_cache(ref_stars, ra, dec, radius, catalog_name, cache_dir="photometry_refstars/cache", verbose=True):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f"{catalog_name}_{ra:.3f}_{dec:.3f}_{radius}.csv")
    fieldnames = ['id', 'ra_deg', 'dec_deg', 'V_mag', 'B_mag', 'raw_g', 'raw_r', 'raw_G', 'raw_BP', 'raw_RP']
    with open(cache_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in ref_stars:
            # Ensure all keys exist in the dictionary for the writer
            row = {k: s.get(k, np.nan) for k in fieldnames}
            writer.writerow(row)
    if verbose:
        print(f"Saved catalog to cache: {cache_file}")

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
            try:
                auid = row.get('AUID', 'Unknown')
                ra_str = row.get('RA', '')
                dec_str = row.get('Dec', '')
                v_str = row.get('V', '')
                bv_str = row.get('B-V', '0.0')
                
                # Extract decimal degrees inside brackets e.g. [144.38566589°]
                ra_match = re.search(r'\[(.*?)[°]\]', ra_str)
                dec_match = re.search(r'\[(.*?)[°]\]', dec_str)
                
                if ra_match and dec_match:
                    ra_deg = float(ra_match.group(1))
                    dec_deg = float(dec_match.group(1))
                else:
                    # Try direct float conversion
                    ra_deg = float(ra_str)
                    dec_deg = float(dec_str)
                
                # Extract magnitudes, ignoring uncertainties in parentheses
                v_mag = float(v_str.split()[0]) if v_str else np.nan
                bv_mag = float(bv_str.split()[0]) if bv_str else 0.0
                
                # Calculate B magnitude: B = (B-V) + V
                b_mag = bv_mag + v_mag
                
                ref_stars.append({
                    'id': auid,
                    'ra_deg': ra_deg,
                    'dec_deg': dec_deg,
                    'V_mag': v_mag,
                    'B_mag': b_mag
                })
            except Exception:
                continue
                
    return ref_stars

def get_ref_stars(ref_catalog_file, center_ra=None, center_dec=None, radius_arcmin=15, verbose=True):
    """
    Helper function to load stars from local CSV or Online Catalog.
    """
    ref_stars = []
    cat_upper = ref_catalog_file.upper()
    is_online = any(k in cat_upper for k in ["ATLAS", "APASS", "GAIA"])

    if is_online and center_ra is not None and center_dec is not None:
        cat_name = cat_upper # Normalized inside fetch_online_catalog
        ref_stars = get_cached_catalog(center_ra, center_dec, radius_arcmin, cat_name, verbose=verbose)
        if not ref_stars:
            ref_stars = fetch_online_catalog(center_ra, center_dec, radius_arcmin, cat_name, verbose=verbose)
            if ref_stars:
                save_to_cache(ref_stars, center_ra, center_dec, radius_arcmin, cat_name, verbose=verbose)
    else:
        ref_stars = read_reference_catalog(ref_catalog_file)
    return ref_stars

def match_and_calibrate(results, ref_catalog_file, filter_name, tolerance_arcsec=2.0, 
                        default_zp=23.399, run_new_calibration=True, output_report=None,
                        center_ra=None, center_dec=None, snr_threshold=10.0,
                        print_to_console=True):
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
        
    ref_stars = get_ref_stars(ref_catalog_file, center_ra, center_dec, verbose=print_to_console)

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
    
    # Identify Source
    source_name = ref_catalog_file
    if ref_catalog_file.upper() in ["ATLAS", "APASS", "GAIA_DR3"]:
        source_name = f"Online VizieR ({ref_catalog_file.upper()})"
    else:
        source_name = f"Local File ({os.path.basename(ref_catalog_file)})"
    
    print(f"Reference Catalog: {source_name}")
    print(f"Using {mag_key} from reference catalog for calibration.")
    print(f"SNR Threshold for calibration stars: {snr_threshold}")
    
    ref_ra = [s['ra_deg'] for s in ref_stars]
    ref_dec = [s['dec_deg'] for s in ref_stars]
    ref_coords = SkyCoord(ra=ref_ra*u.deg, dec=ref_dec*u.deg)
    ref_mags = np.array([s[mag_key] for s in ref_stars])
    
    det_valid = []
    det_ra = []
    det_dec = []
    for rs in results:
        # NEW: Filter by SNR threshold AND saturation for calibration
        if 'ra_deg' in rs and 'dec_deg' in rs and rs['ra_deg'] != "" and not np.isnan(rs.get('mag_inst', np.nan)):
            if rs.get('snr', 0) >= snr_threshold and not rs.get('saturated', False):
                det_valid.append(rs)
                det_ra.append(float(rs['ra_deg']))
                det_dec.append(float(rs['dec_deg']))
            
    if not det_valid:
        print(f"WARNING: No detected stars with SNR >= {snr_threshold} to match.")
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
    report_lines.append(f"- **Source Catalogue**: {source_name}")
    report_lines.append(f"- **Calibration Filter**: {filter_name}")
    report_lines.append(f"- **Matches Found**: {len(matched_det)}\n")
    report_lines.append(f"| Match ID | Catalog RA/Dec (HMS/DMS) | V mag | B mag | B-V | Inst Mag | Zero Point |")
    report_lines.append(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    zps = []
    matched_ref_stars = [ref_stars[i] for i in idx[match_mask]]
    
    for i, det_rs in enumerate(matched_det):
        mag_inst = det_rs['mag_inst']
        ref_star = matched_ref_stars[i]
        mag_ref = ref_star[mag_key]
        zp = mag_ref - mag_inst
        zps.append(zp)
        
        # Format Coordinates
        c = SkyCoord(ra=ref_star['ra_deg']*u.deg, dec=ref_star['dec_deg']*u.deg)
        coord_str = c.to_string('hmsdms', sep=':', precision=2)
        
        v_mag = ref_star.get('V_mag', np.nan)
        b_mag = ref_star.get('B_mag', np.nan)
        bv = b_mag - v_mag if not np.isnan(b_mag) and not np.isnan(v_mag) else np.nan
        
        # Add SNR info to the printout
        snr = det_rs.get('snr', 0)
        if print_to_console:
            print(f"  Match: {det_rs['id']} (SNR: {snr:.1f}) -> ZP: {zp:.3f} (Ref {mag_key}: {mag_ref:.3f}, Inst: {mag_inst:.3f})")
        
        report_lines.append(f"| {det_rs['id']} | {coord_str} | {v_mag:.3f} | {b_mag:.3f} | {bv:+.3f} | {mag_inst:.3f} | **{zp:.3f}** |")
        
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
    for s in stars[:10]:
        print(f"ID: {s['id']}, RA: {s['ra_deg']:.4f}, Dec: {s['dec_deg']:.4f}, V: {s['V_mag']:.3f}, B: {s['B_mag']:.3f}")
