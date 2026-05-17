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
        "GAIA_DR3": "I/355",
        "LANDOLT": ["II/183A", "J/AJ/137/4186", "J/AJ/133/2502", "J/AJ/146/131"]
    }
    
    cat_upper = catalog_name.upper()
    if cat_upper in viz_map:
        catalog_id = viz_map[cat_upper]
    else:
        # Flexible fallback
        if "ATLAS" in cat_upper: catalog_id = viz_map["ATLAS"]
        elif "GAIA" in cat_upper: catalog_id = viz_map["GAIA_DR3"]
        elif "APASS" in cat_upper: catalog_id = viz_map["APASS"]
        elif "LANDOLT" in cat_upper: catalog_id = viz_map["LANDOLT"]
        else:
            print(f"Error: Unknown catalog {catalog_name}. Defaulting to ATLAS.")
            catalog_id = viz_map["ATLAS"]
    
    # Internal normalization for logic branches
    if "ATLAS" in cat_upper: catalog_name = "ATLAS"
    elif "GAIA" in cat_upper: catalog_name = "GAIA_DR3"
    elif "APASS" in cat_upper: catalog_name = "APASS"
    elif "LANDOLT" in cat_upper: catalog_name = "LANDOLT"

    # Increase timeout and configure Vizier
    from astropy.utils.data import Conf
    Conf.remote_timeout.set(60) # 60 seconds timeout
    
    v = Vizier(columns=['*', '_RAJ2000', '_DEJ2000'], row_limit=-1)
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
    print(f"VizieR returned {len(result)} tables for {catalog_name}.")
    ref_stars = []
    
    for table in result:
        cols = table.colnames
        for row in table:
            try:
                # Handle masked elements from VizieR to avoid UserWarnings
                def get_val(col):
                    if col not in cols: return np.nan
                    val = row[col]
                    if np.ma.is_masked(val): return np.nan
                    try: return float(val)
                    except: return np.nan

                if catalog_name == "ATLAS":
                    ra = float(row['RA_ICRS'])
                    dec = float(row['DE_ICRS'])
                    g = float(row['gmag']) if 'gmag' in cols else np.nan
                    r = float(row['rmag']) if 'rmag' in cols else np.nan
                    
                    if not np.isnan(g) and not np.isnan(r):
                        color = g - r
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
                
                    g = get_val('Gmag')
                    bp = get_val('BPmag')
                    rp = get_val('RPmag')
                
                    if not np.isnan(g) and not np.isnan(bp) and not np.isnan(rp):
                        c = bp - rp
                        v_mag = g + 0.02704 - 0.01424 * c + 0.2156 * (c**2) - 0.01426 * (c**3)
                        b_mag = g - 0.01448 + 0.6874 * c + 0.3604 * (c**2) - 0.06718 * (c**3) + 0.006061 * (c**4)
                    else:
                        v_mag = np.nan
                        b_mag = np.nan

                elif catalog_name == "LANDOLT":
                    ra = float(row['_RAJ2000']) if '_RAJ2000' in cols else float(row['RAJ2000'])
                    dec = float(row['_DEJ2000']) if '_DEJ2000' in cols else float(row['DEJ2000'])
                
                    if 'Vmag' in cols: v_mag = get_val('Vmag')
                    elif '<Vmag>' in cols: v_mag = get_val('<Vmag>')
                    else: v_mag = np.nan
                    
                    if 'B-V' in cols: b_v = get_val('B-V')
                    elif '<B-V>' in cols: b_v = get_val('<B-V>')
                    else: b_v = np.nan
                    
                    b_mag = v_mag + b_v if not np.isnan(v_mag) and not np.isnan(b_v) else np.nan

                else: # Default/APASS
                    ra = float(row['RAJ2000'])
                    dec = float(row['DEJ2000'])
                    v_mag = get_val('Vmag')
                    b_mag = get_val('Bmag')

                if not np.isnan(v_mag) and v_mag < 18.0:
                    # Capture a natural identifier for the star
                    cat_id = ""
                    if catalog_name == "GAIA_DR3" and 'Source' in cols:
                        cat_id = f"Gaia {row['Source']}"
                    elif catalog_name == "ATLAS" and 'objID' in cols:
                        cat_id = f"ATLAS {row['objID']}"
                    elif catalog_name == "LANDOLT" and 'Star' in cols:
                        cat_id = str(row['Star'])
                    elif catalog_name == "APASS" and 'recno' in cols:
                        cat_id = f"APASS {row['recno']}"
                    
                    ref_stars.append({
                        'id': f"online_{len(ref_stars)}",
                        'cat_id': cat_id,
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
        
    cache_file = os.path.join(cache_dir, f"{catalog_name}_{ra:.4f}_{dec:.4f}_{radius}.csv")
    
    if os.path.exists(cache_file):
        if verbose:
            print(f"Loading cached catalog from {cache_file}")
        ref_stars = []
        with open(cache_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ref_stars.append({
                    'id': row['id'],
                    'cat_id': row.get('cat_id', ''),
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
    fieldnames = ['id', 'cat_id', 'ra_deg', 'dec_deg', 'V_mag', 'B_mag', 'raw_g', 'raw_r', 'raw_G', 'raw_BP', 'raw_RP']
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
    is_online = any(k in cat_upper for k in ["ATLAS", "APASS", "GAIA", "LANDOLT"])

    if is_online and center_ra is not None and center_dec is not None:
        cat_name = cat_upper # Normalized inside fetch_online_catalog
        ref_stars = get_cached_catalog(center_ra, center_dec, radius_arcmin, cat_name, verbose=verbose)
        if not ref_stars:
            ref_stars = fetch_online_catalog(center_ra, center_dec, radius_arcmin, cat_name, verbose=verbose)
            if ref_stars:
                save_to_cache(ref_stars, center_ra, center_dec, radius_arcmin, cat_name, verbose=verbose)
        
        # Mark variables in the returned set
        mark_variable_stars(ref_stars, center_ra, center_dec, radius_arcmin, verbose=verbose)
    else:
        ref_stars = read_reference_catalog(ref_catalog_file)
    return ref_stars

def fetch_vsx_catalog(ra_deg, dec_deg, radius_arcmin=15, verbose=True):
    if verbose:
        print(f"Fetching VSX catalog from VizieR for RA={ra_deg:.4f}, Dec={dec_deg:.4f}...")
    
    from astropy.utils.data import Conf
    Conf.remote_timeout.set(60)
    
    v = Vizier(catalog="B/vsx/vsx", columns=['OID', 'Name', '_RAJ2000', '_DEJ2000', 'Type'], row_limit=-1)
    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame='icrs')
    
    try:
        result = v.query_region(coord, radius=radius_arcmin * u.arcmin)
    except Exception as e:
        print(f"Error querying VSX: {e}")
        return []
        
    if not result or len(result) == 0:
        if verbose:
            print("No VSX variables found in this region.")
        return []
        
    table = result[0]
    vsx_stars = []
    for row in table:
        try:
            vsx_stars.append({
                'id': str(row['Name']),
                'ra_deg': float(row['_RAJ2000']),
                'dec_deg': float(row['_DEJ2000']),
                'Type': str(row['Type']) if 'Type' in row.colnames and not np.ma.is_masked(row['Type']) else ''
            })
        except:
            continue
            
    if verbose:
        print(f"Successfully retrieved {len(vsx_stars)} variable stars from VSX.")
    return vsx_stars

def get_vsx_stars(ra_deg, dec_deg, radius_arcmin=15, cache_dir="photometry_refstars/cache", verbose=True):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    cache_file = os.path.join(cache_dir, f"VSX_{ra_deg:.4f}_{dec_deg:.4f}_{radius_arcmin}.csv")
    
    vsx_stars = []
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # If Type column is missing, invalidate cache to get CST data
                if 'Type' not in reader.fieldnames:
                    if verbose: print(f"VSX Cache is old (missing Type). Re-fetching...")
                else:
                    if verbose: print(f"Loading cached VSX catalog from {cache_file}")
                    for row in reader:
                        vsx_stars.append({
                            'id': row['id'],
                            'ra_deg': float(row['ra_deg']),
                            'dec_deg': float(row['dec_deg']),
                            'Type': row.get('Type', '')
                        })
                    return vsx_stars
        except Exception as e:
            if verbose: print(f"Error reading VSX cache: {e}")
            vsx_stars = []
        
    vsx_stars = fetch_vsx_catalog(ra_deg, dec_deg, radius_arcmin, verbose)
    
    if vsx_stars:
        with open(cache_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'ra_deg', 'dec_deg', 'Type'])
            writer.writeheader()
            for s in vsx_stars:
                writer.writerow(s)
    return vsx_stars

def mark_variable_stars(star_list, center_ra, center_dec, radius_arcmin, verbose=True):
    """
    Cross-matches a list of stars against the VSX catalog and adds 'is_variable' flag.
    """
    if not star_list: return star_list
    
    vsx_stars = get_vsx_stars(center_ra, center_dec, radius_arcmin, verbose=verbose)
    if not vsx_stars:
        for s in star_list: s['is_variable'] = False
        return star_list
        
    vsx_coords = SkyCoord(ra=[s['ra_deg'] for s in vsx_stars]*u.deg, dec=[s['dec_deg'] for s in vsx_stars]*u.deg)
    
    # Handle different RA/Dec key formats (ra_deg vs RA_ICRS etc)
    ra_keys = ['ra_deg', 'RA_ICRS', '_RAJ2000', 'RAJ2000']
    dec_keys = ['dec_deg', 'DE_ICRS', '_DEJ2000', 'DEJ2000']
    
    ra_key = next((k for k in ra_keys if k in star_list[0]), 'ra_deg')
    dec_key = next((k for k in dec_keys if k in star_list[0]), 'dec_deg')
    
    try:
        star_coords = SkyCoord(ra=[float(s[ra_key]) for s in star_list]*u.deg, 
                               dec=[float(s[dec_key]) for s in star_list]*u.deg)
    except:
        # Fallback if keys are missing or not floatable
        for s in star_list: s['is_variable'] = False
        return star_list
        
    idx, d2d, _ = star_coords.match_to_catalog_sky(vsx_coords)
    for i, s in enumerate(star_list):
        is_close = (d2d[i].arcsec < 5.0)
        matched_vsx = vsx_stars[idx[i]]
        # AAVSO sometimes includes constant stars in VSX for reference. Don't flag them as variable.
        var_type = matched_vsx.get('Type', '')
        if is_close and var_type and 'CST' not in str(var_type).upper():
            s['is_variable'] = True
            s['var_type'] = var_type
        else:
            s['is_variable'] = False
    return star_list


def match_and_calibrate(results, ref_catalog_file, filter_name, tolerance_arcsec=2.0, 
                        default_zp=23.399, run_new_calibration=True, output_report=None,
                        center_ra=None, center_dec=None, snr_threshold=10.0,
                        print_to_console=True, header=None, radius_arcmin=15.0):
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
    ref_stars = get_ref_stars(ref_catalog_file, center_ra, center_dec, radius_arcmin=radius_arcmin, verbose=print_to_console)

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
        
    # Exclude variable stars
    if center_ra is not None and center_dec is not None:
        mark_variable_stars(ref_stars, center_ra, center_dec, radius_arcmin, verbose=print_to_console)
        mark_variable_stars(results, center_ra, center_dec, radius_arcmin, verbose=False)
    else:
        for s in ref_stars: s['is_variable'] = False
        for s in results: s['is_variable'] = False
        
    valid_ref_stars = [s for s in ref_stars if not s.get('is_variable', False)]
    num_vars = len(ref_stars) - len(valid_ref_stars)
    if num_vars > 0:
        print(f"Excluded {num_vars} known variable stars from the reference calibration set.")
        
    mag_key = 'B_mag' if 'B' in filter_name.upper() else 'V_mag'
    
    # Identify Source
    source_name = ref_catalog_file
    if ref_catalog_file.upper() in ["ATLAS", "APASS", "GAIA_DR3", "LANDOLT STANDARD STAR CATALOGUE"]:
        source_name = f"Online VizieR ({ref_catalog_file.upper()})"
    else:
        source_name = f"Local File ({os.path.basename(ref_catalog_file)})"
    
    print(f"Reference Catalog: {source_name}")
    print(f"Using {mag_key} from reference catalog for calibration.")
    print(f"SNR Threshold for calibration stars: {snr_threshold}")
    
    ref_ra = [s['ra_deg'] for s in valid_ref_stars]
    ref_dec = [s['dec_deg'] for s in valid_ref_stars]
    ref_coords = SkyCoord(ra=ref_ra*u.deg, dec=ref_dec*u.deg)
    ref_mags = np.array([s[mag_key] for s in valid_ref_stars])
    
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
    airmass_val = header.get('AIRMASS', 1.0) if header else 1.0
    report_lines.append(f"- **Airmass**: {airmass_val:.3f}\n")
    report_lines.append(f"| Match ID | Catalog RA/Dec (HMS/DMS) | V mag | B mag | B-V | Inst Mag | Zero Point |")
    report_lines.append(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    report_lines.append(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    zps = []
    matched_ref_stars = [valid_ref_stars[i] for i in idx[match_mask]]
    
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
            
    return median_zp, std_zp


def resolve_auid(auid, target_hint=None, verbose=True):
    """
    Resolves an AAVSO Unique Identifier (AUID) to coordinates.
    Prioritizes the VSP Photometry Table if a target star name is provided.
    """
    if verbose:
        print(f"Resolving AUID '{auid}' via AAVSO VSP...")
    
    try:
        import urllib.request
        import re
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        # 1. Try VSP Photometry Table if we have a target hint
        if target_hint:
            if verbose: print(f"  Using target star hint: {target_hint}")
            # Clean target name for URL
            safe_target = urllib.parse.quote(target_hint)
            vsp_url = f"https://apps.aavso.org/vsp/photometry/?star={safe_target}&scale=D&orientation=ccd&type=photometry&fov=600&maglimit=20"
            req = urllib.request.Request(vsp_url, headers={'User-Agent': 'Mozilla/5.0'})
            
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    html = response.read().decode('utf-8', errors='ignore')
                
                # Look for AUID followed by Coords: <td>AUID</td>\s*<td>.*?\[([\d.]+)&deg;\]</td>\s*<td>.*?\[([\d.]+)&deg;\]</td>
                # Example: <td>000-BJS-559</td>\s*<td>09:37:11.94 [144.29974365&deg;]</td>\s*<td>43:58:20.9 [43.97247314&deg;]</td>
                pattern = rf'<td>{re.escape(auid)}</td>\s*<td>.*?\[([\d.]+)&deg;\]</td>\s*<td>.*?\[([\d.]+)&deg;\]</td>'
                match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
                
                if match:
                    ra_deg = float(match.group(1))
                    dec_deg = float(match.group(2))
                    
                    # Try to find the name/TYC label in the same row
                    # The name is usually in the last <td> before the next <tr>
                    row_end = html.find('</tr>', match.end())
                    row_snippet = html[match.end():row_end]
                    name_match = re.findall(r'<td>(.*?)</td>', row_snippet)
                    name = name_match[-1].strip() if name_match else auid
                    
                    if verbose:
                        print(f"  AUID {auid} found in {target_hint} chart as {name} at RA={ra_deg:.4f}, Dec={dec_deg:.4f}")
                    return {'ra': ra_deg, 'dec': dec_deg, 'name': name}
            except Exception as e:
                if verbose: print(f"  VSP Table lookup failed: {e}")

        # 2. Fallback: Try general VSX details page
        if verbose: print("  Falling back to general VSX detail search...")
        url = f"https://vsx.aavso.org/index.php?view=detail.top&ident={auid}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        name_match = re.search(r'Name:</td>\s*<td class="formfield">(.*?)</td>', html)
        name = name_match.group(1).strip() if name_match else auid
        
        coord_match = re.search(r'Coords \(J2000\):</td>\s*<td class="formfield">.*?\(([-\d.]+)\s+([-\d.]+)\)</td>', html, re.DOTALL)
        if coord_match:
            return {'ra': float(coord_match.group(1)), 'dec': float(coord_match.group(2)), 'name': name}
        else:
            hms_match = re.search(r'Coords \(J2000\):</td>\s*<td class="formfield">([\d\s:.+-]+)</td>', html, re.DOTALL)
            if hms_match:
                c = SkyCoord(hms_match.group(1).strip(), unit=(u.hourangle, u.deg))
                return {'ra': c.ra.deg, 'dec': c.dec.deg, 'name': name}
                
    except Exception as e:
        if verbose:
            print(f"AUID resolution failed: {e}")
    
    return None

def fetch_aavso_chart_stars(target_name, fov_arcmin=60.0, verbose=True):
    """
    Fetches the full list of comparison stars for a target from AAVSO VSP.
    Returns a list of dicts: {'auid', 'name', 'ra', 'dec', 'V_mag', 'B_mag'}
    """
    if verbose:
        print(f"Fetching AAVSO comparison stars for {target_name} (FOV={fov_arcmin}')...")
    
    try:
        import urllib.request
        import re
        import numpy as np
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        safe_target = urllib.parse.quote(target_name)
        # Use B=on to get explicit B magnitudes and use the user-defined FOV
        vsp_url = f"https://apps.aavso.org/vsp/photometry/?star={safe_target}&scale=D&orientation=ccd&type=photometry&fov={fov_arcmin}&maglimit=16&B=on"
        req = urllib.request.Request(vsp_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        # 1. Parse Headers to find column indices
        head_match = re.search(r'<thead>(.*?)</thead>', html, re.DOTALL)
        col_map = {'AUID': 0, 'RA': 1, 'Dec': 2, 'Label': 3, 'V': 4, 'B': -1, 'B-V': -1}
        if head_match:
            ths = re.findall(r'<th>(.*?)</th>', head_match.group(1), re.DOTALL)
            ths = [re.sub(r'<.*?>', '', t).strip() for t in ths]
            for i, th in enumerate(ths):
                if 'AUID' in th: col_map['AUID'] = i
                elif 'RA' in th: col_map['RA'] = i
                elif 'Dec' in th: col_map['Dec'] = i
                elif th == 'V': col_map['V'] = i
                elif th == 'B': col_map['B'] = i
                elif 'B-V' in th: col_map['B-V'] = i

        stars = []
        rows = re.findall(r'<tbody>(.*?)</tbody>', html, re.DOTALL)
        if not rows: rows = [html] # Fallback
        
        row_list = re.findall(r'<tr>(.*?)</tr>', rows[0], re.DOTALL)
        
        for row in row_list:
            if '000-' not in row: continue
            
            try:
                # 1. Get all cell text contents
                tds = re.findall(r'<td>(.*?)</td>', row, re.DOTALL)
                cells = [re.sub(r'<.*?>', '', t).strip() for t in tds]
                if len(cells) < 4: continue
                
                # 2. AUID and Coords
                auid = cells[col_map['AUID']]
                coord_matches = re.findall(r'\[([-\d.]+)&deg;\]', row)
                if len(coord_matches) < 2: continue
                ra = float(coord_matches[0])
                dec = float(coord_matches[1])
                
                # 3. Magnitudes
                v_mag = 0.0
                b_mag = 0.0
                
                # V Mag
                v_idx = col_map.get('V', 4)
                if len(cells) > v_idx:
                    v_val = cells[v_idx].split('(')[0].strip()
                    if v_val: v_mag = float(v_val)
                
                # B or B-V Mag
                b_idx = col_map.get('B', -1)
                bv_idx = col_map.get('B-V', -1)
                
                if b_idx != -1 and len(cells) > b_idx and cells[b_idx]:
                    b_val = cells[b_idx].split('(')[0].strip()
                    if b_val: b_mag = float(b_val)
                elif bv_idx != -1 and len(cells) > bv_idx and cells[bv_idx]:
                    bv_val = cells[bv_idx].split('(')[0].strip()
                    if bv_val: b_mag = v_mag + float(bv_val)
                
                if b_mag == 0.0: b_mag = v_mag
                
                # 4. Extract Name
                name = auid
                for c in reversed(cells):
                    if c and not re.match(r'[\d.\-\s()]+', c) and 'deg' not in c:
                        name = c
                        break
                
                stars.append({
                    'name': name,
                    'auid': auid,
                    'ra': ra,
                    'dec': dec,
                    'V_mag': round(v_mag, 3),
                    'B_mag': round(b_mag, 3)
                })
            except:
                continue
        
        if verbose:
            print(f"  Successfully retrieved {len(stars)} comparison stars for {target_name}.")
        return stars
        
    except Exception as e:
        if verbose:
            print(f"AAVSO Chart fetch failed: {e}")
    
    return []

if __name__ == '__main__':
    # Test the parser
    test_file = r'c:\Astro\StarID\photometry_refstars\reference_stars.csv'
    print(f"Testing parsing of: {test_file}")
    stars = read_reference_catalog(test_file)
    print(f"Successfully loaded {len(stars)} reference stars:")
    for s in stars[:10]:
        print(f"ID: {s['id']}, RA: {s['ra_deg']:.4f}, Dec: {s['dec_deg']:.4f}, V: {s['V_mag']:.3f}, B: {s['B_mag']:.3f}")
