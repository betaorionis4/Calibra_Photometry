import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# 'SkyCoord' from astropy is a powerful object used to represent, manipulate, and transform celestial coordinates.
from astropy.coordinates import SkyCoord
# 'astropy.units' allows us to attach physical units (like degrees or arcseconds) to numbers, preventing unit-conversion errors.
import astropy.units as u

# This imports a custom function from another file in this project.
from photometry.calibration import get_ref_stars, get_vsx_stars

def compute_zero_points(Tbv, Tb_bv, Tv_bv, kB, kV, B_ref, V_ref, b_ref, v_ref, XB_ref, XV_ref):
    """ Computes the photometric zero points for the field based on a single reference star. """
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
    """ Applies the zero points to calculate standard magnitudes.

    v4.16: B is computed as V + (B-V), so that the reported B, V and B-V are
    always consistent (previously B came from the separate B equation and,
    with mutually inconsistent coefficients, B - V differed from B-V by up to
    0.2 mag). Z_B and Tb_bv are therefore no longer needed here (kept in the
    signature for compatibility).
    """
    b0_t = b_t - kB * XB_t
    v0_t = v_t - kV * XV_t
    bv0_t = b0_t - v0_t
    BV_t = Tbv * bv0_t + Z_BV
    V_t = v0_t + Tv_bv * BV_t + Z_V
    B_t = V_t + BV_t
    return B_t, V_t, BV_t


def target_errors(sig_b, sig_v, BV_t, bv0_t, Tbv, Tv_bv, sZ_BV, sZ_V, sTbv=0.0, sTv=0.0,
                  BV_comp=0.0, bv0_comp=0.0):
    """
    v4.16: 1-sigma uncertainties of (B, V, B-V) for one star.
    Contributions: instrumental errors of b and v, zero-point errors (error of
    the ensemble mean), and colour-coefficient errors, which act on the colour
    difference to the comparison stars (coefficients are exact at the mean
    comparison-star colour). Correlations between the zero points are ignored.
      B-V = Tbv*(b0-v0) + Z_BV
      V   = v0 + Tv*(B-V) + Z_V = (1 - Tv*Tbv) v0 + Tv*Tbv b0 + ...
      B   = V + (B-V)       = (1 - a) v0 + a b0 + ...,  a = (1+Tv)*Tbv
    """
    d_bv0 = bv0_t - bv0_comp
    d_BV = BV_t - BV_comp
    s_bv = np.sqrt(Tbv**2 * (sig_b**2 + sig_v**2) + sZ_BV**2 + (d_bv0 * sTbv)**2)
    c = Tv_bv * Tbv
    s_v = np.sqrt((1 - c)**2 * sig_v**2 + c**2 * sig_b**2 + (Tv_bv * sZ_BV)**2 + sZ_V**2 + (d_BV * sTv)**2
                  + (Tv_bv * d_bv0 * sTbv)**2)
    a = (1 + Tv_bv) * Tbv
    s_b = np.sqrt((1 - a)**2 * sig_v**2 + a**2 * sig_b**2 + ((1 + Tv_bv) * sZ_BV)**2 + sZ_V**2 + (d_BV * sTv)**2
                  + ((1 + Tv_bv) * d_bv0 * sTbv)**2)
    return s_b, s_v, s_bv


def run_differential_photometry(csv_b, csv_v, ref_catalog, k_b, k_v, Tbv, Tb_bv, Tv_bv, radius_arcmin=15.0, manual_ref_coord=None, target_mode='all', manual_target_coord=None, axes=None,
                                comp_mode='ensemble', min_snr=30.0, peak_limit=30000.0, exclude_blended=True,
                                Tbv_err=0.0, Tb_bv_err=0.0, Tv_bv_err=0.0, check_coord=None, bv_range=(-0.2, 1.6)):
    """
    Reads two CSVs (B and V), matches the stars, determines the zero points and
    applies differential photometry to all common stars.

    v4.16:
      comp_mode='ensemble' (default): zero points Z_BV, Z_V (and Z_B) are the
        sigma-clipped mean over ALL suitable comparison stars (catalogue match,
        not a known variable, SNR >= min_snr in B and V, peak <= peak_limit,
        no neighbour in the annulus, catalogue B-V within bv_range); their
        uncertainty is the error of the mean.
      comp_mode='single' or a manual/named reference star: one comparison star
        (previous behaviour; automatic choice now = star closest in brightness
        and colour to the median of the field instead of the BRIGHTEST one).
      All results carry 1-sigma errors (B_err, V_err, B_V_err), and an optional
      check star (check_coord = (ra, dec) in deg) is reported against the
      catalogue.
    """
    # 1. Read CSVs
    # This is an inner function (a function defined inside another function). It acts as a local helper
    # and has access to variables in the outer function's scope.
    def read_csv_data(filepath):
        data = []  # Initialize an empty list
        if not os.path.exists(filepath): return data  # Guard clause: return early if file doesn't exist
        
        # 'with open(...)' is a context manager. It automatically ensures the file is safely closed 
        # after the block finishes, even if an error occurs inside the block.
        with open(filepath, 'r') as f:
            # DictReader reads the CSV and automatically converts each row into a Python dictionary,
            # using the first row (headers) as the dictionary keys.
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    data_b = read_csv_data(csv_b)
    data_v = read_csv_data(csv_v)
    
    # Check if lists are empty. In Python, empty lists evaluate to 'False' in a boolean context.
    if not data_b or not data_v:
        return "Error: Could not read B or V CSV files."
        
    # 2. Match B and V stars
    # 'List Comprehensions' are a concise, "Pythonic" way to create new lists based on existing ones.
    # The 'if' statement at the end acts as a filter.
    # Here we filter the data to only include rows that have 'ra_deg', 'dec_deg', and 'mag_inst' populated.
    # 'r.get("key")' safely attempts to get a value from a dictionary; if the key doesn't exist, it returns None.
    valid_b = [r for r in data_b if r.get('ra_deg') and r.get('dec_deg') and r.get('mag_inst')]
    
    # Extract just the RA and Dec values, converting them from strings to floating-point numbers.
    ra_b = [float(r['ra_deg']) for r in valid_b]
    dec_b = [float(r['dec_deg']) for r in valid_b]
    
    valid_v = [r for r in data_v if r.get('ra_deg') and r.get('dec_deg') and r.get('mag_inst')]
    ra_v = [float(r['ra_deg']) for r in valid_v]
    dec_v = [float(r['dec_deg']) for r in valid_v]
    
    if not valid_b or not valid_v:
        return "Error: Missing valid coordinates or instrumental magnitudes in input CSVs."
        
    # Create SkyCoord arrays. By multiplying by 'u.deg', we attach the "degrees" unit.
    # SkyCoord allows us to perform spherical geometry (like calculating angular distances on the sky) without writing complex trig formulas.
    coords_b = SkyCoord(ra=ra_b*u.deg, dec=dec_b*u.deg)
    coords_v = SkyCoord(ra=ra_v*u.deg, dec=dec_v*u.deg)
    
    # 'match_to_catalog_sky' is an Astropy method that finds the closest neighbor for each coordinate.
    # It returns 'idx' (the indices of the closest matches), 'd2d' (the 2D angular distances), and 'd3d' (3D distances, ignored here using '_').
    # Using '_' as a variable name is a Python convention indicating "I don't care about this value."
    idx, d2d, _ = coords_b.match_to_catalog_sky(coords_v)
    
    # Vectorized boolean masking: 'mask' becomes a numpy array of True/False values,
    # where True means the distance 'd2d' is less than 2.0 arcseconds.
    mask = d2d < 2.0 * u.arcsec
    
    matched_pairs = []
    # 'enumerate' iterates over a list and provides both the index ('i') and the value ('is_match') simultaneously.
    for i, is_match in enumerate(mask):
        if is_match:
            # We append a dictionary to our list, storing the full data row for both the B and V detections that matched.
            matched_pairs.append({
                'b': valid_b[i],
                'v': valid_v[idx[i]]
            })
            
    if not matched_pairs:
        return "Error: No matching stars found between B and V results."
        
    # 3. Query Reference Catalog
    # We need the central coordinates of the image to query the online catalog efficiently.
    # We calculate the geometric center by averaging the Right Ascension (RA) and Declination (Dec) of all matched stars.
    valid_coords = [float(p['v']['ra_deg']) for p in matched_pairs]
    center_ra = sum(valid_coords) / len(valid_coords)
    valid_dec = [float(p['v']['dec_deg']) for p in matched_pairs]
    center_dec = sum(valid_dec) / len(valid_dec)
    
    # 'get_ref_stars' makes an HTTP request to the VizieR database to download standard catalog data for the requested area.
    ref_stars = get_ref_stars(ref_catalog, center_ra, center_dec, radius_arcmin=radius_arcmin, verbose=False)
    if not ref_stars:
        return f"Error: Could not retrieve reference stars from {ref_catalog}."
        
    # We reuse the coords_pairs object created earlier (the actual stars detected in the image)
    coords_pairs = SkyCoord(ra=[float(p['v']['ra_deg']) for p in matched_pairs]*u.deg, 
                            dec=[float(p['v']['dec_deg']) for p in matched_pairs]*u.deg)
                            
    # Fetch VSX catalog to mark variable stars
    vsx_stars = get_vsx_stars(center_ra, center_dec, radius_arcmin=radius_arcmin, verbose=False)
    if vsx_stars:
        vsx_coords = SkyCoord(ra=[s['ra_deg'] for s in vsx_stars]*u.deg, dec=[s['dec_deg'] for s in vsx_stars]*u.deg)
        idx_vsx, d2d_vsx, _ = coords_pairs.match_to_catalog_sky(vsx_coords)
        for i, pair in enumerate(matched_pairs):
            matched_vsx = vsx_stars[idx_vsx[i]]
            var_type = matched_vsx.get('Type', '')
            # Exclude constant stars (CST) from being marked as variable stars
            if d2d_vsx[i].arcsec < 2.0 and not (var_type and 'CST' in str(var_type).upper()):
                pair['is_variable'] = True
            else:
                pair['is_variable'] = False
    else:
        for pair in matched_pairs:
            pair['is_variable'] = False
            
    # We create another SkyCoord object, this time for the online catalog stars.
    coords_cat = SkyCoord(ra=[s['ra_deg'] for s in ref_stars]*u.deg, 
                          dec=[s['dec_deg'] for s in ref_stars]*u.deg)
                            
    # We match our detected stars to the downloaded reference catalog to figure out which star is which.
    idx_cat, d2d_cat, _ = coords_pairs.match_to_catalog_sky(coords_cat)
    mask_cat = d2d_cat < 2.0 * u.arcsec
    
    # 4. Comparison stars and zero points
    def _f(d, k, default=np.nan):
        try:
            v = d.get(k, default)
            return float(v) if v not in (None, "") else default
        except (TypeError, ValueError):
            return default

    def _is_yes(v):
        return str(v).strip().lower() in ('yes', 'true', '1')

    def _pair_ok(pair):
        for side in ('b', 'v'):
            d = pair[side]
            if _is_yes(d.get('saturated', 'No')):
                return False
            if exclude_blended and _is_yes(d.get('blended', 'No')):
                return False
            if _f(d, 'snr', 0.0) < min_snr:
                return False
            peak = _f(d, 'peak_raw')
            if not np.isfinite(peak):
                peak = _f(d, 'peak_adu')
            if peak_limit and np.isfinite(peak) and peak > peak_limit:
                return False
        return True

    def _zps_for(pair, cat_star):
        B_r, V_r = cat_star['B_mag'], cat_star['V_mag']
        b_r, v_r = float(pair['b']['mag_inst']), float(pair['v']['mag_inst'])
        XB_r = float(pair['b'].get('airmass') or 1.0)
        XV_r = float(pair['v'].get('airmass') or 1.0)
        return compute_zero_points(Tbv, Tb_bv, Tv_bv, k_b, k_v, B_r, V_r, b_r, v_r, XB_r, XV_r)

    comp_list = []   # list of (pair, cat_star)
    best_ref_pair = None
    best_ref_cat = None

    if manual_ref_coord:
        # Manual Mode (single named/coordinate comparison star)
        man_coord = SkyCoord(ra=manual_ref_coord[0]*u.deg, dec=manual_ref_coord[1]*u.deg)
        seps = coords_pairs.separation(man_coord)
        best_idx = np.argmin(seps)
        min_sep = seps[best_idx].arcsec
        if min_sep > 4.0:
            return f"Error: No star found within 4 arcsec of the manual coordinates (closest is {min_sep:.1f} arcsec away)."
        if not mask_cat[best_idx]:
            return "Error: The selected manual star was not found in the reference catalog."
        best_ref_pair = matched_pairs[best_idx]
        best_ref_cat = ref_stars[idx_cat[best_idx]]
        if np.isnan(best_ref_cat.get('B_mag', np.nan)) or np.isnan(best_ref_cat.get('V_mag', np.nan)):
            return "Error: The selected manual star does not have valid B or V magnitudes in the reference catalog."
        comp_list = [(best_ref_pair, best_ref_cat)]
    else:
        candidates = []
        for i, is_match in enumerate(mask_cat):
            if not is_match:
                continue
            pair = matched_pairs[i]
            cat_star = ref_stars[idx_cat[i]]
            if pair.get('is_variable', False) or not _pair_ok(pair):
                continue
            b_cat = cat_star.get('B_mag', np.nan)
            v_cat = cat_star.get('V_mag', np.nan)
            if np.isnan(b_cat) or np.isnan(v_cat):
                continue
            if not (bv_range[0] <= b_cat - v_cat <= bv_range[1]):
                continue
            candidates.append((pair, cat_star))
        if not candidates:
            return (f"Error: No suitable comparison stars (catalogue match, not variable, SNR >= {min_snr:g}, "
                    f"peak <= {peak_limit:g} ADU, {bv_range[0]} <= B-V <= {bv_range[1]}).")
        if comp_mode == 'single':
            # One star: closest to the median brightness and colour of the candidates
            vs = np.array([c[1]['V_mag'] for c in candidates]); bvs = np.array([c[1]['B_mag'] - c[1]['V_mag'] for c in candidates])
            dist = np.hypot((vs - np.median(vs)) / 1.0, (bvs - np.median(bvs)) / 0.3)
            comp_list = [candidates[int(np.argmin(dist))]]
        else:
            comp_list = candidates
        best_ref_pair, best_ref_cat = comp_list[0]

    # Zero points per comparison star and their robust mean
    from astropy.stats import sigma_clip
    Z = np.array([_zps_for(p, c) for p, c in comp_list])   # columns Z_BV, Z_B, Z_V
    if len(comp_list) >= 3:
        clip = sigma_clip(Z, sigma=3.0, maxiters=5, axis=0, masked=True)
        keep = ~np.any(np.ma.getmaskarray(clip), axis=1)
        if keep.sum() < 3:
            keep = np.ones(len(comp_list), bool)
    else:
        keep = np.ones(len(comp_list), bool)
    Zk = Z[keep]
    Z_BV, Z_B, Z_V = (float(np.mean(Zk[:, j])) for j in range(3))
    n_comp = int(keep.sum())
    if n_comp >= 2:
        sZ = np.std(Zk, axis=0, ddof=1) / np.sqrt(n_comp)
        scatter = np.std(Zk, axis=0, ddof=1)
    else:
        # single comparison star: instrumental errors + assumed catalogue error 0.02 mag
        p0 = comp_list[0][0]
        eb0 = _f(p0['b'], 'mag_inst_err', 0.01); ev0 = _f(p0['v'], 'mag_inst_err', 0.01)
        eb0 = eb0 if np.isfinite(eb0) else 0.01; ev0 = ev0 if np.isfinite(ev0) else 0.01
        sZ = np.array([np.hypot(Tbv * np.hypot(eb0, ev0), 0.028), np.hypot(eb0, 0.02), np.hypot(ev0, 0.02)])
        scatter = np.array([np.nan, np.nan, np.nan])
    sZ_BV, sZ_B, sZ_V = (float(x) for x in sZ)
    used_comps = [comp_list[i] for i in range(len(comp_list)) if keep[i]]
    comp_ids = set(id(p) for p, _ in used_comps)
    # mean comparison colours (coefficient errors act on the colour difference to these)
    bv0_comp = float(np.mean([(float(p['b']['mag_inst']) - k_b * float(p['b'].get('airmass') or 1.0))
                              - (float(p['v']['mag_inst']) - k_v * float(p['v'].get('airmass') or 1.0)) for p, _ in used_comps]))
    BV_comp = float(np.mean([c['B_mag'] - c['V_mag'] for _, c in used_comps]))

    # 5. Extract Reference Star Data (first comparison star, for the report)
    B_ref = best_ref_cat['B_mag']
    V_ref = best_ref_cat['V_mag']
    b_ref = float(best_ref_pair['b']['mag_inst'])
    v_ref = float(best_ref_pair['v']['mag_inst'])
    XB_ref = float(best_ref_pair['b'].get('airmass') or 1.0)
    XV_ref = float(best_ref_pair['v'].get('airmass') or 1.0)
    
    # 6. Apply to all matched targets
    output_rows = []
    
    target_pairs = matched_pairs
    if target_mode in ['manual', 'name'] and manual_target_coord is not None:
        tgt_coord = SkyCoord(ra=manual_target_coord[0]*u.deg, dec=manual_target_coord[1]*u.deg)
        seps = coords_pairs.separation(tgt_coord)
        best_idx = np.argmin(seps)
        min_sep = seps[best_idx].arcsec
        
        if min_sep > 4.0:
            return f"Error: No star found within 4 arcsec of the target coordinates (closest is {min_sep:.1f} arcsec away)."
        
        target_pairs = [matched_pairs[best_idx]]
    
    # Now we loop over every single matched star pair (B and V detections)
    for pair in target_pairs:
        # We extract the raw instrumental magnitudes and airmasses
        b_inst = float(pair['b']['mag_inst'])
        v_inst = float(pair['v']['mag_inst'])
        XB = float(pair['b'].get('airmass') or 1.0)
        XV = float(pair['v'].get('airmass') or 1.0)
        
        # We pass them into our helper function to compute the calibrated, true magnitudes
        B_t, V_t, BV_t = compute_target_BV(b_inst, v_inst, XB, XV, Tbv, Tb_bv, Tv_bv, Z_BV, Z_B, Z_V, k_b, k_v)
        sb = _f(pair['b'], 'mag_inst_err', np.nan); sv = _f(pair['v'], 'mag_inst_err', np.nan)
        bv0_t = (b_inst - k_b * XB) - (v_inst - k_v * XV)
        if np.isfinite(sb) and np.isfinite(sv):
            eB, eV, eBV = target_errors(sb, sv, BV_t, bv0_t, Tbv, Tv_bv, sZ_BV, sZ_V, Tbv_err, Tv_bv_err, BV_comp, bv0_comp)
        else:
            eB = eV = eBV = np.nan
        
        # We build a dictionary for this star and append it to our output list.
        # Notice the use of 'f-strings' (e.g. f"{B_t:.4f}"). This is Python's standard way of formatting strings.
        # The ':.4f' tells Python to format the floating-point number to exactly 4 decimal places.
        output_rows.append({
            'id_v': pair['v']['id'],
            'id_b': pair['b']['id'],
            'ra_deg': pair['v']['ra_deg'],
            'dec_deg': pair['v']['dec_deg'],
            'ra_hms': pair['v'].get('ra_hms', ''),
            'dec_dms': pair['v'].get('dec_dms', ''),
            'B_mag': f"{B_t:.4f}",
            'V_mag': f"{V_t:.4f}",
            'B_V': f"{BV_t:.4f}",
            'B_err': f"{eB:.4f}" if np.isfinite(eB) else "",
            'V_err': f"{eV:.4f}" if np.isfinite(eV) else "",
            'B_V_err': f"{eBV:.4f}" if np.isfinite(eBV) else "",
            'is_comp': 'Yes' if id(pair) in comp_ids else 'No',
            'v_inst': f"{v_inst:.4f}",
            'b_inst': f"{b_inst:.4f}",
            'airmass_v': f"{XV:.4f}",
            'airmass_b': f"{XB:.4f}",
            'is_variable': 'Yes' if pair.get('is_variable', False) else 'No'
        })
        
    # 7. Write output
    # 'os.path.dirname' extracts just the directory path from the full file path.
    output_dir = os.path.dirname(csv_v)
    # 'os.path.join' safely concatenates paths regardless of operating system (e.g. using \ on Windows and / on Linux).
    out_csv = os.path.join(output_dir, "differential_photometry_results.csv")
    out_md_report = os.path.join(output_dir, "differential_photometry_report.md")
    out_md_results = os.path.join(output_dir, "differential_photometry_results.md")
    
    # 'newline=''' is required in Python 3 when writing CSVs to prevent extra blank lines on Windows.
    with open(out_csv, 'w', newline='') as f:
        # 'DictWriter' takes a list of dictionaries and writes them to a CSV. 
        # 'fieldnames=output_rows[0].keys()' automatically grabs the dictionary keys to use as the CSV header row.
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
        
    ref_cat_id = best_ref_cat.get('cat_id', '')
    ref_gaia_id = best_ref_cat.get('gaia_id', '')
    ref_auid = best_ref_cat.get('auid', '')
    ref_vsx_name = best_ref_cat.get('vsx_name', '')
    ref_display_parts = []
    if ref_auid:
        ref_display_parts.append(f"AUID: {ref_auid}")
    if ref_vsx_name:
        ref_display_parts.append(f"VSX: {ref_vsx_name}")
    if ref_cat_id:
        ref_display_parts.append(ref_cat_id)
    if ref_gaia_id and ref_gaia_id not in ref_cat_id:
        ref_display_parts.append(ref_gaia_id)
    ref_name = " | ".join(ref_display_parts) if ref_display_parts else best_ref_cat['id']

    with open(out_md_report, 'w') as f:
        f.write("# Differential Photometry Report\n\n")
        f.write("## Comparison Stars\n")
        if n_comp > 1:
            f.write(f"- **Ensemble** of {n_comp} comparison stars ({len(comp_list) - n_comp} rejected by 3-sigma clipping); "
                    f"selection: SNR >= {min_snr:g}, peak <= {peak_limit:g} ADU, {bv_range[0]} <= B-V <= {bv_range[1]}, not variable"
                    f"{', no neighbour in the annulus' if exclude_blended else ''}.\n")
            f.write(f"- Star-to-star scatter of the zero points: Z_BV {scatter[0]:.4f}, Z_B {scatter[1]:.4f}, Z_V {scatter[2]:.4f} mag\n")
            f.write(f"- Mean comparison-star colour (B-V) = {BV_comp:.3f}\n\n")
            f.write("## First comparison star (example)\n")
        else:
            f.write("- **Single comparison star** (catalogue error assumed 0.02 mag)\n\n")
            f.write("## Reference Star\n")
        f.write(f"- Name/ID: {ref_name}\n")
        f.write(f"- RA/Dec: {best_ref_cat['ra_deg']:.5f}, {best_ref_cat['dec_deg']:.5f}\n")
        f.write(f"- Standard B: {B_ref:.4f}\n")
        f.write(f"- Standard V: {V_ref:.4f}\n")
        f.write(f"- Standard B-V: {B_ref - V_ref:.4f}\n")
        f.write(f"- Instrumental B: {b_ref:.4f}\n")
        f.write(f"- Instrumental V: {v_ref:.4f}\n")
        f.write(f"- Airmass B/V: {XB_ref:.3f} / {XV_ref:.3f}\n\n")
        
        f.write("## Calculated Zero Points\n")
        f.write(f"- $Z_{{BV}}$: {Z_BV:.4f} ± {sZ_BV:.4f}\n")
        f.write(f"- $Z_B$: {Z_B:.4f} ± {sZ_B:.4f} (not used: B = V + (B-V) since v4.16)\n")
        f.write(f"- $Z_V$: {Z_V:.4f} ± {sZ_V:.4f}\n")
        f.write(f"- Coefficients: Tbv = {Tbv:.4f} ± {Tbv_err:.4f}, Tv_bv = {Tv_bv:.4f} ± {Tv_bv_err:.4f} (Tb_bv = {Tb_bv:.4f} only enters Z_B)\n\n")
        f.write("Uncertainties of B, V, B-V (columns B_err, V_err, B_V_err) include the instrumental errors, the zero-point errors "
                "and the coefficient errors times the colour difference to the comparison stars.\n\n")

        # Check star
        if check_coord is not None:
            chk = SkyCoord(ra=check_coord[0]*u.deg, dec=check_coord[1]*u.deg)
            seps_c = coords_pairs.separation(chk)
            ci = int(np.argmin(seps_c))
            f.write("## Check Star\n")
            if seps_c[ci].arcsec > 4.0:
                f.write(f"No measured star within 4 arcsec of the check-star position (closest {seps_c[ci].arcsec:.1f}\").\n\n")
            else:
                pr = matched_pairs[ci]
                bi, vi = float(pr['b']['mag_inst']), float(pr['v']['mag_inst'])
                XBc, XVc = float(pr['b'].get('airmass') or 1.0), float(pr['v'].get('airmass') or 1.0)
                Bc, Vc, BVc = compute_target_BV(bi, vi, XBc, XVc, Tbv, Tb_bv, Tv_bv, Z_BV, Z_B, Z_V, k_b, k_v)
                sbc, svc = _f(pr['b'], 'mag_inst_err', 0.01), _f(pr['v'], 'mag_inst_err', 0.01)
                eBc, eVc, eBVc = target_errors(sbc, svc, BVc, (bi - k_b*XBc) - (vi - k_v*XVc), Tbv, Tv_bv, sZ_BV, sZ_V,
                                               Tbv_err, Tv_bv_err, BV_comp, bv0_comp)
                f.write(f"- Measured: V = {Vc:.3f} ± {eVc:.3f}, B = {Bc:.3f} ± {eBc:.3f}, B-V = {BVc:.3f} ± {eBVc:.3f}\n")
                if mask_cat[ci]:
                    cc = ref_stars[idx_cat[ci]]
                    if np.isfinite(cc.get('V_mag', np.nan)) and np.isfinite(cc.get('B_mag', np.nan)):
                        f.write(f"- Catalogue: V = {cc['V_mag']:.3f}, B = {cc['B_mag']:.3f}, B-V = {cc['B_mag'] - cc['V_mag']:.3f}\n")
                        f.write(f"- Difference (meas - cat): dV = {Vc - cc['V_mag']:+.3f}, dB = {Bc - cc['B_mag']:+.3f}, "
                                f"d(B-V) = {BVc - (cc['B_mag'] - cc['V_mag']):+.3f}\n")
                if id(pr) in comp_ids:
                    f.write("- Note: this star is also in the comparison ensemble.\n")
                f.write("\n")
        
        f.write(f"Detailed results saved to: `{out_csv}` and `{out_md_results}`\n\n")
        
    # 8. Accuracy Evaluation
    if len(target_pairs) > 1 and target_mode == 'all':
        # We can assign multiple variables on a single line by separating them with commas.
        dB_list, dV_list, dBV_list = [], [], []
        b_inst_list, v_inst_list = [], []
        b_cat_list, v_cat_list = [], []
        star_info_B, star_info_V, star_info_BV = [], [], []
        
        for i, is_match in enumerate(mask_cat):
            if is_match:
                cat_star = ref_stars[idx_cat[i]]
                # Single comparison star: skip it (its deviation is trivially 0).
                # Ensemble: comparison stars are included (in-sample; noted in the report).
                if n_comp == 1 and cat_star['id'] == best_ref_cat['id']:
                    continue
                    
                # 'np.nan' represents "Not a Number". We use it to gracefully handle missing catalog data.
                b_cat = cat_star.get('B_mag', np.nan)
                v_cat = cat_star.get('V_mag', np.nan)
                
                # If either value is missing (NaN), we skip this star.
                if np.isnan(b_cat) or np.isnan(v_cat):
                    continue
                    
                # Skip known variable stars to avoid polluting the scatter metrics
                if target_pairs[i].get('is_variable', False):
                    continue
                    
                bv_cat = b_cat - v_cat
                row = output_rows[i]
                
                dB_list.append(float(row['B_mag']) - b_cat)
                dV_list.append(float(row['V_mag']) - v_cat)
                dBV_list.append(float(row['B_V']) - bv_cat)
                
                b_inst = float(row['b_inst'])
                v_inst = float(row['v_inst'])
                b_cal = float(row['B_mag'])
                v_cal = float(row['V_mag'])
                bv_cal = float(row['B_V'])
                
                b_inst_list.append(b_inst)
                v_inst_list.append(v_inst)
                b_cat_list.append(b_cat)
                v_cat_list.append(v_cat)
                
                ra_str = row['ra_hms'].replace('h', ':').replace('m', ':').replace('s', '')
                dec_str = row['dec_dms'].replace('d', ':').replace('m', ':').replace('s', '')
                
                cat_id = cat_star.get('cat_id', '')
                gaia_id = cat_star.get('gaia_id', '')
                auid = cat_star.get('auid', '')
                vsx_name = cat_star.get('vsx_name', '')
                
                display_parts = []
                if auid:
                    display_parts.append(f"AUID: {auid}")
                if vsx_name:
                    display_parts.append(f"VSX: {vsx_name}")
                if cat_id:
                    display_parts.append(cat_id)
                if gaia_id and gaia_id not in cat_id:
                    display_parts.append(gaia_id)
                    
                star_id = " | ".join(display_parts) if display_parts else cat_star['id']
                star_info_B.append({
                    'id': star_id, 'ra_hms': ra_str, 'dec_dms': dec_str,
                    'cat_mag': b_cat, 'inst_mag': b_inst, 'cal_mag': b_cal, 'dev': b_cal - b_cat
                })
                star_info_V.append({
                    'id': star_id, 'ra_hms': ra_str, 'dec_dms': dec_str,
                    'cat_mag': v_cat, 'inst_mag': v_inst, 'cal_mag': v_cal, 'dev': v_cal - v_cat
                })
                star_info_BV.append({
                    'id': star_id, 'ra_hms': ra_str, 'dec_dms': dec_str,
                    'cat_mag': bv_cat, 'inst_mag': b_inst - v_inst, 'cal_mag': bv_cal, 'dev': bv_cal - bv_cat
                })
                
        # In Python, non-empty lists evaluate to 'True'. This checks if we actually collected any data.
        if dB_list and dV_list and dBV_list:
            # We convert standard Python lists into highly optimized Numpy arrays for mathematical operations.
            dB, dV, dBV = np.array(dB_list), np.array(dV_list), np.array(dBV_list)
            b_inst_arr = np.array(b_inst_list)
            v_inst_arr = np.array(v_inst_list)
            b_cat_arr = np.array(b_cat_list)
            v_cat_arr = np.array(v_cat_list)
            
            bv_inst_arr = b_inst_arr - v_inst_arr
            bv_cat_arr = b_cat_arr - v_cat_arr
            
            # Perform linear regressions
            from scipy.stats import linregress
            if len(b_inst_arr) > 1:
                slope_B, intercept_B, r_val_B, p_val_B, std_err_B = linregress(b_inst_arr, b_cat_arr)
                std_B_res = np.std(b_cat_arr - (slope_B * b_inst_arr + intercept_B))
            else:
                slope_B, intercept_B, std_B_res = 1.0, 0.0, 0.0
                
            if len(v_inst_arr) > 1:
                slope_V, intercept_V, r_val_V, p_val_V, std_err_V = linregress(v_inst_arr, v_cat_arr)
                std_V_res = np.std(v_cat_arr - (slope_V * v_inst_arr + intercept_V))
            else:
                slope_V, intercept_V, std_V_res = 1.0, 0.0, 0.0
                
            if len(bv_inst_arr) > 1:
                slope_BV, intercept_BV, r_val_BV, p_val_BV, std_err_BV = linregress(bv_inst_arr, bv_cat_arr)
                std_BV_res = np.std(bv_cat_arr - (slope_BV * bv_inst_arr + intercept_BV))
            else:
                slope_BV, intercept_BV, std_BV_res = 1.0, 0.0, 0.0
            
            # 'norm.fit' takes an array of data and calculates the best-fit Gaussian distribution.
            # It returns the mean (mu) and standard deviation (std).
            mu_B, std_B = norm.fit(dB)
            mu_V, std_V = norm.fit(dV)
            mu_BV, std_BV = norm.fit(dBV)
            
            # Plotting
            if axes is None:
                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                fig.suptitle('Differential Photometry Accuracy vs Catalog', fontsize=14)
                is_standalone = True
            else:
                fig = axes.flatten()[0].figure
                is_standalone = False
            
            axes_flat = axes.flatten()
            for ax in axes_flat: ax.clear()
            
            # Plot 1: B instrumental vs true magnitude
            ax = axes_flat[0]
            scatter_B = ax.scatter(b_inst_arr, b_cat_arr, alpha=0.7, color='steelblue', edgecolors='k', label='Stars', picker=5)
            scatter_B.star_info_list = star_info_B
            if len(b_inst_arr) > 1:
                x_fit = np.linspace(np.min(b_inst_arr), np.max(b_inst_arr), 100)
                y_fit = slope_B * x_fit + intercept_B
                ax.plot(x_fit, y_fit, color='firebrick', linestyle='--', linewidth=1.5, label='Fit')
            ax.set_title(f"B-Filter: True vs Inst\nSlope = {slope_B:.3f} | Residual $\\sigma$ = {std_B_res:.3f} mag", fontsize=9)
            ax.set_xlabel("Instrumental Mag (b_inst)", fontsize=8)
            ax.set_ylabel("Catalog Mag (B)", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='best', fontsize=8)
            
            # Plot 2: V instrumental vs true magnitude
            ax = axes_flat[1]
            scatter_V = ax.scatter(v_inst_arr, v_cat_arr, alpha=0.7, color='seagreen', edgecolors='k', label='Stars', picker=5)
            scatter_V.star_info_list = star_info_V
            if len(v_inst_arr) > 1:
                x_fit = np.linspace(np.min(v_inst_arr), np.max(v_inst_arr), 100)
                y_fit = slope_V * x_fit + intercept_V
                ax.plot(x_fit, y_fit, color='firebrick', linestyle='--', linewidth=1.5, label='Fit')
            ax.set_title(f"V-Filter: True vs Inst\nSlope = {slope_V:.3f} | Residual $\\sigma$ = {std_V_res:.3f} mag", fontsize=9)
            ax.set_xlabel("Instrumental Mag (v_inst)", fontsize=8)
            ax.set_ylabel("Catalog Mag (V)", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='best', fontsize=8)
            
            # Plot 3: B-V instrumental vs true color
            ax = axes_flat[2]
            scatter_BV = ax.scatter(bv_inst_arr, bv_cat_arr, alpha=0.7, color='darkorchid', edgecolors='k', label='Stars', picker=5)
            scatter_BV.star_info_list = star_info_BV
            if len(bv_inst_arr) > 1:
                x_fit = np.linspace(np.min(bv_inst_arr), np.max(bv_inst_arr), 100)
                y_fit = slope_BV * x_fit + intercept_BV
                ax.plot(x_fit, y_fit, color='firebrick', linestyle='--', linewidth=1.5, label='Fit')
            ax.set_title(f"B-V Color: True vs Inst\nSlope = {slope_BV:.3f} | Residual $\\sigma$ = {std_BV_res:.3f} mag", fontsize=9)
            ax.set_xlabel("Instrumental Color (b-v)", fontsize=8)
            ax.set_ylabel("Catalog Color (B-V)", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='best', fontsize=8)

            def plot_hist(ax, data, mu, std, title, xlabel):
                n, bins, patches = ax.hist(data, bins='auto', density=True, alpha=0.6, color='steelblue')
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                ax.plot(x, p, 'k', linewidth=2)
                title_str = f"{title}\n$\\mu$ = {mu:.3f} | $\\sigma$ = {std:.3f} mag"
                ax.set_title(title_str, fontsize=9)
                ax.set_xlabel(xlabel, fontsize=8)
                ax.set_ylabel('Density', fontsize=8)
                ax.tick_params(labelsize=8)
                ax.grid(True, linestyle=':', alpha=0.6)
                
            plot_hist(axes_flat[3], dB, mu_B, std_B, 'Residuals $\\Delta B$ (Calibrated - Catalog)', '$\\Delta B$ [mag]')
            plot_hist(axes_flat[4], dV, mu_V, std_V, 'Residuals $\\Delta V$ (Calibrated - Catalog)', '$\\Delta V$ [mag]')
            plot_hist(axes_flat[5], dBV, mu_BV, std_BV, 'Residuals $\\Delta(B-V)$ (Calibrated - Catalog)', '$\\Delta(B-V)$ [mag]')
            
            # Interactive Picker event callback (creates a non-modal popup near cursor)
            def on_pick(event):
                artist = event.artist
                if hasattr(artist, 'star_info_list'):
                    ind = event.ind[0]
                    info = artist.star_info_list[ind]
                    
                    msg = f"Star ID: {info['id']}\n"
                    msg += f"RA/Dec: {info['ra_hms']} / {info['dec_dms']}\n"
                    msg += f"Catalog Mag/Color: {info['cat_mag']:.3f}\n"
                    msg += f"Instrumental Mag/Color: {info['inst_mag']:.3f}\n"
                    msg += f"Calibrated Mag/Color: {info['cal_mag']:.3f}\n"
                    msg += f"Deviation (Cal - Cat): {info['dev']:+.3f} mag"
                    
                    try:
                        import tkinter as tk
                        parent_win = fig.canvas.get_tk_widget().winfo_toplevel()
                        
                        # Close previous non-modal star info window if it exists and is open
                        if hasattr(fig, '_star_info_win') and fig._star_info_win:
                            try:
                                fig._star_info_win.destroy()
                            except Exception:
                                pass
                            fig._star_info_win = None
                            
                        # Create a new non-modal popup window
                        pop = tk.Toplevel(parent_win)
                        pop.title("Star Info")
                        pop.geometry("340x160")
                        pop.resizable(False, False)
                        pop.transient(parent_win)
                        
                        # Position it near the mouse pointer to act as a tooltip
                        mx, my = parent_win.winfo_pointerxy()
                        pop.geometry(f"+{mx + 15}+{my - 50}")
                        
                        frame = tk.Frame(pop, padx=10, pady=10)
                        frame.pack(fill="both", expand=True)
                        
                        text_box = tk.Text(frame, font=("Consolas", 9), bg=frame.cget("bg"), fg="black", relief="flat", borderwidth=0, height=6, wrap="word")
                        text_box.insert("1.0", msg)
                        text_box.config(state="disabled")
                        text_box.pack(fill="both", expand=True)
                        
                        # Add simple right-click menu to Copy All
                        def show_context_menu(e):
                            menu = tk.Menu(pop, tearoff=0)
                            menu.add_command(label="Copy All", command=lambda: [parent_win.clipboard_clear(), parent_win.clipboard_append(msg)])
                            menu.post(e.x_root, e.y_root)
                        text_box.bind("<Button-3>", show_context_menu)
                        
                        fig._star_info_win = pop
                    except Exception:
                        print(f"\n[Picked Star Info]\n{msg}\n")
                        
            # Register pick handler (clear older pickers if any to prevent duplicate popups)
            if hasattr(fig.canvas, '_calibra_pick_cid'):
                fig.canvas.mpl_disconnect(fig.canvas._calibra_pick_cid)
            fig.canvas._calibra_pick_cid = fig.canvas.mpl_connect('pick_event', on_pick)
            
            fig.tight_layout()
            # Explicit padding adjustment to prevent labels from overlapping neighboring subplots
            fig.subplots_adjust(wspace=0.38, hspace=0.48)
            
            os.makedirs('photometry_plots', exist_ok=True)
            plot_path = os.path.join('photometry_plots', 'diff_photometry_deviations.png')
            if is_standalone:
                plt.savefig(plot_path)
                plt.close(fig)
            else:
                fig.savefig(plot_path)
            
            # Append to report
            with open(out_md_report, 'a') as f:
                f.write("## Accuracy Evaluation\n")
                if n_comp == 1:
                    f.write("Comparison of computed magnitudes against catalog values (excluding the reference star).\n\n")
                else:
                    f.write(f"Comparison of computed magnitudes against catalog values (all non-variable catalogue stars; "
                            f"the {n_comp} ensemble comparison stars are included, i.e. partly in-sample).\n\n")
                f.write(f"- **$\\Delta B$ (Calibrated - Catalog)**: $\\mu$ = {mu_B:.4f}, $\\sigma$ = {std_B:.4f}\n")
                f.write(f"- **$\\Delta V$ (Calibrated - Catalog)**: $\\mu$ = {mu_V:.4f}, $\\sigma$ = {std_V:.4f}\n")
                f.write(f"- **$\\Delta(B-V)$**: $\\mu$ = {mu_BV:.4f}, $\\sigma$ = {std_BV:.4f}\n\n")
                f.write("## Instrumental vs. Catalog Regression\n")
                f.write(f"- **B-Filter**: Slope = {slope_B:.4f}, Residual $\\sigma$ = {std_B_res:.4f} mag\n")
                f.write(f"- **V-Filter**: Slope = {slope_V:.4f}, Residual $\\sigma$ = {std_V_res:.4f} mag\n")
                f.write(f"- **B-V Color**: Slope = {slope_BV:.4f}, Residual $\\sigma$ = {std_BV_res:.4f} mag\n\n")
                f.write(f"Plot saved to `{plot_path}`.\n\n")

    # Sort output rows by RA (increasing)
    # We use the 'sort' method with a 'key' argument. 
    # 'lambda r: float(r['ra_deg'])' is a tiny, anonymous function that tells the sorter to look at the 'ra_deg' value of each row.
    output_rows.sort(key=lambda r: float(r['ra_deg']))
    
    with open(out_md_results, 'w') as f:
        f.write("# Differential Photometry Results\n\n")
        f.write("| RA (HMS) / Dec (DMS) | RA (deg) | Dec (deg) | V mag | B mag | B-V | Variable |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for row in output_rows:
            # Replace h, m, s and d, m, s with colons
            ra_str = row['ra_hms'].replace('h', ':').replace('m', ':').replace('s', '')
            dec_str = row['dec_dms'].replace('d', ':').replace('m', ':').replace('s', '')
            
            # Add HMS/DMS first, then deg, then V, B, B-V
            coord_hms = f"{ra_str} / {dec_str}"
            f.write(f"| {coord_hms} | {float(row['ra_deg']):.5f} | {float(row['dec_deg']):.5f} | {row['V_mag']} | {row['B_mag']} | {row['B_V']} | {row['is_variable']} |\n")
        
    return (f"Success! Used {'an ensemble of ' + str(n_comp) + ' comparison stars' if n_comp > 1 else '1 comparison star'} "
            f"to calibrate {len(output_rows)} stars (Z_V = {Z_V:.3f} ± {sZ_V:.3f}). Saved to {out_csv}.")
