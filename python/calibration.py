import csv
import re
import os
from python import paths
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats, sigma_clip
from astroquery.vizier import Vizier
from astropy.table import Table

def fetch_online_catalog(ra_deg, dec_deg, radius_arcmin=15, catalog_name="ATLAS", verbose=True, use_cache=True):
    """
    Photometric reference stars of a field (ATLAS-RefCat2, APASS, Gaia DR3, Landolt).
    v4.46: despite the historical name, offline sources come first - local catalogue copy,
    disk cache, local RefCat2 as substitute - and VizieR only after them (see below).
    Every star carries 'cat_source' (where it came from).
    """
    cache_name = catalog_name              # cache files keep the name as passed (older files stay valid)
    if verbose:
        print(f"Reference catalogue {catalog_name} for RA={ra_deg:.4f}, Dec={dec_deg:.4f}...")
    
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

    # v4.46: offline first. Order:
    #   1. local copy of the chosen catalogue (Settings -> local catalogue folder)
    #   2. disk cache of an earlier online query of the chosen catalogue (<user folder>/cache/catalogs)
    #   3. APASS only: DR10 download from the AAVSO archive (option, default off) -> then local
    #   4. local ATLAS-RefCat2 copy as substitute (option 'prefer local catalogues', default on)
    #   5. VizieR (if online queries are allowed and VizieR answered recently)
    #   6. local ATLAS-RefCat2 substitute as last resort (if 4 was switched off)
    try:
        from python import local_catalogs as _lc
    except Exception:
        _lc = None

    def _local(substitute):
        if _lc is None:
            return None
        try:
            rows, used = _lc.query_best(catalog_name, ra_deg, dec_deg, radius_arcmin, verbose=verbose,
                                        substitute=substitute)
        except Exception as e:
            print(f"Local catalogue could not be read ({e}).")
            return None
        if rows is None:
            return None
        stars = stars_from_local_rows(rows, used)
        src = "local copy" if used == catalog_name else f"local {used} copy instead of {catalog_name}"
        for s_ in stars:
            s_['cat_source'] = src
        return _finish_ref_stars(stars, used, ra_deg, dec_deg, radius_arcmin, verbose, src)

    got = _local(substitute=False)                                    # 1
    if got is not None:
        return got
    if use_cache:                                                     # 2
        try:
            cached = get_cached_catalog(ra_deg, dec_deg, radius_arcmin, cache_name, verbose=verbose)
        except Exception:
            cached = None
        if cached:
            for s_ in cached:
                s_['cat_source'] = "cache of an earlier online query"
            return cached
    online_ok = _lc is None or _lc.online_allowed()
    prefer_off = _lc is not None and _lc.prefer_offline()
    if catalog_name == "APASS" and online_ok and _lc is not None and _lc.apass_auto():   # 3
        try:
            from python import apass_dr10 as _ad
            if _ad.auto_download(ra_deg, dec_deg, radius_arcmin, log=print):
                local_rows = _lc.query(catalog_name, ra_deg, dec_deg, radius_arcmin, verbose=verbose)
                if local_rows is not None:
                    ref_stars = stars_from_local_rows(local_rows, catalog_name)
                    for s_ in ref_stars:
                        s_['cat_source'] = "APASS DR10 download"
                    return _finish_ref_stars(ref_stars, catalog_name, ra_deg, dec_deg, radius_arcmin, verbose,
                                             "APASS DR10 download")
        except Exception as e:
            print(f"APASS DR10 automatic download: {e}")
    if prefer_off or not online_ok:                                   # 4
        got = _local(substitute=True)
        if got is not None:
            return got
    if not online_ok:
        print(f"{catalog_name}: not available in the local catalogue folder for this field, and online "
              "queries are switched off (Settings).")
        return []

    # v4.34: shorter timeout; a server that did not answer is skipped for a while
    from python import online as _online
    if not _online.available('vizier'):
        if verbose:
            print(f"{catalog_name}: VizieR was not reachable a short while ago - skipped (local data only).")
        return (_local(substitute=True) or []) if not prefer_off else []      # 6
    _online.set_timeout()
    
    v = Vizier(columns=['*', '_RAJ2000', '_DEJ2000'], row_limit=-1)
    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame='icrs')
    
    try:
        if verbose:
            print(f"Querying VizieR {catalog_id}...")
        result = v.query_region(coord, radius=radius_arcmin * u.arcmin, catalog=catalog_id)
        _online.ok('vizier')
    except Exception as e:
        print(f"Error querying VizieR: {e}")
        _online.failed('vizier', e)
        return (_local(substitute=True) or []) if not prefer_off else []      # 6

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
                    
                    v_mag, b_mag = atlas_to_vb(g, r)
                    
                    if 'Vmag' in cols and not np.isnan(float(row['Vmag'])): v_mag = float(row['Vmag'])
                    if 'Bmag' in cols and not np.isnan(float(row['Bmag'])): b_mag = float(row['Bmag'])

                elif catalog_name == "GAIA_DR3":
                    ra = float(row['RA_ICRS'])
                    dec = float(row['DE_ICRS'])
                
                    g = get_val('Gmag')
                    bp = get_val('BPmag')
                    rp = get_val('RPmag')
                
                    v_mag, b_mag = gaia_to_vb(g, bp, rp)

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
                    elif catalog_name == "APASS":
                        if 'recno' in cols:
                            cat_id = f"APASS {row['recno']}"
                        else:
                            try:
                                c_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                                ra_str = c_coord.ra.to_string(unit='hour', sep='', precision=2, pad=True)
                                dec_str = c_coord.dec.to_string(unit='degree', sep='', precision=1, pad=True, alwayssign=True).replace(' ', '')
                                cat_id = f"APASS J{ra_str}{dec_str}"
                            except:
                                cat_id = f"APASS_{len(ref_stars)}"
                    
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

    for s_ in ref_stars:
        s_['cat_source'] = "VizieR"
    ref_stars = _finish_ref_stars(ref_stars, catalog_name, ra_deg, dec_deg, radius_arcmin, verbose, "VizieR")
    if ref_stars and use_cache:            # v4.46: only online results are cached (local copies are fast)
        try:
            save_to_cache(ref_stars, ra_deg, dec_deg, radius_arcmin, cache_name, verbose=verbose)
        except Exception as e:
            print(f"Catalogue cache not written: {e}")
    return ref_stars


def atlas_to_vb(g, r):
    """Johnson V, B from Pan-STARRS g, r (Kostov et al. 2017). nan if g or r is missing
    (v4.15: no B/V without both bands)."""
    if g is None or r is None or np.isnan(g) or np.isnan(r):
        return np.nan, np.nan
    color = g - r
    return (g - 0.020 - 0.498 * color - 0.008 * color ** 2,
            g + 0.199 + 0.540 * color + 0.016 * color ** 2)


def gaia_to_vb(G, bp, rp):
    """Johnson V, B from Gaia G, BP, RP (Gaia DR3 documentation polynomials)."""
    if any(x is None or np.isnan(x) for x in (G, bp, rp)):
        return np.nan, np.nan
    c = bp - rp
    return (G + 0.02704 - 0.01424 * c + 0.2156 * c ** 2 - 0.01426 * c ** 3,
            G - 0.01448 + 0.6874 * c + 0.3604 * c ** 2 - 0.06718 * c ** 3 + 0.006061 * c ** 4)


def stars_from_local_rows(rows, catalog_name):
    """v4.26: convert raw rows of a local catalogue copy (python.local_catalogs.query)
    into Calibra's reference-star dicts - same conversions as for VizieR results."""
    from python import local_catalogs as lc
    cat = lc.normalize_name(catalog_name)
    out = []
    for row in rows:
        ra, dec = row['_ra'], row['_dec']
        g = r = G = bp = rp = np.nan
        if cat == 'ATLAS':
            g, r = lc.value(row, 'g_sdss'), lc.value(row, 'r_sdss')
            G, bp, rp = lc.value(row, 'G'), lc.value(row, 'BP'), lc.value(row, 'RP')   # v4.46: Gaia part of RefCat2
            v_mag, b_mag = atlas_to_vb(g, r)
            V_, B_ = lc.value(row, 'V'), lc.value(row, 'B')      # explicit V/B columns, if present
            if not np.isnan(V_): v_mag = V_
            if not np.isnan(B_): b_mag = B_
        elif cat == 'GAIA_DR3':
            G, bp, rp = lc.value(row, 'G'), lc.value(row, 'BP'), lc.value(row, 'RP')
            v_mag, b_mag = gaia_to_vb(G, bp, rp)
        elif cat == 'LANDOLT':
            v_mag = lc.value(row, 'V')
            bv = lc.value(row, 'BV')
            b_mag = v_mag + bv if not (np.isnan(v_mag) or np.isnan(bv)) else lc.value(row, 'B')
        else:  # APASS
            v_mag, b_mag = lc.value(row, 'V'), lc.value(row, 'B')
        if np.isnan(v_mag) or v_mag >= 18.0:
            continue
        ident = lc.text(row, 'id')
        prefix = {'GAIA_DR3': 'Gaia', 'ATLAS': 'ATLAS', 'APASS': 'APASS', 'LANDOLT': ''}.get(cat, '')
        if ident:
            cat_id = f"{prefix} {ident}".strip()
        else:
            try:
                c_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                cat_id = (f"{prefix or 'J'} J" if prefix else "J") + \
                    c_coord.ra.to_string(unit='hour', sep='', precision=2, pad=True) + \
                    c_coord.dec.to_string(unit='degree', sep='', precision=1, pad=True, alwayssign=True)
            except Exception:
                cat_id = f"{prefix}_{len(out)}"
        out.append({'id': f"online_{len(out)}", 'cat_id': cat_id, 'ra_deg': ra, 'dec_deg': dec,
                    'V_mag': v_mag, 'B_mag': b_mag,
                    'raw_g': g if cat == 'ATLAS' else np.nan, 'raw_r': r if cat == 'ATLAS' else np.nan,
                    'raw_G': G, 'raw_BP': bp, 'raw_RP': rp})
    return out


def _finish_ref_stars(ref_stars, catalog_name, ra_deg, dec_deg, radius_arcmin, verbose, source):
    if ref_stars:
        # Cross match with GAIA DR3 if not already GAIA_DR3
        if catalog_name != "GAIA_DR3":
            cross_match_with_gaia(ref_stars, ra_deg, dec_deg, radius_arcmin, verbose=verbose)
        else:
            for s in ref_stars:
                s['gaia_id'] = s.get('cat_id', '')
        # Cross match with local AUIDs
        cross_match_with_local_auids(ref_stars, verbose=verbose)
    if verbose:
        print(f"Successfully retrieved {len(ref_stars)} stars from {catalog_name} ({source}).")
    return ref_stars

def cross_match_with_gaia(ref_stars, ra_deg, dec_deg, radius_arcmin, verbose=True):
    if not ref_stars:
        return
    for s in ref_stars:
        s['gaia_id'] = ""
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        # v4.26: local Gaia copy first
        from python import local_catalogs as _lc
        rows = _lc.query("GAIA_DR3", ra_deg, dec_deg, radius_arcmin * 1.1, verbose=False)
        if rows is not None:
            if not rows:
                return
            gaia_table = {'Source': np.array([_lc.text(r, 'id') for r in rows], dtype=object)}
            gaia_coords = SkyCoord(ra=[r['_ra'] for r in rows], dec=[r['_dec'] for r in rows], unit=u.deg)
            gaia_table = [{'Source': x} for x in gaia_table['Source']]
        else:
            from python import online as _online
            # v4.46: Gaia source IDs are only labels - not fetched online when local catalogues
            # are preferred (RefCat2 has Gaia photometry but no source IDs)
            if not _lc.online_allowed() or _lc.prefer_offline() or not _online.available('vizier'):
                return
            if verbose:
                print(f"Querying GAIA DR3 for cross-matching near RA={ra_deg:.4f}, Dec={dec_deg:.4f}...")
            _online.set_timeout()
            v = Vizier(columns=['Source', '_RAJ2000', '_DEJ2000'], row_limit=-1)
            coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame='icrs')
            try:
                result = v.query_region(coord, radius=(radius_arcmin * 1.1) * u.arcmin, catalog="I/355")
                _online.ok('vizier')
            except Exception as e:
                _online.failed('vizier', e)
                raise

            if not result or len(result) == 0:
                if verbose:
                    print("No Gaia stars found for cross-matching.")
                return

            gaia_table = result[0]
            gaia_coords = SkyCoord(ra=gaia_table['_RAJ2000'], dec=gaia_table['_DEJ2000'], unit=u.deg)
        star_coords = SkyCoord(ra=[s['ra_deg'] for s in ref_stars], dec=[s['dec_deg'] for s in ref_stars], unit=u.deg)
        
        idx, d2d, _ = star_coords.match_to_catalog_sky(gaia_coords)
        for i, star in enumerate(ref_stars):
            dist_arcsec = d2d[i].arcsec
            if dist_arcsec < 2.0:
                star['gaia_id'] = f"Gaia {gaia_table[idx[i]]['Source']}"
        if verbose:
            matched_count = sum(1 for s in ref_stars if s.get('gaia_id'))
            print(f"Cross-matched {matched_count}/{len(ref_stars)} stars with GAIA DR3.")
    except Exception as e:
        print(f"Error cross-matching with Gaia: {e}")

def cross_match_with_local_auids(ref_stars, local_csv_path=None, verbose=True):
    local_csv_path = local_csv_path or paths.REFERENCE_STARS_FILE
    if not ref_stars:
        return
    for s in ref_stars:
        s['auid'] = ""
        
    if not os.path.exists(local_csv_path):
        return
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        local_stars = read_reference_catalog(local_csv_path)
        if not local_stars:
            return
            
        local_coords = SkyCoord(ra=[s['ra_deg'] for s in local_stars], dec=[s['dec_deg'] for s in local_stars], unit=u.deg)
        star_coords = SkyCoord(ra=[s['ra_deg'] for s in ref_stars], dec=[s['dec_deg'] for s in ref_stars], unit=u.deg)
        
        idx, d2d, _ = star_coords.match_to_catalog_sky(local_coords)
        for i, star in enumerate(ref_stars):
            dist_arcsec = d2d[i].arcsec
            if dist_arcsec < 2.0:
                star['auid'] = local_stars[idx[i]]['id']
        if verbose:
            matched_count = sum(1 for s in ref_stars if s.get('auid'))
            print(f"Cross-matched {matched_count}/{len(ref_stars)} stars with local AUIDs.")
    except Exception as e:
        print(f"Error cross-matching with local AUIDs: {e}")

def get_cached_catalog(ra, dec, radius, catalog_name, cache_dir=None, verbose=True):
    """
    Checks for a cached version of the catalog query.
    """
    cache_dir = cache_dir or paths.CATALOG_CACHE_DIR
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    cache_file = os.path.join(cache_dir, f"{catalog_name}_{ra:.3f}_{dec:.3f}_{float(radius):.1f}.csv")
    
    if os.path.exists(cache_file):
        if verbose:
            print(f"Loading cached catalog from {cache_file}")
        ref_stars = []
        with open(cache_file, 'r') as f:
            reader = csv.DictReader(f)
            # Treat files without the newer columns as cache misses to regenerate them
            if not reader.fieldnames or 'cat_id' not in reader.fieldnames or 'gaia_id' not in reader.fieldnames or 'auid' not in reader.fieldnames or 'vsx_name' not in reader.fieldnames:
                if verbose:
                    print(f"Cache file {cache_file} is outdated. Treating as cache miss.")
                return None
            for row in reader:
                ref_stars.append({
                    'id': row['id'],
                    'cat_id': '' if row.get('cat_id', '') in ('nan', None) else row.get('cat_id', ''),
                    'gaia_id': '' if row.get('gaia_id', '') in ('nan', None) else row.get('gaia_id', ''),
                    'auid': '' if row.get('auid', '') in ('nan', None) else row.get('auid', ''),
                    'vsx_name': '' if row.get('vsx_name', '') in ('nan', None) else row.get('vsx_name', ''),
                    'ra_deg': float(row['ra_deg']),
                    'dec_deg': float(row['dec_deg']),
                    'V_mag': float(row['V_mag']),
                    'B_mag': float(row['B_mag']),
                    'raw_g': float(row['raw_g']) if row.get('raw_g') and row['raw_g'] != 'nan' else np.nan,
                    'raw_r': float(row['raw_r']) if row.get('raw_r') and row['raw_r'] != 'nan' else np.nan,
                    'raw_G': float(row['raw_G']) if row.get('raw_G') and row['raw_G'] != 'nan' else np.nan,
                    'raw_BP': float(row['raw_BP']) if row.get('raw_BP') and row['raw_BP'] != 'nan' else np.nan,
                    'raw_RP': float(row['raw_RP']) if row.get('raw_RP') and row['raw_RP'] != 'nan' else np.nan
                })
        return ref_stars
    return None

def save_to_cache(ref_stars, ra, dec, radius, catalog_name, cache_dir=None, verbose=True):
    cache_dir = cache_dir or paths.CATALOG_CACHE_DIR
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f"{catalog_name}_{ra:.3f}_{dec:.3f}_{float(radius):.1f}.csv")
    fieldnames = ['id', 'cat_id', 'gaia_id', 'auid', 'vsx_name', 'ra_deg', 'dec_deg', 'V_mag', 'B_mag', 'raw_g', 'raw_r', 'raw_G', 'raw_BP', 'raw_RP']
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
    ref_catalog_file = paths.translate_legacy(ref_catalog_file)   # v5.0: old relative paths
    cat_upper = ref_catalog_file.upper()
    is_online = (not os.path.isfile(ref_catalog_file)
                 and any(k in cat_upper for k in ["ATLAS", "APASS", "GAIA", "LANDOLT"]))

    if is_online and center_ra is not None and center_dec is not None:
        cat_name = cat_upper # Normalized inside fetch_online_catalog
        # v4.46: local copy -> cache -> local RefCat2 substitute -> VizieR, all inside
        # fetch_online_catalog (before: the cache of old online results came before the local copy)
        ref_stars = fetch_online_catalog(center_ra, center_dec, radius_arcmin, cat_name, verbose=verbose)
        if ref_stars:
            mark_variable_stars(ref_stars, center_ra, center_dec, radius_arcmin, verbose=verbose)
    else:
        ref_stars = read_reference_catalog(ref_catalog_file)
    return ref_stars

def fetch_vsx_catalog(ra_deg, dec_deg, radius_arcmin=15, verbose=True):
    if verbose:
        print(f"Fetching VSX catalog from VizieR for RA={ra_deg:.4f}, Dec={dec_deg:.4f}...")
    
    # v4.26: local VSX copy first
    out = _vsx_local(ra_deg, dec_deg, radius_arcmin, verbose)
    if out is not None:
        return out
    try:
        from python import local_catalogs as _lc
    except Exception:
        _lc = None
    if _lc is not None and not _lc.online_allowed():
        return []
    from python import online as _online
    if not _online.available('vizier'):
        return []
    _online.set_timeout()
    
    v = Vizier(catalog="B/vsx/vsx", columns=['OID', 'Name', '_RAJ2000', '_DEJ2000', 'Type', 'max', 'min', 'f_min', 'Period'], row_limit=-1)
    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame='icrs')
    
    try:
        result = v.query_region(coord, radius=radius_arcmin * u.arcmin)
        _online.ok('vizier')
    except Exception as e:
        print(f"Error querying VSX: {e}")
        _online.failed('vizier', e)
        return []
        
    if not result or len(result) == 0:
        if verbose:
            print("No VSX variables found in this region.")
        return []
        
    table = result[0]
    vsx_stars = []
    for row in table:
        try:
            max_val = float(row['max']) if 'max' in row.colnames and not np.ma.is_masked(row['max']) else np.nan
            min_val = float(row['min']) if 'min' in row.colnames and not np.ma.is_masked(row['min']) else np.nan
            f_min_val = str(row['f_min']).strip().upper() if 'f_min' in row.colnames and not np.ma.is_masked(row['f_min']) else ''
            
            # If minimum magnitude column is actually the amplitude (f_min is Y or ( or min_val < max_val)
            if not np.isnan(max_val) and not np.isnan(min_val):
                is_amp = (f_min_val in ['Y', '(']) or (min_val < max_val)
                if is_amp:
                    min_val = max_val + min_val
                    
            vsx_stars.append({
                'id': str(row['Name']),
                'ra_deg': float(row['_RAJ2000']),
                'dec_deg': float(row['_DEJ2000']),
                'Type': str(row['Type']) if 'Type' in row.colnames and not np.ma.is_masked(row['Type']) else '',
                'max': max_val,
                'min': min_val,
                'Period': float(row['Period']) if 'Period' in row.colnames and not np.ma.is_masked(row['Period']) else np.nan
            })
        except:
            continue
            
    if verbose:
        print(f"Successfully retrieved {len(vsx_stars)} variable stars from VSX.")
    return vsx_stars

def _vsx_local(ra_deg, dec_deg, radius_arcmin, verbose=False):
    """Variables from the local VSX copy, or None if there is none for this field."""
    try:
        from python import local_catalogs as _lc
        rows = _lc.query("VSX", ra_deg, dec_deg, radius_arcmin, verbose=verbose)
    except Exception:
        return None
    if rows is None:
        return None
    out = []
    for r_ in rows:
        mx, mn = _lc.value(r_, 'max'), _lc.value(r_, 'min')
        if np.isfinite(mx) and np.isfinite(mn) and mn < mx:
            mn = mx + mn           # amplitude given instead of minimum
        out.append({'id': _lc.text(r_, 'id'), 'ra_deg': r_['_ra'], 'dec_deg': r_['_dec'],
                    'Type': _lc.text(r_, 'type'), 'max': mx, 'min': mn, 'Period': _lc.value(r_, 'period')})
    if verbose:
        print(f"VSX (local copy): {len(out)} variable stars.")
    return out


def _vsx_from_local_gcvs(ra_deg, dec_deg, radius_arcmin, verbose=False):
    """v4.46: variables of the local GCVS list in VSX format (offline fallback), or None."""
    try:
        from python import local_catalogs as _lc
        tg = _lc.gcvs_in_field(ra_deg, dec_deg, radius_arcmin)
    except Exception:
        return None
    if tg is None:
        return None
    out = []
    for t in tg:
        def _f(k):
            try:
                return float(t.get(k))
            except (TypeError, ValueError):
                return np.nan
        out.append({'id': t.get('name', ''), 'ra_deg': float(t['ra']), 'dec_deg': float(t['dec']),
                    'Type': t.get('type', '') or '', 'max': _f('max'), 'min': _f('min'), 'Period': _f('period')})
    if verbose:
        print(f"VSX not available - {len(out)} variables from the local GCVS list used instead.")
    return out


def get_vsx_stars(ra_deg, dec_deg, radius_arcmin=15, cache_dir=None, verbose=True):
    """Known variables of a field. v4.46 order: local VSX copy -> cache of earlier online
    queries -> VizieR -> local GCVS list (if VizieR is off or not reachable)."""
    cache_dir = cache_dir or paths.CATALOG_CACHE_DIR
    local = _vsx_local(ra_deg, dec_deg, radius_arcmin, verbose)
    if local is not None:
        return local
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    cache_file = os.path.join(cache_dir, f"VSX_{ra_deg:.4f}_{dec_deg:.4f}_{radius_arcmin}.csv")
    
    # Check if a cache file for this query or one covering it exists
    best_cache_file = None
    if os.path.exists(cache_file):
        best_cache_file = cache_file
    else:
        # Scan cache directory for a file with close coordinates and a larger or equal radius
        try:
            for fname in os.listdir(cache_dir):
                if fname.startswith("VSX_") and fname.endswith(".csv"):
                    parts = fname[:-4].split("_")
                    if len(parts) == 4:
                        try:
                            c_ra = float(parts[1])
                            c_dec = float(parts[2])
                            c_rad = float(parts[3])
                            
                            # Check if the center is within 0.05 degrees (3 arcmin)
                            dist_deg = np.sqrt(((c_ra - ra_deg) * np.cos(np.radians(dec_deg)))**2 + (c_dec - dec_deg)**2)
                            if dist_deg < 0.05:
                                # Check if cached radius adjusted for the offset covers the requested radius (with a 2 arcmin tolerance)
                                if (c_rad - dist_deg * 60.0) >= (radius_arcmin - 2.0):
                                    if verbose:
                                        print(f"VSX Cache: Reusing {fname} (covers requested region)")
                                    best_cache_file = os.path.join(cache_dir, fname)
                                    break
                        except ValueError:
                            continue
        except Exception as e:
            if verbose: print(f"Error scanning VSX cache directory: {e}")

    vsx_stars = []
    if best_cache_file and os.path.exists(best_cache_file):
        try:
            with open(best_cache_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # Allow loading old cached files that don't have the Period column yet (using nan fallback)
                is_old = 'Period' not in reader.fieldnames
                if is_old:
                    if verbose: print(f"VSX Cache: Reusing old cache (missing Period).")
                else:
                    if verbose: print(f"Loading cached VSX catalog from {best_cache_file}")
                
                for row in reader:
                    # Only keep stars within the requested radius of the requested center
                    star_ra = float(row['ra_deg'])
                    star_dec = float(row['dec_deg'])
                    dist_arcmin = np.sqrt(((star_ra - ra_deg) * np.cos(np.radians(dec_deg)))**2 + (star_dec - dec_deg)**2) * 60.0
                    if dist_arcmin <= radius_arcmin:
                        max_val = float(row['max']) if row.get('max') and row['max'] != 'nan' else np.nan
                        min_val = float(row['min']) if row.get('min') and row['min'] != 'nan' else np.nan
                        if not np.isnan(max_val) and not np.isnan(min_val) and min_val < max_val:
                            min_val = max_val + min_val
                        vsx_stars.append({
                            'id': row['id'],
                            'ra_deg': star_ra,
                            'dec_deg': star_dec,
                            'Type': row.get('Type', ''),
                            'max': max_val,
                            'min': min_val,
                            'Period': float(row['Period']) if (not is_old and row.get('Period') and row['Period'] != 'nan') else np.nan
                        })
                return vsx_stars
        except Exception as e:
            if verbose: print(f"Error reading VSX cache: {e}")
            vsx_stars = []
        
    vsx_stars = fetch_vsx_catalog(ra_deg, dec_deg, radius_arcmin, verbose)
    if not vsx_stars:
        # v4.46: VizieR switched off or not reachable -> local GCVS list (not cached as VSX)
        try:
            from python import local_catalogs as _lc
            from python import online as _online
            offline = not _lc.online_allowed() or not _online.available('vizier')
        except Exception:
            offline = True
        if offline:
            g = _vsx_from_local_gcvs(ra_deg, dec_deg, radius_arcmin, verbose)
            return g if g is not None else []
    
    if vsx_stars:
        with open(cache_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'ra_deg', 'dec_deg', 'Type', 'max', 'min', 'Period']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in vsx_stars:
                row = {k: s.get(k, np.nan) for k in fieldnames}
                writer.writerow(row)
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
        is_close = (d2d[i].arcsec < 10.0)  # Relaxed from 5.0 to 10.0 to handle boundary WCS distortions
        matched_vsx = vsx_stars[idx[i]]
        if is_close:
            s['vsx_name'] = matched_vsx['id']
        else:
            s['vsx_name'] = ""
        # AAVSO sometimes includes constant stars in VSX for reference. Don't flag them as variable.
        var_type = matched_vsx.get('Type', '')
        if is_close and 'CST' not in str(var_type or '').upper():
            s['is_variable'] = True
            s['var_type'] = var_type
        else:
            s['is_variable'] = False
    return star_list


def match_and_calibrate(results, ref_catalog_file, filter_name, tolerance_arcsec=2.0, 
                        default_zp=23.399, run_new_calibration=True, output_report=None,
                        center_ra=None, center_dec=None, snr_threshold=10.0,
                        print_to_console=True, header=None, radius_arcmin=15.0, band=None,
                        peak_limit=None, exclude_blended=True):
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
        
    # v4.15: band decided by the central filter classifier (was: 'B' in FILTER,
    # which treated e.g. "Bessell V" as B).
    if band not in ('B', 'V'):
        from python.filters import filter_band
        band = filter_band(filter_name)
    mag_key = 'B_mag' if band == 'B' else 'V_mag'
    
    # Identify Source
    source_name = ref_catalog_file
    if os.path.isfile(str(ref_catalog_file)):
        source_name = f"Local File ({os.path.basename(ref_catalog_file)})"
    else:
        source_name = f"Online VizieR ({ref_catalog_file})"
    
    print(f"Reference Catalog: {source_name}")
    print(f"Using {mag_key} from reference catalog for calibration.")
    print(f"SNR Threshold for calibration stars: {snr_threshold}")
    if peak_limit:
        print(f"Peak limit for calibration stars: {peak_limit:.0f} ADU (raw pixel value; protects against non-linearity)")
    
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
            # v4.16: also reject stars near non-linearity (peak_limit), stars whose
            # aperture is off the image and stars with a neighbour in the annulus
            peak = rs.get('peak_raw', rs.get('peak_adu', 0))
            too_bright = bool(peak_limit) and peak is not None and np.isfinite(peak) and peak > peak_limit
            if (rs.get('snr', 0) >= snr_threshold and not rs.get('saturated', False) and not too_bright
                    and not rs.get('edge', False) and not (exclude_blended and rs.get('blended', False))):
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
    # v4.16: uncertainty of the zero point itself = error of the (clipped) median,
    # 1.2533*sigma/sqrt(N). Previously the star-to-star scatter (std) was used as
    # the ZP error and added to every star (and was 0 for a single star).
    clipped = sigma_clip(zps, sigma=3.0, maxiters=5, masked=True)
    n_used = int(np.sum(~clipped.mask)) if np.ma.isMaskedArray(clipped) else len(zps)
    if n_used >= 2:
        zp_err = 1.2533 * std_zp / np.sqrt(n_used)
    else:
        zp_err = 0.05  # single calibration star: conservative floor
        print("WARNING: only one calibration star - ZP uncertainty set to 0.05 mag.")
    
    print(f"\nCalculated Zero Point: {median_zp:.3f} ± {zp_err:.4f} (median of {n_used} stars; star-to-star scatter {std_zp:.3f})")

    # v4.16: colour dependence of the zero point (diagnostic; NOT applied).
    colour_txt = None
    try:
        bvs = np.array([(rf.get('B_mag', np.nan) - rf.get('V_mag', np.nan)) for rf in matched_ref_stars], dtype=float)
        ok = np.isfinite(bvs) & np.isfinite(zps) & (~clipped.mask if np.ma.isMaskedArray(clipped) else True)
        if np.sum(ok) >= 10 and np.ptp(bvs[ok]) > 0.3:
            from scipy.stats import linregress
            x, y = bvs[ok], zps[ok]
            keep = np.ones(len(x), bool)
            for _ in range(5):
                lr = linregress(x[keep], y[keep])
                res = y - (lr.intercept + lr.slope * x)
                mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
                new_keep = np.abs(res) < 3 * max(mad, 1e-4)
                if np.array_equal(new_keep, keep): break
                keep = new_keep
            bv_med = float(np.median(x[keep]))
            colour_txt = (f"ZP colour dependence: dZP/d(B-V) = {lr.slope:+.3f} ± {lr.stderr:.3f} mag/mag "
                          f"({int(keep.sum())} stars, median comparison B-V = {bv_med:.2f}). "
                          f"The pipeline magnitudes are NOT colour-transformed: for a star with colour (B-V)*, "
                          f"the transformed value would differ by about {lr.slope:+.3f} x ((B-V)* - {bv_med:.2f}).")
            print(colour_txt)
    except Exception as e:
        print(f"(Colour diagnostic skipped: {e})")
    
    report_lines.append(f"\n## Results")
    report_lines.append(f"- **Calculated Median Zero Point**: {median_zp:.3f} ± {zp_err:.4f} (error of the median, {n_used} stars)")
    report_lines.append(f"- **Star-to-star scatter (std)**: {std_zp:.3f}")
    report_lines.append(f"- **Calibrated magnitudes are untransformed** (no colour term applied).")
    if colour_txt:
        report_lines.append(f"- {colour_txt}")
    report_lines.append("")
    
    if output_report:
        with open(output_report, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    
    for rs in results:
        if 'mag_inst' in rs and not np.isnan(rs['mag_inst']):
            rs['mag_calibrated'] = rs['mag_inst'] + median_zp
            mag_inst_err = rs.get('mag_inst_err', np.nan)
            if not np.isnan(mag_inst_err):
                rs['mag_calibrated_err'] = np.sqrt(mag_inst_err**2 + zp_err**2)
            else:
                rs['mag_calibrated_err'] = np.nan
        else:
            rs['mag_calibrated'] = np.nan
            rs['mag_calibrated_err'] = np.nan
            
    return median_zp, zp_err


def resolve_auid(auid, target_hint=None, verbose=True):
    """
    Resolves an AAVSO Unique Identifier (AUID) to coordinates.
    Prioritizes the VSP Photometry Table if a target star name is provided.
    """
    if verbose:
        print(f"Resolving AUID '{auid}' via AAVSO VSP...")
    
    try:
        import urllib.request
        import urllib.parse
        import re
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        from python import online as _online
        # 1. Try VSP Photometry Table if we have a target hint
        if target_hint and _online.available('aavso'):
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
                _online.failed('aavso', e)

        # 2. Fallback: Try VizieR VSX catalog directly by Name
        if verbose: print("  Falling back to VizieR VSX name search...")
        from astroquery.vizier import Vizier
        v = Vizier(columns=['OID', 'Name', '_RAJ2000', '_DEJ2000'], row_limit=1)
        try:
            if not _online.available('vizier'):
                return None
            _online.set_timeout()
            try:
                result = v.query_constraints(catalog='B/vsx/vsx', Name=auid)
            except Exception as e_:
                _online.failed('vizier', e_)
                raise
            if result and len(result) > 0:
                table = result[0]
                if len(table) > 0:
                    ra_deg = float(table['_RAJ2000'][0])
                    dec_deg = float(table['_DEJ2000'][0])
                    name = str(table['Name'][0])
                    if verbose:
                        print(f"  Found {name} in VSX VizieR at RA={ra_deg:.4f}, Dec={dec_deg:.4f}")
                    return {'ra': ra_deg, 'dec': dec_deg, 'name': name}
        except Exception as e:
            if verbose: print(f"  VizieR VSX lookup failed: {e}")
                
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
