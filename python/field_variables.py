"""
Light curves of several (known) variable stars in the same field  (v4.19).

Two ways to get the targets:
  a) parse_target_list(): a text/CSV file with names and coordinates
  b) query_field_variables(): AAVSO VSX (includes the GCVS) or the GCVS itself
     (VizieR B/gcvs) for the field covered by an image

Comparison stars for each target can be chosen automatically from the photometric
reference catalogue (choose_catalog_comparisons), so that calibrated magnitudes
can be produced for all targets in one run (time_series.run_multi_target_photometry).
"""
import csv
import os
from python import paths
import re
import numpy as np


# --------------------------------------------------------------------------
# a) Target list from a file
# --------------------------------------------------------------------------
# Column names (lower case, see _norm_col) accepted in list files. v4.23: also the
# Swedish headers of the SAAF variable-star section's GCVS export
# ("Var ID","Namn","Typ","RA","Dekl","Max","Min","Band","Epok","Period",...).
_NAME_COLS = ('name', 'namn', 'star', 'star name', 'object', 'objekt', 'target', 'var', 'variable',
              'designation', 'gcvs', 'gcvs name', 'vsx name', 'varname', 'id')
_RA_COLS = ('ra', 'ra_deg', 'raj2000', 'ra(deg)', 'ra (deg)', 'ra_j2000', 'alpha', 'ra_hms', 'ra(j2000)',
            'ra (j2000)', 'ra2000', 'ra j2000', 'rektascension', 'right ascension', '_raj2000')
_DEC_COLS = ('dec', 'dec_deg', 'dej2000', 'decj2000', 'dec(deg)', 'dec (deg)', 'dec_j2000', 'delta', 'dec_dms', 'de',
             'dec(j2000)', 'dec (j2000)', 'dec2000', 'dec j2000', 'decl', 'dekl', 'deklination', 'declination',
             'dek', '_dej2000')
_OPT_COLS = {'type': ('type', 'vartype', 'var_type', 'typ', 'variabeltyp', 'var type'),
             'period': ('period', 'per', 'p', 'period (d)', 'period_d', 'period(d)', 'per (d)'),
             'max': ('max', 'magmax', 'mag_max', 'max mag', 'maxmag'),
             'min': ('min', 'magmin', 'mag_min', 'min1', 'min mag', 'minmag'),
             'epoch': ('epoch', 'epok', 't0', 'epoch (hjd)', 'epoch_hjd', 'hjd0', 'jd0')}
_ALL_COLS = set(_NAME_COLS) | set(_RA_COLS) | set(_DEC_COLS) | {a for v in _OPT_COLS.values() for a in v}


def _norm_col(t):
    """'"Dekl."' -> 'dekl', 'RA  (J2000)' -> 'ra (j2000)'."""
    t = str(t).strip().strip('"').strip("'").strip().lower().rstrip('.:')
    return re.sub(r'\s+', ' ', t)


def _mag_value(txt, ref=None):
    """Magnitude from catalogue text: '11.60', '<15.0', '12.3:', 'V 12.1'; an
    amplitude in parentheses '(0.30)' (GCVS convention) is added to ref (max)."""
    t = str(txt or '').strip()
    if not t:
        return None
    amp = t.startswith('(')
    t = re.sub(r'[^0-9.+\-eE]', '', t.replace('(', '').replace(')', ''))
    v = _to_float(t)
    if v is None:
        return None
    if amp:
        return ref + v if ref is not None else None
    return v


def _to_float(x):
    try:
        v = float(str(x).strip())
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


_SEXA_RE = re.compile(r'^\s*([+-]?)\s*(\d{1,3})[\s:]+(\d{1,2})(?:[\s:]+(\d{1,2}(?:\.\d*)?))?\s*$')


def _sexa(txt):
    """v5.0.1: fast path for 'h m s' / 'h:m:s' / '+d m s' text -> (sign, value) or None."""
    m = _SEXA_RE.match(str(txt))
    if not m:
        return None
    a, b, c = float(m.group(2)), float(m.group(3)), float(m.group(4) or 0.0)
    if b >= 60 or c >= 60:
        return None
    return (-1.0 if m.group(1) == '-' else 1.0), a + b / 60.0 + c / 3600.0


def parse_coordinates(ra_txt, dec_txt):
    """RA/Dec from decimal degrees or sexagesimal text ('09:36:53.2' / '9 36 53.2' /
    '9h36m53.2s'; Dec '+44:04:00'). Decimal RA is taken as degrees. Returns (ra, dec) in deg."""
    ra_f, dec_f = _to_float(ra_txt), _to_float(dec_txt)
    if ra_f is not None and dec_f is not None:
        if not (0 <= ra_f < 360 and -90 <= dec_f <= 90):
            raise ValueError(f"coordinates out of range: {ra_txt} {dec_txt}")
        return ra_f, dec_f
    # v5.0.1: plain 'h m s' / '+d m s' without astropy (one SkyCoord per star took ~1 ms -
    # ~27 s for the 70000 rows of the local GCVS lists)
    r_, d_ = _sexa(ra_txt), _sexa(dec_txt)
    if r_ is not None and d_ is not None and r_[0] > 0:
        ra_d, dec_d = 15.0 * r_[1], d_[0] * d_[1]
        if 0 <= ra_d < 360 and -90 <= dec_d <= 90:
            return ra_d, dec_d
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(str(ra_txt).strip(), str(dec_txt).strip(), unit=(u.hourangle, u.deg))
    return float(c.ra.deg), float(c.dec.deg)


def _split_line(line, delim):
    if delim is None:
        return line.split()
    return [t.strip() for t in next(csv.reader([line], delimiter=delim))]


def _parse_tokens_no_header(tokens):
    """name + coordinates from one data line without header (name may contain spaces)."""
    # last 6 numeric tokens as 'h m s d m s'
    if len(tokens) >= 6 and all(_to_float(t) is not None for t in tokens[-6:]):
        try:
            ra, dec = parse_coordinates(" ".join(tokens[-6:-3]), " ".join(tokens[-3:]))
            return " ".join(tokens[:-6]).strip(), ra, dec
        except Exception:
            pass
    # last 2 tokens as coordinates (decimal or 'hh:mm:ss' form)
    if len(tokens) >= 2:
        try:
            ra, dec = parse_coordinates(tokens[-2], tokens[-1])
            return " ".join(tokens[:-2]).strip(), ra, dec
        except Exception:
            pass
    # last 6 tokens as 'h m s d m s'
    if len(tokens) >= 6:
        try:
            ra, dec = parse_coordinates(" ".join(tokens[-6:-3]), " ".join(tokens[-3:]))
            return " ".join(tokens[:-6]).strip(), ra, dec
        except Exception:
            pass
    # name only -> resolved later
    return " ".join(tokens).strip(), None, None


def parse_target_list(path):
    """
    Read targets from a text or CSV file. Accepted forms (one star per line,
    lines starting with '#' are ignored):
      * with header:  name,ra,dec[,type,period,...]   (comma, semicolon or tab)
      * without header:  RR Lyr  19:25:27.9  +42:47:03.7
                         RR Lyr, 291.3663, 42.7843
                         RR Lyr  19 25 27.9  +42 47 03.7
                         RR Lyr                     (name only: resolved via SIMBAD/VSX)
    RA may be decimal degrees or sexagesimal hours; Dec degrees.
    Returns (targets, problems): targets = list of dicts {name, ra, dec, type, period,
    max, min, source}; ra/dec are None for names that still need resolving.
    """
    with open(path, 'rb') as f:
        data = f.read()
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = data.decode('cp1252', errors='replace')     # Windows/Excel export (å, ä, ö)
    raw = [l.rstrip('\r') for l in text.split('\n')]
    lines = [l for l in raw if l.strip() and not l.lstrip().startswith('#')]
    targets, problems = [], []
    if not lines:
        return targets, ["file is empty"]

    sample = "\n".join(lines[:20])
    delim = None
    for d in (',', ';', '\t'):
        if d in sample:
            delim = d
            break

    first = [_norm_col(t) for t in _split_line(lines[0], delim)]
    # v4.23: a first line made of known column names is a header (also without RA/Dec)
    has_header = sum(t in _ALL_COLS for t in first) >= 2 or (
        any(t in _RA_COLS for t in first) and any(t in _DEC_COLS for t in first))
    src = f"file:{os.path.basename(path)}"

    if has_header:
        def col(names):
            for nm in names:                     # alias order = priority
                if nm in first:
                    return first.index(nm)
            return None
        i_ra, i_dec, i_name = col(_RA_COLS), col(_DEC_COLS), col(_NAME_COLS)
        if i_ra is None or i_dec is None:
            missing = " and ".join(x for x, i in (("RA", i_ra), ("Dec", i_dec)) if i is None)
            raise ValueError(
                f"The list has no {missing} column.\n\nA target list needs at least the columns "
                f"name, RA and Dec. Found columns: {', '.join(first)}.\n"
                "Accepted names e.g. RA / RAJ2000 / Rektascension and Dec / DEJ2000 / Dekl / Declination; "
                "RA as hh mm ss (or degrees), Dec as +dd mm ss (or degrees).")
        if i_name is None:
            problems.append("The list has no name column (e.g. 'name' or 'Namn') - the stars were named "
                            "Star_1, Star_2, ... Please add names.")
        opt = {k: col(v) for k, v in _OPT_COLS.items()}
        for n, line in enumerate(lines[1:], start=2):
            tok = _split_line(line, delim)
            def get(i):
                return tok[i].strip() if (i is not None and i < len(tok)) else ''
            name = get(i_name) or f"Star_{n - 1}"
            if not get(i_ra) or not get(i_dec):
                problems.append(f"line {n} ({name}): RA or Dec missing - skipped")
                continue
            try:
                ra, dec = parse_coordinates(get(i_ra), get(i_dec))
            except Exception as e:
                problems.append(f"line {n} ({name}): coordinates '{get(i_ra)}' '{get(i_dec)}' not readable ({e})")
                continue
            t = {'name': name, 'ra': ra, 'dec': dec, 'source': src}
            t['type'] = get(opt['type']) or None
            t['period'] = _to_float(get(opt['period']))
            t['max'] = _mag_value(get(opt['max']))
            t['min'] = _mag_value(get(opt['min']), t['max'])
            t['epoch'] = _to_float(get(opt['epoch']))
            targets.append(t)
    else:
        for n, line in enumerate(lines, start=1):
            # delimiter per line (files may mix separators)
            ld = next((d for d in (',', ';', '\t') if d in line), None)
            tok = [t for t in _split_line(line, ld) if t != '']
            if ld is not None and len(tok) < 3:
                tok = line.split()
            elif ld is not None:
                # 'RR Lyr, 19 25 27.9, +42 47 03.7' -> split coordinate fields further
                tok = [tok[0]] + [x for t in tok[1:] for x in t.split()]
            name, ra, dec = _parse_tokens_no_header(tok)
            if not name and ra is None:
                problems.append(f"line {n}: could not read '{line.strip()[:60]}'")
                continue
            targets.append({'name': name or f"Star_{n}", 'ra': ra, 'dec': dec, 'source': src,
                            'type': None, 'period': None, 'max': None, 'min': None})
    return targets, problems


def resolve_missing_coordinates(targets):
    """Resolve targets given by name only (local GCVS list, SIMBAD, then AAVSO VSX). Returns problems."""
    problems = []
    for t in targets:
        if t.get('ra') is not None and t.get('dec') is not None:
            continue
        try:
            from python.names import from_name
            c = from_name(t['name'])                 # v4.30: local GCVS list first, then SIMBAD
            t['ra'], t['dec'] = float(c.ra.deg), float(c.dec.deg)
        except Exception:
            try:
                from python.calibration import resolve_auid
                r = resolve_auid(t['name'], verbose=False)
                if r:
                    t['ra'], t['dec'] = float(r['ra']), float(r['dec'])
            except Exception:
                pass
        if t.get('ra') is None:
            problems.append(f"'{t['name']}': name could not be resolved (no internet, or unknown name)")
    return problems


def save_target_list(path, targets):
    """Write targets as CSV (name, ra, dec in deg + optional columns) - re-readable."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['name', 'ra', 'dec', 'type', 'period', 'max', 'min', 'epoch', 'source'])
        for t in targets:
            w.writerow([t.get('name', ''), f"{t['ra']:.6f}" if t.get('ra') is not None else '',
                        f"{t['dec']:.6f}" if t.get('dec') is not None else '',
                        t.get('type') or '', '' if t.get('period') in (None, '') or not np.isfinite(_to_float(t.get('period')) or np.nan) else t['period'],
                        '' if _to_float(t.get('max')) is None else t['max'],
                        '' if _to_float(t.get('min')) is None else t['min'],
                        '' if _to_float(t.get('epoch')) is None else t['epoch'], t.get('source', '')])


# --------------------------------------------------------------------------
# b) Variables in the field from an online catalogue
# --------------------------------------------------------------------------
def image_field(header, shape=None):
    """(wcs, center_ra, center_dec, radius_arcmin, (ny, nx)) of an image, radius = largest
    centre-to-corner distance. Returns None without a celestial WCS."""
    import warnings
    from astropy.wcs import WCS
    from astropy.utils.exceptions import AstropyWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        w = WCS(header)
    if not w.has_celestial:
        return None
    w = w.celestial
    if shape is None:
        shape = (int(header.get('NAXIS2', 0)), int(header.get('NAXIS1', 0)))
    ny, nx = shape
    if not nx or not ny:
        return None
    centre = w.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
    corners = [w.pixel_to_world(x, y) for x, y in ((0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1))]
    radius = max(centre.separation(c).arcmin for c in corners)
    return w, float(centre.ra.deg), float(centre.dec.deg), float(radius), (ny, nx)


def image_footprints(fits_files):
    """
    (path, wcs, (ny, nx)) for every image with a celestial WCS - headers only, so
    this is fast also for many large frames. Images without WCS are returned with
    wcs=None (they are reported, not silently dropped).
    """
    import warnings
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.utils.exceptions import AstropyWarning
    out = []
    for p in fits_files:
        w, shape = None, None
        try:
            h = fits.getheader(p)
            if h.get('NAXIS', 0) < 2:
                with fits.open(p) as _hl:     # v5.0.1: file closed again (Windows file locks)
                    if len(_hl) > 1:
                        h = _hl[1].header.copy()
            shape = (int(h.get('NAXIS2', 0)), int(h.get('NAXIS1', 0)))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                ww = WCS(h)
            if ww.has_celestial and shape[0] and shape[1]:
                w = ww.celestial
        except Exception:
            pass
        out.append((p, w, shape))
    return out


def in_footprints(ra, dec, footprints, margin_px=20):
    """Boolean array: is (ra, dec) [deg; scalars or arrays] at least margin_px inside each
    image? Shape (n_images,) for scalars, (n_images, n_stars) for arrays. Images
    without WCS count as 'not inside'."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    scalar = np.ndim(ra) == 0
    c = SkyCoord(np.atleast_1d(ra) * u.deg, np.atleast_1d(dec) * u.deg)
    rows = []
    for _, w, shape in footprints:
        if w is None:
            rows.append(np.zeros(len(c), bool))
            continue
        x, y = w.world_to_pixel(c)
        ny, nx = shape
        rows.append(np.isfinite(x) & np.isfinite(y) & (x >= margin_px) & (y >= margin_px)
                    & (x <= nx - 1 - margin_px) & (y <= ny - 1 - margin_px))
    m = np.array(rows, bool).reshape(len(footprints), len(c))
    return m[:, 0] if scalar else m


def _first_col(row, colnames, candidates):
    for c in candidates:
        if c in colnames:
            v = row[c]
            if np.ma.is_masked(v):
                return None
            return v
    return None


def query_gcvs(ra_deg, dec_deg, radius_arcmin, vizier_cls=None):
    """
    GCVS stars around a position (VizieR catalogue B/gcvs/gcvs_cat).
    vizier_cls is only used for testing (simulated VizieR); normally astroquery's Vizier.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    if vizier_cls is None:
        from astroquery.vizier import Vizier as vizier_cls
    v = vizier_cls(catalog="B/gcvs/gcvs_cat", columns=['**', '_RAJ2000', '_DEJ2000'], row_limit=-1)
    res = v.query_region(SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg), radius=radius_arcmin * u.arcmin)
    out = []
    if not res or len(res) == 0:
        return out
    tab = res[0]
    cols = tab.colnames
    for row in tab:
        ra = _first_col(row, cols, ('_RAJ2000', 'RAJ2000'))
        dec = _first_col(row, cols, ('_DEJ2000', 'DEJ2000'))
        try:
            ra_f, dec_f = float(ra), float(dec)
        except (TypeError, ValueError):
            try:
                ra_f, dec_f = parse_coordinates(ra, dec)
            except Exception:
                continue
        name = _first_col(row, cols, ('GCVS', 'VarName', 'Name', 'Var'))
        out.append({'name': ' '.join(str(name).split()) if name is not None else f"GCVS_{ra_f:.4f}{dec_f:+.4f}",
                    'ra': ra_f, 'dec': dec_f,
                    'type': str(_first_col(row, cols, ('VarType', 'Type')) or '').strip(),
                    'max': _to_float(_first_col(row, cols, ('magMax', 'max', 'Max'))),
                    'min': _to_float(_first_col(row, cols, ('Min1', 'magMin', 'min', 'Min'))),
                    'period': _to_float(_first_col(row, cols, ('Period', 'Per'))),
                    'source': 'GCVS'})
    return out


def query_vsx(ra_deg, dec_deg, radius_arcmin, fetch=None):
    """
    AAVSO VSX stars around a position. Uses Calibra's VSX cache, but older cache
    files hold only names and coordinates (no type, range or period) - in that
    case VSX is queried again (and the complete answer cached). If the query
    fails, the names/coordinates from the old cache are returned.
    fetch: only for testing (replaces calibration.fetch_vsx_catalog).
    """
    from python import calibration as cal
    stars = cal.get_vsx_stars(ra_deg, dec_deg, radius_arcmin=radius_arcmin, verbose=False) or []
    complete = any(str(s.get('Type', '') or '').strip() or _to_float(s.get('max')) is not None for s in stars)
    if stars and not complete:
        try:
            fresh = (fetch or cal.fetch_vsx_catalog)(ra_deg, dec_deg, radius_arcmin, verbose=False)
        except Exception:
            fresh = []
        if fresh:
            stars = fresh
            try:
                cache = os.path.join(paths.CATALOG_CACHE_DIR, f"VSX_{ra_deg:.4f}_{dec_deg:.4f}_{radius_arcmin}.csv")
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                with open(cache, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=['id', 'ra_deg', 'dec_deg', 'Type', 'max', 'min', 'Period'])
                    w.writeheader()
                    for s in fresh:
                        w.writerow({k: s.get(k, np.nan) for k in w.fieldnames})
            except OSError:
                pass
    out = []
    for s in stars:
        out.append({'name': str(s.get('id', '')).strip(), 'ra': float(s['ra_deg']), 'dec': float(s['dec_deg']),
                    'type': str(s.get('Type', '') or '').strip(), 'max': _to_float(s.get('max')),
                    'min': _to_float(s.get('min')), 'period': _to_float(s.get('Period')), 'source': 'VSX'})
    return out


def query_field_variables(source, ra_deg, dec_deg, radius_arcmin):
    source = (source or 'VSX').upper()
    if source.startswith('GCVS'):
        # v4.26: local GCVS list (catalogue folder) first, online only without it
        try:
            from python import local_catalogs as lc
            local = lc.gcvs_in_field(ra_deg, dec_deg, radius_arcmin)
        except Exception as e:
            print(f"Local GCVS list not usable ({e}) - trying online.")
            local = None
        if local is not None:
            return local
        return query_gcvs(ra_deg, dec_deg, radius_arcmin)
    return query_vsx(ra_deg, dec_deg, radius_arcmin)


def filter_variables(stars, wcs=None, shape=None, edge_px=30, faintest_max=None, min_amplitude=None,
                     min_period=None, max_period=None, include_unknown_period=True,
                     type_filter="", exclude_constant=True):
    """
    Keep variables that are on the image (>= edge_px from the border), brighter
    at maximum than `faintest_max`, with amplitude (min - max) >= min_amplitude,
    period within [min_period, max_period] days (stars without a period are
    kept if include_unknown_period), and whose type contains one of the
    words in type_filter (e.g. "M SR L"; empty = all). CST (constant) is
    dropped when exclude_constant. Returns the filtered list (with 'x', 'y').
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    words = [w.upper() for w in re.split(r'[\s,;]+', type_filter or '') if w]
    out = []
    for s in stars:
        if s.get('ra') is None:
            continue
        typ = str(s.get('type') or '').upper()
        if exclude_constant and ('CST' in typ or typ in ('CONST', 'NSV')):
            continue
        if words and not any(w in typ for w in words):
            continue
        mx, mn = _to_float(s.get('max')), _to_float(s.get('min'))
        if faintest_max is not None and mx is not None and mx > faintest_max:
            continue
        if faintest_max is not None and mx is None and mn is not None and mn > faintest_max + 2:
            continue
        if min_amplitude is not None and min_amplitude > 0:
            if mx is None or mn is None or (mn - mx) < min_amplitude:
                continue
        per = _to_float(s.get('period'))
        if per is None or per <= 0:
            if not include_unknown_period and (min_period or max_period):
                continue
        else:
            if min_period is not None and per < min_period:
                continue
            if max_period is not None and per > max_period:
                continue
        if wcs is not None and shape is not None:
            x, y = wcs.world_to_pixel(SkyCoord(s['ra'] * u.deg, s['dec'] * u.deg))
            ny, nx = shape
            if not (np.isfinite(x) and np.isfinite(y)) or x < edge_px or y < edge_px or x > nx - 1 - edge_px or y > ny - 1 - edge_px:
                continue
            s = dict(s, x=float(x), y=float(y))
        out.append(s)
    return out


# --------------------------------------------------------------------------
# Automatic comparison stars from the reference catalogue
# --------------------------------------------------------------------------
def choose_catalog_comparisons(target, catalog, variables=(), wcs=None, shape=None, band='V',
                               n_comp=5, n_check=1, radius_arcmin=12.0, mag_window=1.5,
                               blend_arcsec=20.0, edge_px=40, bright_limit=8.0, footprints=None):
    """
    Pick comparison (and check) stars for one target from the photometric
    catalogue (list of dicts with ra_deg, dec_deg, V_mag, B_mag):
      * on the image, not a known variable, no other catalogue star within blend_arcsec,
      * within radius_arcmin of the target (widened up to 2x if necessary),
      * brightness close to the target's catalogue magnitude (window widened
        stepwise), never brighter than bright_limit (saturation),
      * closest in brightness first, then distance.
    Returns (comps, check, target_mag, target_bv) - comps/check in the ensemble
    format of run_multi_target_photometry ({'name','ra','dec','mag_std','bv_std'});
    target_mag/target_bv from the catalogue entry of the target (None if not found;
    target_mag then from the variable's catalogued range).
    footprints (from image_footprints, optional): comparison stars must then be
    on every image that contains the target (several nights with different
    pointings); if too few qualify, stars on >= 80 % of those images are allowed.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if not catalog:
        return [], None, None, None
    mag_key = 'B_mag' if str(band).upper() == 'B' else 'V_mag'
    cra = np.array([c['ra_deg'] for c in catalog], float)
    cdec = np.array([c['dec_deg'] for c in catalog], float)
    cm = np.array([c.get(mag_key, np.nan) for c in catalog], float)
    cbv = np.array([c.get('B_mag', np.nan) - c.get('V_mag', np.nan) for c in catalog], float)
    cc = SkyCoord(cra * u.deg, cdec * u.deg)
    tc = SkyCoord(target['ra'] * u.deg, target['dec'] * u.deg)
    dist = tc.separation(cc).arcmin

    # target brightness: catalogue match, else mean of the catalogue range
    target_mag, target_bv = None, None
    j = int(np.argmin(dist))
    if dist[j] * 60 < 3.0 and np.isfinite(cm[j]):
        target_mag = float(cm[j])
        if np.isfinite(cbv[j]):
            target_bv = float(cbv[j])
    else:
        mx, mn = _to_float(target.get('max')), _to_float(target.get('min'))
        if mx is not None and mn is not None:
            target_mag = 0.5 * (mx + mn)
        elif mx is not None:
            target_mag = mx + 0.5

    ok = np.isfinite(cm) & (dist * 60 > 3.0) & (cm > bright_limit)
    # isolated stars only
    idx, sep2, _ = cc.match_to_catalog_sky(cc, nthneighbor=2)
    ok &= sep2.arcsec > blend_arcsec
    # not a known variable
    if variables:
        vc = SkyCoord([v['ra'] for v in variables] * u.deg, [v['dec'] for v in variables] * u.deg)
        _, vsep, _ = cc.match_to_catalog_sky(vc)
        ok &= vsep.arcsec > 5.0
    # on the image
    if wcs is not None and shape is not None:
        px, py = wcs.world_to_pixel(cc)
        ny, nx = shape
        ok &= np.isfinite(px) & np.isfinite(py) & (px > edge_px) & (py > edge_px) & (px < nx - 1 - edge_px) & (py < ny - 1 - edge_px)
    ok &= np.isfinite(cbv)
    if footprints:
        t_in = in_footprints(target['ra'], target['dec'], footprints, margin_px=edge_px / 2)
        if np.any(t_in):
            fp_t = [f for f, k in zip(footprints, t_in) if k]
            frac = in_footprints(cra, cdec, fp_t, margin_px=edge_px).mean(axis=0)
            strict = ok & (frac >= 1.0)
            ok = strict if strict.sum() >= n_comp + n_check else ok & (frac >= 0.8)
    # stars already flagged as variable in the catalogue (VSX cross-match)
    ok &= np.array([not c.get('is_variable', False) and str(c.get('vsx_name', '') or '').strip().lower() in ('', 'nan', 'none')
                    for c in catalog], bool)

    need = n_comp + n_check
    chosen = np.array([], int)
    for r_fac in (1.0, 1.5, 2.0):
        for win in ((mag_window, mag_window * 1.7, mag_window * 2.7, 99.0) if target_mag is not None else (99.0,)):
            sel = ok & (dist <= radius_arcmin * r_fac)
            if target_mag is not None:
                sel &= np.abs(cm - target_mag) <= win
            cand = np.where(sel)[0]
            if len(cand) >= need or (r_fac == 2.0 and win == 99.0):
                dm = np.abs(cm[cand] - target_mag) if target_mag is not None else np.zeros(len(cand))
                score = dm + dist[cand] / radius_arcmin
                chosen = cand[np.argsort(score)][:need]
                break
        if len(chosen) >= need:
            break

    def as_ens(k):
        c = catalog[k]
        nm = str(c.get('auid') or c.get('cat_id') or c.get('id') or f"C{k}")
        return {'name': nm, 'ra': float(cra[k]), 'dec': float(cdec[k]), 'mag_std': float(cm[k]), 'bv_std': float(cbv[k])}

    comps = [as_ens(k) for k in chosen[:n_comp]]
    check = as_ens(chosen[n_comp]) if len(chosen) > n_comp else None
    return comps, check, target_mag, target_bv


# --------------------------------------------------------------------------
# Summary statistics and overview plot
# --------------------------------------------------------------------------
def summarize_light_curve(results):
    """Simple variability statistics of one light curve (OK points only)."""
    if not results:
        return None
    mode = results[0].get('lc_mode', 'calibrated')
    ok = [r for r in results if r.get('flag', 'OK') == 'OK']
    key = 'mag' if mode == 'calibrated' else 'flux'
    ekey = 'mag_err' if mode == 'calibrated' else 'flux_err'
    v = np.array([r.get(key, np.nan) for r in ok], float)
    e = np.array([r.get(ekey, np.nan) for r in ok], float)
    good = np.isfinite(v)
    v, e = v[good], e[good]
    if v.size == 0:
        return {'n': 0, 'mode': mode}
    st = {'n': int(v.size), 'n_total': len(results), 'mode': mode,
          'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)) if v.size > 1 else np.nan,
          'p2p': float(np.percentile(v, 95) - np.percentile(v, 5)) if v.size > 2 else np.nan,
          'med_err': float(np.nanmedian(e)) if np.any(np.isfinite(e)) else np.nan,
          't0': float(min(r['hjd'] for r in results)), 't1': float(max(r['hjd'] for r in results))}
    if v.size > 1 and np.isfinite(st['med_err']) and st['med_err'] > 0:
        w = 1.0 / np.where(np.isfinite(e) & (e > 0), e, st['med_err']) ** 2
        mu = np.sum(w * v) / np.sum(w)
        st['chi2_red'] = float(np.sum(w * (v - mu) ** 2) / (v.size - 1))
    else:
        st['chi2_red'] = np.nan
    if mode != 'calibrated' and st['mean'] > 0:
        st['std_mag'] = 1.0857 * st['std'] / st['mean']
    return st


def plot_overview(curves, output_path, title="Field variables"):
    """
    One small panel per target. curves: list of (name, results). Magnitudes are
    plotted inverted, fluxes normal; SUSPECT points in red.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    curves = [(n, r) for n, r in curves if r]
    if not curves:
        return None
    n = len(curves)
    ncol = 3 if n > 4 else (2 if n > 1 else 1)
    nrow = int(np.ceil(n / ncol))
    fig = Figure(figsize=(5.0 * ncol, 2.8 * nrow))
    FigureCanvasAgg(fig)
    t0 = int(min(min(x['hjd'] for x in r) for _, r in curves))
    for k, (name, res) in enumerate(curves):
        ax = fig.add_subplot(nrow, ncol, k + 1)
        mode = res[0].get('lc_mode', 'calibrated')
        key, ekey = ('mag', 'mag_err') if mode == 'calibrated' else ('flux', 'flux_err')
        t = np.array([x['hjd'] for x in res]) - t0
        y = np.array([x.get(key, np.nan) for x in res], float)
        e = np.array([x.get(ekey, np.nan) for x in res], float)
        e = np.where(np.isfinite(e), e, 0.0)
        bad = np.array([x.get('flag', 'OK') != 'OK' for x in res])
        ax.errorbar(t[~bad], y[~bad], yerr=e[~bad], fmt='o', ms=3, color='darkblue', ecolor='gray', capsize=1)
        if np.any(bad):
            ax.plot(t[bad], y[bad], 'o', ms=4, mfc='none', mec='red')
        if mode == 'calibrated':
            ax.invert_yaxis()
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        unit = {'calibrated': 'mag', 'relative': 'rel. flux', 'raw': res[0].get('flux_unit', 'ADU')}[mode]
        ax.set_ylabel(unit, fontsize=7)
        ax.set_xlabel(f"HJD - {t0}", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    return output_path
