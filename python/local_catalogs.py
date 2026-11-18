"""
Local copies of photometric catalogues (v4.26).

The user points Calibra to ONE folder (Settings -> Astro & Catalog -> Local
catalogue folder). Its subfolders are recognised automatically:

    <root>/ATLAS...  or  .../refcat...      -> ATLAS-RefCat2
    <root>/Gaia...                          -> Gaia DR3
    <root>/APASS...                         -> APASS DR9
    <root>/VSX...                           -> AAVSO VSX (optional)
    <root>/Landolt...                       -> Landolt standards (optional)

(by folder name; folders with other names are recognised by the column names
of their files). The choice is stored in <user folder>/config/local_catalogs.json
and used again at the next start.

Two file layouts are understood:

  * tiles  - ATLAS-RefCat2 as distributed by MAST/STScI: 1x1 degree files named
             'RRR+DD' / 'RRR-DD' (extension .rc2, .csv, .txt or none, optionally .gz),
             usually in magnitude subfolders such as 00_m_16, 16_m_17, 17_m_18.
             Columns without header in the RefCat2 order (ra, dec, plx, ..., g, dg,
             ..., r, dr, ...); integer-scaled values (RA/Dec in 1e-8 deg, magnitudes
             in millimag) are recognised and converted. Files with a header are
             read by column name.
  * tables - any CSV / TSV / whitespace files (optionally .gz) with a header
             line, e.g. exports from the Gaia archive, VizieR, AAVSO APASS. Lines
             starting with '#' are skipped. The sky area of every file is indexed
             once (<user folder>/cache/catalogs/local_index_<cat>.json), so that a
             query only reads the files that can contain the field.

query() returns None when the catalogue is not available locally or does not
cover the field. v4.46: query_best() then tries the local ATLAS-RefCat2 copy as a
substitute (option prefer_offline, default on) before the caller goes online.
"""
import csv
import glob
import gzip
import io
import json
import os
from python import paths
import re
import time
import numpy as np

DATA_EXT = ('.csv', '.txt', '.dat', '.tsv', '.rc2', '.cat', '.tab', '.ecsv', '.psv')

CATALOGS = ('ATLAS', 'GAIA_DR3', 'APASS', 'VSX', 'LANDOLT', 'GCVS')
_NAME_HINTS = {
    'ATLAS': ('atlas', 'refcat'),
    'GAIA_DR3': ('gaia',),
    'APASS': ('apass',),
    'VSX': ('vsx',),
    'LANDOLT': ('landolt',),
    'GCVS': ('gcvs',),
}

# column aliases (lower case) -------------------------------------------------
_RA = ('ra', 'ra_icrs', 'raj2000', '_raj2000', 'radeg', 'ra_deg', 'ra(deg)', 'ra_j2000', 'alpha', 'ra2000')
_DEC = ('dec', 'de_icrs', 'dej2000', '_dej2000', 'decdeg', 'dec_deg', 'dec(deg)', 'dec_j2000', 'de', 'delta', 'dec2000')
_COLS = {
    'g_sdss': ('g', 'gmag', 'g_mag', 'sloan_g', 'g_psf'),
    'r_sdss': ('r', 'rmag', 'r_mag', 'sloan_r', 'r_psf'),
    'G': ('phot_g_mean_mag', 'gmag', 'g', 'gaia', 'g_mag'),
    'BP': ('phot_bp_mean_mag', 'bpmag', 'bp', 'bp_mag'),
    'RP': ('phot_rp_mean_mag', 'rpmag', 'rp', 'rp_mag'),
    'V': ('johnson_v', 'vmag', 'v', 'v_mag', 'mag_v'),
    'B': ('johnson_b', 'bmag', 'b', 'b_mag', 'mag_b'),
    'BV': ('b-v', 'b_v', 'bv'),
    'id': ('source_id', 'source', 'objid', 'designation', 'name', 'star', 'recno', 'id', 'auid', 'oid'),
    'type': ('type', 'vartype', 'var_type'),
    'max': ('max', 'magmax', 'mag_max'),
    'min': ('min', 'magmin', 'mag_min'),
    'period': ('period', 'per'),
}
# which quantities each catalogue needs
_NEEDS = {
    'ATLAS': ('g_sdss', 'r_sdss'),
    'GAIA_DR3': ('G', 'BP', 'RP'),
    'APASS': ('V', 'B'),
    'VSX': ('type',),
    'LANDOLT': ('V',),
}
# RefCat2 column order (files without header)
REFCAT2_COLS = ['ra', 'dec', 'plx', 'dplx', 'pmra', 'dpmra', 'pmdec', 'dpmdec', 'gaia', 'dgaia', 'bp', 'dbp',
                'rp', 'drp', 'teff', 'agaia', 'dupvar', 'ag', 'rp1', 'r1', 'r10', 'g', 'dg', 'gchi', 'gcontrib',
                'r', 'dr', 'rchi', 'rcontrib', 'i', 'di', 'ichi', 'icontrib', 'z', 'dz', 'zchi', 'zcontrib',
                'nstat', 'j', 'dj', 'h', 'dh', 'k', 'dk']
_TILE_RE = re.compile(r'^(\d{3})([+-]\d{2})(\.[a-z0-9]+)?(\.gz)?$', re.IGNORECASE)
_MAGDIR_RE = re.compile(r'(\d+)_m_(\d+)', re.IGNORECASE)

_STATE = {'root': None, 'catalogs': {}, 'use_local': True, 'allow_online': True, 'apass_auto': False,
          'prefer_offline': True, 'loaded': False}
# v4.46: catalogue used instead when the chosen one is not available locally for a field
# (ATLAS-RefCat2 contains Gaia G/BP/RP and gives V/B from Pan-STARRS g/r)
OFFLINE_SUBSTITUTE = 'ATLAS'


# ------------------------------------------------------------------ configuration
def _load_config():
    if _STATE['loaded']:
        return
    _STATE['loaded'] = True
    try:
        with open(paths.LOCAL_CATALOGS_FILE, encoding='utf-8') as f:
            cfg = json.load(f)
        _STATE.update(root=cfg.get('root'), catalogs=cfg.get('catalogs', {}),
                      use_local=cfg.get('use_local', True), allow_online=cfg.get('allow_online', True),
                      apass_auto=cfg.get('apass_auto', False), prefer_offline=cfg.get('prefer_offline', True))
    except (OSError, ValueError):
        pass


def _save_config():
    try:
        os.makedirs(os.path.dirname(paths.LOCAL_CATALOGS_FILE), exist_ok=True)
        with open(paths.LOCAL_CATALOGS_FILE, 'w', encoding='utf-8') as f:
            json.dump({'root': _STATE['root'], 'catalogs': _STATE['catalogs'], 'use_local': _STATE['use_local'],
                       'allow_online': _STATE['allow_online'], 'apass_auto': _STATE.get('apass_auto', False),
                       'prefer_offline': _STATE.get('prefer_offline', True)},
                      f, indent=2)
    except OSError as e:
        print(f"Local catalogues: could not save {paths.LOCAL_CATALOGS_FILE}: {e}")


def get_config():
    _load_config()
    return dict(_STATE)


def set_options(use_local=None, allow_online=None, apass_auto=None, prefer_offline=None):
    _load_config()
    if prefer_offline is not None:
        _STATE['prefer_offline'] = bool(prefer_offline)
    if apass_auto is not None:
        _STATE['apass_auto'] = bool(apass_auto)
    if use_local is not None:
        _STATE['use_local'] = bool(use_local)
    if allow_online is not None:
        _STATE['allow_online'] = bool(allow_online)
    _save_config()


def apass_auto():
    """v4.32: download APASS DR10 automatically for fields missing locally (default off)."""
    _load_config()
    return bool(_STATE.get('apass_auto', False))


def register_folder(cat, folder):
    """v4.32: (re)describe one catalogue folder without scanning all others (e.g. after an
    APASS DR10 download into a new folder) and save the configuration."""
    _load_config()
    info = _describe(folder)
    if info:
        _STATE['catalogs'][cat] = info
        _save_config()
    return info


def online_allowed():
    _load_config()
    return _STATE['allow_online']


def prefer_offline():
    """v4.46: offline first (default on). If the chosen catalogue is not available locally for
    a field, the local ATLAS-RefCat2 copy is used BEFORE any online query (VizieR is tried
    only if RefCat2 does not cover the field either); Gaia source IDs, which are only labels,
    are not fetched online."""
    _load_config()
    return bool(_STATE['use_local'] and _STATE.get('prefer_offline', True))


def has_local(catalog_name):
    """True if a local copy of this catalogue is configured (field coverage not checked)."""
    _load_config()
    info = _STATE['catalogs'].get(normalize_name(catalog_name) or '')
    return bool(_STATE['use_local'] and info and not info.get('unreadable')
                and os.path.isdir(info.get('path', '')))


def query_best(catalog_name, ra, dec, radius_arcmin, verbose=False, substitute=True):
    """
    v4.46: offline look-up with substitution. Returns (rows, used_catalog):
    rows of the chosen catalogue's local copy if it covers the field, otherwise (if
    substitute and the chosen catalogue is not RefCat2 itself) rows of the local
    ATLAS-RefCat2 copy; (None, None) if neither is available locally.
    """
    cat = normalize_name(catalog_name)
    rows = query(catalog_name, ra, dec, radius_arcmin, verbose=verbose)
    if rows is not None:
        return rows, cat
    if substitute and cat in ('GAIA_DR3', 'APASS', 'LANDOLT') and cat != OFFLINE_SUBSTITUTE:
        rows = query(OFFLINE_SUBSTITUTE, ra, dec, radius_arcmin, verbose=verbose)
        if rows is not None:
            if verbose:
                print(f"{cat} is not available locally for this field - using the local "
                      f"{OFFLINE_SUBSTITUTE}-RefCat2 copy instead (offline first).")
            return rows, OFFLINE_SUBSTITUTE
    return None, None


def set_root(path, rescan=True):
    """Set the catalogue folder, detect the catalogues in its subfolders and save the
    result for the next start. Returns the detection result {cat: info}."""
    _load_config()
    _STATE['root'] = os.path.abspath(path) if path else None
    if rescan:
        _STATE['catalogs'] = detect_catalogs(_STATE['root']) if _STATE['root'] else {}
    _save_config()
    return _STATE['catalogs']


def normalize_name(catalog_name):
    c = str(catalog_name or '').upper()
    if 'ATLAS' in c or 'REFCAT' in c:
        return 'ATLAS'
    if 'GAIA' in c:
        return 'GAIA_DR3'
    if 'APASS' in c:
        return 'APASS'
    if 'VSX' in c:
        return 'VSX'
    if 'LANDOLT' in c:
        return 'LANDOLT'
    return None


# ------------------------------------------------------------------ file helpers
def _open_text(path):
    if path.lower().endswith('.gz'):
        return io.TextIOWrapper(gzip.open(path, 'rb'), encoding='utf-8', errors='replace')
    return open(path, encoding='utf-8', errors='replace')


def _is_data_file(name):
    n = name.lower()
    if n.startswith('.') or n.endswith(('.json', '.md', '.pdf', '.html', '.py', '.readme')):
        return False
    if n.endswith('.gz'):
        n = n[:-3]
    return n.endswith(DATA_EXT) or bool(_TILE_RE.match(name))


def _data_files(folder, max_depth=3):
    out = []
    base = folder.rstrip(os.sep).count(os.sep)
    for dp, dn, fn in os.walk(folder):
        if dp.count(os.sep) - base >= max_depth:
            dn[:] = []
        dn[:] = [d for d in dn if not d.startswith('.')]
        out += [os.path.join(dp, f) for f in fn if _is_data_file(f)]
    return sorted(out)


def _head_lines(path, n=3):
    lines = []
    try:
        with _open_text(path) as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                lines.append(line.rstrip('\n\r'))
                if len(lines) >= n:
                    break
    except OSError:
        pass
    return lines


def _delimiter(line):
    for d in (',', '\t', ';', '|'):
        if d in line:
            return d
    return None


def _split(line, delim):
    if delim is None:
        return line.split()
    return [t.strip().strip('"') for t in next(csv.reader([line], delimiter=delim))]


def _is_number(t):
    try:
        float(t)
        return True
    except ValueError:
        return False


def _norm_col(c):
    """Column name for matching: lower case, quotes removed, a trailing '(...)' dropped -
    e.g. APASS DR10 'Johnson_V (V)' -> 'johnson_v', 'raerr(")' -> 'raerr' (v4.31)."""
    c = str(c).strip().strip('"\'').lower()
    c2 = re.sub(r'\s*\(.*\)\s*$', '', c)
    return c2 or c


def _find(cols_lower, aliases):
    for a in aliases:                       # exact names first (e.g. 'ra(deg)')
        if a in cols_lower:
            return cols_lower.index(a)
    norm = [_norm_col(c) for c in cols_lower]
    for a in aliases:
        if _norm_col(a) in norm:
            return norm.index(_norm_col(a))
    return None


def _file_layout(path):
    """('tiles'|'table', header list or None, delimiter)."""
    lines = _head_lines(path, 2)
    if not lines:
        return None, None, None
    delim = _delimiter(lines[0])
    toks = [t for t in _split(lines[0], delim) if t != '']
    # v4.45: a line is a header only if most of its fields are not numbers. Real RefCat2
    # tiles contain hexadecimal flag fields such as '2d' or '0d' (gcontrib, rcontrib ...);
    # before, such a first data line was taken as the header and the whole file was lost.
    n_num = sum(1 for t in toks if _is_number(t))
    header = None if (not toks or n_num >= 0.5 * len(toks)) else [t.strip().lower() for t in _split(lines[0], delim)]
    name = os.path.basename(path)
    if _TILE_RE.match(name):
        return 'tiles', header, delim
    return 'table', header, delim


def _guess_catalog_from_header(header):
    if not header:
        return None
    h = set(header) | {_norm_col(c) for c in header}
    if {'phot_g_mean_mag', 'phot_bp_mean_mag'} & h or ({'bpmag', 'rpmag'} <= h):
        return 'GAIA_DR3'
    if {'johnson_v', 'johnson_b'} & h or ({'vmag', 'bmag'} <= h and 'sloan_g' in h):
        return 'APASS'
    if {'gcontrib', 'rcontrib'} & h or ({'g', 'r', 'dg', 'dr'} <= h):
        return 'ATLAS'
    if ('type' in h or 'vartype' in h) and ({'max', 'min'} & h or 'period' in h):
        return 'VSX'
    if {'vmag', 'b-v'} <= h or {'v', 'b-v'} <= h:
        return 'LANDOLT'
    return None


# ------------------------------------------------------------------ detection
def detect_catalogs(root):
    """Find the catalogue subfolders of root. Returns {cat: {'path', 'layout', 'n_files',
    'mag_dirs' (tiles)}}; a catalogue not found is absent."""
    found = {}
    if not root or not os.path.isdir(root):
        return found
    # v4.26: GCVS lists (e.g. GCVS_selected.csv from the SAAF variable-star section) - any data
    # file with 'gcvs' in its name, in the folder itself or in a 'GCVS' subfolder
    gcvs_files = [os.path.join(root, f) for f in sorted(os.listdir(root))
                  if os.path.isfile(os.path.join(root, f)) and 'gcvs' in f.lower() and _is_data_file(f)]
    subs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('.'))
    for d in subs:
        if 'gcvs' in d.lower():
            gcvs_files += _data_files(os.path.join(root, d), max_depth=2)
    if gcvs_files:
        found['GCVS'] = {'path': os.path.dirname(gcvs_files[0]), 'layout': 'list', 'n_files': len(gcvs_files),
                         'files': gcvs_files}
    unassigned = []
    for d in subs:
        dl = d.lower()
        if 'gcvs' in dl:
            continue
        cat = next((c for c, hints in _NAME_HINTS.items() if any(h in dl for h in hints)), None)
        if cat and cat not in found:
            info = _describe(os.path.join(root, d))
            if info:
                found[cat] = info
                continue
        unassigned.append(d)
    for d in unassigned:                     # recognise by content
        p = os.path.join(root, d)
        files = _data_files(p, max_depth=3)[:5]
        for f in files:
            layout, header, _ = _file_layout(f)
            cat = 'ATLAS' if (layout == 'tiles' and not header) else _guess_catalog_from_header(header)
            if cat and cat not in found:
                info = _describe(p)
                if info:
                    found[cat] = info
                break
    return found


def _describe(folder):
    files = _data_files(folder)
    if not files:
        return None
    n_tiles = sum(1 for f in files if _TILE_RE.match(os.path.basename(f)))
    layout = 'tiles' if n_tiles > 0.5 * len(files) else 'table'
    info = {'path': folder, 'layout': layout, 'n_files': len(files)}
    if layout == 'tiles':
        tiles = [f for f in files if _TILE_RE.match(os.path.basename(f))]
        info['mag_dirs'] = sorted({os.path.relpath(os.path.dirname(f), folder) for f in tiles})
        exts = sorted({os.path.basename(f)[6:] for f in tiles})          # '.rc2', '', '.csv.gz' ...
        info['tile_ext'] = exts
    return info


def summary_lines():
    """Human-readable list of the detected catalogues (for the GUI)."""
    _load_config()
    names = {'ATLAS': 'ATLAS-RefCat2', 'GAIA_DR3': 'Gaia DR3', 'APASS': 'APASS (DR9/DR10)', 'VSX': 'AAVSO VSX',
             'LANDOLT': 'Landolt', 'GCVS': 'GCVS list'}
    out = []
    for cat in CATALOGS:
        info = _STATE['catalogs'].get(cat)
        if info:
            rel = os.path.relpath(info['path'], _STATE['root']) if _STATE['root'] else info['path']
            if info['layout'] == 'list':
                rel = ", ".join(os.path.relpath(f, _STATE['root']) for f in info.get('files', [])[:3])
                extra = "used by Field Variables and for star names"
            else:
                extra = (f"{info['n_files']} tile files" + (f" in {len(info.get('mag_dirs', []))} magnitude folders"
                                                             if len(info.get('mag_dirs', [])) > 1 else "")
                         if info['layout'] == 'tiles' else f"{info['n_files']} table file(s)")
            if cat == 'LANDOLT' and info['layout'] == 'table':     # v5.0.1: CDS tables
                try:
                    n_l = len(landolt_stars(info['path']))
                except Exception:
                    n_l = 0
                if n_l:
                    extra = f"{n_l} standard stars from the CDS tables"
            if info.get('unreadable') and not (cat == 'LANDOLT' and 'standard stars' in extra):     # v5.0.1
                extra += " - FORMAT NOT READABLE, not used"
            out.append(f"{names[cat]}: {rel}  ({extra})")
        else:
            out.append(f"{names[cat]}: not found locally" + (" - online" if _STATE['allow_online'] else ""))
    return out


# ------------------------------------------------------------------ reading
def _tile_names(ra, dec, radius_deg):
    """'RRR+DD' names of the 1x1 deg tiles touched by the circle."""
    names = set()
    d0, d1 = max(-90.0, dec - radius_deg), min(89.999, dec + radius_deg)
    cosd = max(np.cos(np.radians(min(89.0, max(abs(d0), abs(d1))))), 1e-3)   # worst case: largest |dec|
    dra = radius_deg / cosd
    if max(abs(d0), abs(d1)) > 89.0 or dra >= 180:
        ras = range(0, 360)
    else:
        ras = {int(np.floor(r)) % 360 for r in np.arange(ra - dra, ra + dra + 1.0, 1.0)}
        ras.add(int(np.floor((ra + dra) % 360)))
    for dd in range(int(np.floor(d0)), int(np.floor(d1)) + 1):
        for rr in ras:
            names.add(f"{rr:03d}{dd:+03d}")
    return names


def _rows_from_file(path, header, delim, ra, dec, radius_deg, fixed_cols=None):
    """Read one file and keep rows within radius_deg of (ra, dec). Returns list of dicts
    (lower-case column name -> text)."""
    cols = header or fixed_cols
    if not cols:
        return []
    i_ra, i_dec = _find(cols, _RA), _find(cols, _DEC)
    if i_ra is None or i_dec is None:
        return []
    out = []
    cosd = np.cos(np.radians(dec))
    skip_header = header is not None
    scale = None
    # v4.45: RefCat2 tiles store RA/Dec in units of 1e-8 deg. The scale is taken from the
    # tile name (RRR+DD) when there is one, so that a first star very close to RA 0 / Dec 0
    # cannot select the wrong scale; otherwise from the first data line as before.
    m_tile = _TILE_RE.match(os.path.basename(path))
    tile_ra = float(m_tile.group(1)) if m_tile else None
    with _open_text(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            if skip_header:
                skip_header = False
                continue
            t = _split(line.rstrip('\n\r'), delim)
            try:
                r_, d_ = float(t[i_ra]), float(t[i_dec])
            except (ValueError, IndexError):
                continue
            if scale is None:
                if tile_ra is not None and header is None:
                    scale = 1e-8 if abs(r_ * 1e-8 - tile_ra - 0.5) < abs(r_ - tile_ra - 0.5) else 1.0
                else:
                    scale = 1e-8 if (abs(r_) > 360 or abs(d_) > 90) else 1.0
            r_ *= scale; d_ *= scale
            if abs(d_ - dec) > radius_deg:
                continue
            dr = ((r_ - ra + 180.0) % 360.0 - 180.0) * cosd
            if dr * dr + (d_ - dec) ** 2 > radius_deg ** 2:
                continue
            row = dict(zip(cols, t))
            row['_ra'], row['_dec'] = r_, d_
            out.append(row)
    return out


def _index_file(cat):
    return os.path.join(paths.CATALOG_CACHE_DIR, f"local_index_{cat}.json")


def _table_index(cat, info, progress=None):
    """{path: {'mtime','size','ra0','ra1','dec0','dec1','n'}} - built once, updated for
    new or changed files."""
    try:
        with open(_index_file(cat), encoding='utf-8') as f:
            idx = json.load(f)
    except (OSError, ValueError):
        idx = {}
    files = _data_files(info['path'])
    changed = False
    for k, p in enumerate(files):
        st = os.stat(p)
        ent = idx.get(p)
        if ent and ent.get('mtime') == st.st_mtime and ent.get('size') == st.st_size:
            continue
        _, header, delim = _file_layout(p)
        i_ra = _find(header or [], _RA); i_dec = _find(header or [], _DEC)
        ra0 = dec0 = 1e9; ra1 = dec1 = -1e9; n = 0
        if i_ra is not None and i_dec is not None:
            with _open_text(p) as f:
                first = True
                for line in f:
                    if not line.strip() or line.lstrip().startswith('#'):
                        continue
                    if first:
                        first = False
                        continue
                    t = _split(line.rstrip('\n\r'), delim)
                    try:
                        r_, d_ = float(t[i_ra]), float(t[i_dec])
                    except (ValueError, IndexError):
                        continue
                    ra0 = min(ra0, r_); ra1 = max(ra1, r_); dec0 = min(dec0, d_); dec1 = max(dec1, d_); n += 1
        idx[p] = {'mtime': st.st_mtime, 'size': st.st_size, 'ra0': ra0, 'ra1': ra1, 'dec0': dec0, 'dec1': dec1, 'n': n}
        changed = True
        if progress:
            progress(k + 1, len(files), p)
    for p in [p for p in idx if p not in files]:
        del idx[p]; changed = True
    if changed:
        try:
            os.makedirs(paths.CATALOG_CACHE_DIR, exist_ok=True)
            with open(_index_file(cat), 'w', encoding='utf-8') as f:
                json.dump(idx, f)
        except OSError:
            pass
    return idx


def _mark_unreadable(cat, info, idx):
    """v5.0.1: a table catalogue whose files give no stars at all (e.g. the binary Cartes du
    Ciel Gaia catalogue, 'CDCSTAR1' .dat files) is marked 'unreadable' in the configuration
    and skipped from then on - before, every look-up checked all its files again (~30 s for
    73000 files) for nothing. Choosing/rescanning the catalogue folder clears the mark.
    Returns True if the catalogue is unreadable."""
    if not idx or any(e.get('n', 0) > 0 for e in idx.values()):
        return False
    info['unreadable'] = True
    print(f"Local {cat} copy in {info.get('path')}: none of its {len(idx)} files could be read "
          "(format not recognised, e.g. a binary catalogue of another program) - not used. "
          "Other local catalogues are used instead.")
    _save_config()
    return True


def build_indexes(progress=None):
    """Index all table-type catalogues (run once after choosing the folder)."""
    _load_config()
    for cat, info in _STATE['catalogs'].items():
        if info.get('layout') == 'table':
            idx = _table_index(cat, info, progress=(lambda k, n, p, c=cat: progress(c, k, n, p)) if progress else None)
            _mark_unreadable(cat, info, idx)


# ------------------------------------------------------------------ Landolt (CDS tables)
_LANDOLT_CACHE = {}


def _readme_year(folder):
    """Publication year from the first line of a CDS ReadMe in folder ('... (Landolt, 2009)'), else 0."""
    for f in glob.glob(os.path.join(folder, 'ReadMe*')):
        try:
            with open(f, encoding='latin-1') as fh:
                m = re.search(r'\(\D*(\d{4})\)\s*$', fh.readline().rstrip())
            if m:
                return int(m.group(1))
        except OSError:
            pass
    m = re.search(r'(19|20)\d{2}', os.path.basename(folder))
    return int(m.group(0)) if m else 0


def _cds_pipe_table(path):
    """Stars of a CDS/VizieR download ('#Table: ...', columns separated by '|', header
    rows above a '---|---' line). [(name, ra, dec, V, B-V)] or None if not this format or
    without V and B-V columns."""
    from python.field_variables import parse_coordinates
    try:
        with open(path, encoding='latin-1') as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    if not lines or not lines[0].startswith('#Table:'):
        return None
    sep = next((k for k, l in enumerate(lines) if '|' in l and set(l.strip()) <= set('-|')), None)
    if sep is None:
        return None
    head = [l for l in lines[1:sep] if '|' in l]
    ncol = len(lines[sep].split('|'))
    names = [' '.join(h.split('|')[c].strip() for h in head if len(h.split('|')) > c).strip() for c in range(ncol)]
    def col(test):
        return next((c for c, n in enumerate(names) if test(n.replace(' ', '').lower())), None)
    c_pos = col(lambda n: 'ah' in n and ('ed' in n or 'de' in n))
    c_v = col(lambda n: n.replace('<', '').replace('>', '') == 'vmag')
    c_bv = col(lambda n: n.replace('<', '').replace('>', '') == 'b-v')
    if c_pos is None or c_v is None or c_bv is None:
        return None
    out = []
    for l in lines[sep + 1:]:
        t = l.split('|')
        if len(t) < ncol or l.startswith('#'):
            continue
        p = t[c_pos].split()
        try:
            ra, dec = parse_coordinates(' '.join(p[:3]), ' '.join(p[3:]))
            v, bv = float(t[c_v]), float(t[c_bv])
        except (ValueError, IndexError):
            continue
        out.append((t[0].strip(), ra, dec, v, bv))
    return out


def _cds_raw_table(path):
    """Stars of a raw CDS table (fixed width, e.g. table2.dat) described in a ReadMe in the
    same folder (astropy 'cds' reader). [(name, ra, dec, V, B-V)] or None."""
    readme = next(iter(glob.glob(os.path.join(os.path.dirname(path), 'ReadMe*'))), None)
    if readme is None:
        return None
    try:
        from astropy.io import ascii as _ascii
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tab = _ascii.read(path, format='cds', readme=readme)
    except Exception:
        return None
    cols = {c.lower().replace('<', '').replace('>', ''): c for c in tab.colnames}
    need = ('rah', 'ram', 'ras', 'ded', 'dem', 'des', 'vmag', 'b-v')
    if not all(k in cols for k in need):
        return None
    name_c = tab.colnames[0]
    sign_c = cols.get('de-')
    out = []
    for row in tab:
        try:
            ra = 15.0 * (float(row[cols['rah']]) + float(row[cols['ram']]) / 60 + float(row[cols['ras']]) / 3600)
            dec = float(row[cols['ded']]) + float(row[cols['dem']]) / 60 + float(row[cols['des']]) / 3600
            if sign_c is not None and str(row[sign_c]).strip() == '-':
                dec = -dec
            v, bv = float(row[cols['vmag']]), float(row[cols['b-v']])
        except (ValueError, TypeError):
            continue
        if np.isfinite(v) and np.isfinite(bv):
            out.append((str(row[name_c]).strip(), ra, dec, v, bv))
    return out


def landolt_stars(folder):
    """
    v5.0.1: Landolt standard stars from the original CDS tables in folder (and subfolders):
    VizieR/CDS downloads ('#Table:' + '|' columns) and raw tables with their ReadMe
    (e.g. 1992/table2.dat, 2009/table2.dat, 2013/...table2.dat.txt). Only tables with V and
    B-V; positions J2000. The same star in several tables (1992 / 2009 / 2013, raw + download)
    is kept once, from the newest publication. 1992 positions can be up to ~30" off (rounded,
    proper motion) and some stars were renamed (92 311 -> 92 312): same star = same name
    (without 'BD'/'HD') within 60", or V within 0.03 mag within 30", or any two within 2".
    Returns a list of dicts (name, ra, dec, V, BV, year, file); [] if nothing readable.
    """
    files = sorted(p for p in glob.glob(os.path.join(folder, '**', '*'), recursive=True)
                   if os.path.isfile(p) and not os.path.basename(p).lower().startswith('readme')
                   and not p.lower().endswith('.pdf'))
    key = (folder, tuple((p, os.path.getmtime(p)) for p in files))
    if _LANDOLT_CACHE.get('key') == key:
        return _LANDOLT_CACHE['stars']
    found = []
    for p in files:
        got = _cds_pipe_table(p)
        if got is None:
            got = _cds_raw_table(p)
        if got:
            yr = _readme_year(os.path.dirname(p))
            found += [{'name': n, 'ra': r, 'dec': d, 'V': v, 'BV': bv, 'year': yr, 'file': p}
                      for n, r, d, v, bv in got]
    found.sort(key=lambda s: -s['year'])
    kept = []
    if found:
        def nkey(n):                   # 'BD -12 2918' = '-12 2918', 'HD 121968' = '121968', case
            return re.sub(r'^(bd|hd)', '', re.sub(r'\s+', '', n).lower())
        for s in found:
            dup = False
            cosd = np.cos(np.radians(s['dec']))
            for k in kept:
                if abs(k['dec'] - s['dec']) > 0.02:
                    continue
                sep = np.hypot((k['ra'] - s['ra']) * cosd, k['dec'] - s['dec']) * 3600.0
                if (sep < 2.0 or (sep < 60.0 and nkey(k['name']) == nkey(s['name']))
                        or (sep < 30.0 and abs(k['V'] - s['V']) < 0.03)):
                    dup = True
                    break
            if not dup:
                kept.append(s)
    _LANDOLT_CACHE.update(key=key, stars=kept)
    return kept


def _landolt_query(info, ra, dec, rdeg):
    """Rows in the format of query() from the Landolt CDS tables, None if the folder holds
    no such table or no Landolt star lies in the field (-> substitute / online)."""
    stars = landolt_stars(info['path'])
    if not stars:
        return None
    if info.get('unreadable'):         # marked by an earlier version that could not read CDS tables
        info.pop('unreadable', None)
        _save_config()
    d1 = np.radians(dec)
    rows = []
    for s in stars:
        d2 = np.radians(s['dec'])
        cosang = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(np.radians(s['ra'] - ra))
        if np.degrees(np.arccos(np.clip(cosang, -1, 1))) <= rdeg:
            rows.append({'name': s['name'], 'vmag': f"{s['V']:.3f}", 'b-v': f"{s['BV']:.3f}",
                         'ref': f"Landolt {s['year']}" if s['year'] else 'Landolt', '_ra': s['ra'], '_dec': s['dec']})
    return rows or None


def query(catalog_name, ra, dec, radius_arcmin, verbose=False):
    """
    Raw rows of a local catalogue within radius_arcmin of (ra, dec), or None if the
    catalogue is not available locally or the field is not covered (-> go online).
    Each row is a dict of lower-case column names -> text plus '_ra', '_dec' (deg).
    """
    _load_config()
    cat = normalize_name(catalog_name)
    if not _STATE['use_local'] or cat is None:
        return None
    info = _STATE['catalogs'].get(cat)
    if not info or not os.path.isdir(info.get('path', '')):
        return None
    rdeg = radius_arcmin / 60.0
    t0 = time.time()
    rows = []
    if info['layout'] == 'tiles':
        want = _tile_names(ra, dec, rdeg)
        # magnitude folders such as 18_m_19 hold only stars fainter than 18 - not needed;
        # tile paths are built directly (no directory walk: the full RefCat2 has ~300000 files)
        files = []
        for md in info.get('mag_dirs', ['.']):
            m = _MAGDIR_RE.search(md)
            if m is not None and int(m.group(1)) >= 18:
                continue
            for name in want:
                for ext in info.get('tile_ext', ['', '.rc2']):
                    p_ = os.path.join(info['path'], md, name + ext)
                    if os.path.exists(p_):
                        files.append(p_)
                        break
        if not files:
            return None                    # field not covered locally
        n_dirs = max(1, sum(1 for md in info.get('mag_dirs', ['.'])
                            if not (_MAGDIR_RE.search(md) and int(_MAGDIR_RE.search(md).group(1)) >= 18)))
        if len(files) < len(want) * n_dirs and verbose:
            print(f"Local {cat}: {len(want) * n_dirs - len(files)} of {len(want) * n_dirs} tile files of this field "
                  "are not in the local copy (stars there are missing).")
        unread = []
        for f in files:
            _, header, delim = _file_layout(f)
            got = _rows_from_file(f, header, delim, ra, dec, rdeg, fixed_cols=REFCAT2_COLS)
            if not got and os.path.getsize(f) > 0 and not _rows_from_file(f, header, delim, ra, dec, 180.0,
                                                                           fixed_cols=REFCAT2_COLS)[:1]:
                unread.append(os.path.relpath(f, info['path']))
            rows += got
        if unread:                     # v4.45: tile files that exist but give no stars at all
            print(f"Local {cat}: {len(unread)} tile file(s) could not be read (format not recognised): "
                  + ", ".join(unread[:5]) + (" ..." if len(unread) > 5 else ""))
    else:
        if cat == 'LANDOLT':               # v5.0.1: original CDS tables (fixed width / '|' downloads)
            if landolt_stars(info['path']):
                rows = _landolt_query(info, ra, dec, rdeg)
                if verbose:
                    print(f"Local {cat}: {len(rows or [])} standard stars within {radius_arcmin:.1f}' "
                          f"({time.time() - t0:.2f} s)")
                return rows
        if info.get('unreadable'):         # v5.0.1: format not recognised (see _mark_unreadable)
            return None
        idx = _table_index(cat, info)
        if _mark_unreadable(cat, info, idx):
            return None
        cand = []
        covered = False
        for p, e in idx.items():
            if e['n'] == 0:
                continue
            if e['dec0'] - rdeg <= dec <= e['dec1'] + rdeg:
                full_ra = (e['ra1'] - e['ra0']) > 180
                cosd = max(np.cos(np.radians(dec)), 1e-3)
                in_ra = full_ra or (e['ra0'] - rdeg / cosd <= ra <= e['ra1'] + rdeg / cosd)
                if in_ra:
                    cand.append(p)
                    if e['dec0'] <= dec <= e['dec1'] and (full_ra or e['ra0'] <= ra <= e['ra1']):
                        covered = True
        if not covered:
            return None
        for p in cand:
            _, header, delim = _file_layout(p)
            rows += _rows_from_file(p, header, delim, ra, dec, rdeg)
        if len(cand) > 1:              # v4.31: overlapping downloads (e.g. APASS DR10 boxes) - drop duplicates
            seen, uniq = set(), []
            for r_ in rows:
                k_ = (round(r_['_ra'], 5), round(r_['_dec'], 5))
                if k_ not in seen:
                    seen.add(k_)
                    uniq.append(r_)
            rows = uniq
    if verbose:
        print(f"Local {cat}: {len(rows)} stars within {radius_arcmin:.1f}' ({time.time() - t0:.2f} s)")
    return rows


# ------------------------------------------------------------------ values
def value(row, key):
    """Float value of a quantity ('g_sdss', 'r_sdss', 'G', 'BP', 'RP', 'V', 'B', 'BV', 'max', ...)
    from a raw row; millimag integers are converted; missing -> nan."""
    cols = list(row.keys())
    aliases = _COLS[key]
    if key in ('G',) and 'phot_g_mean_mag' not in row and 'gaia' in row and 'g' in row:
        aliases = ('gaia',)            # RefCat2: 'g' is Pan-STARRS g, 'gaia' is Gaia G
    i = _find([c.lower() for c in cols], aliases)
    if i is None:
        return np.nan
    try:
        v = float(str(row[cols[i]]).strip())
    except ValueError:
        return np.nan
    if key in ('max', 'min', 'period', 'BV'):
        return v if np.isfinite(v) else np.nan
    if abs(v) >= 100:                  # RefCat2 integer millimag
        v = v / 1000.0
    if not np.isfinite(v) or v <= 0 or v > 30:
        return np.nan
    return v


def text(row, key):
    cols = list(row.keys())
    i = _find([c.lower() for c in cols], _COLS[key])
    return str(row[cols[i]]).strip() if i is not None else ''


# ------------------------------------------------------------------ GCVS lists
_GCVS_CACHE = {}


def gcvs_available():
    _load_config()
    info = _STATE['catalogs'].get('GCVS')
    return bool(info and any(os.path.exists(f) for f in info.get('files', [])))


def gcvs_in_field(ra, dec, radius_arcmin):
    """
    v4.26: variables from the local GCVS list file(s) (same format as the target lists
    of Field Variables, e.g. the SAAF GCVS export) within radius_arcmin of (ra, dec).
    Returns a list of target dicts (name, ra, dec, type, period, max, min, epoch, source)
    or None if no local GCVS list is configured.
    """
    _load_config()
    info = _STATE['catalogs'].get('GCVS')
    if not _STATE['use_local'] or not info:
        return None
    stars = []
    for f, ent in _gcvs_lists(info):
        _, targets, (tra, tdec), _names = ent
        if not targets:
            continue
        d1, d2 = np.radians(dec), np.radians(tdec)
        cosang = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(np.radians(tra - ra))
        sep = np.degrees(np.arccos(np.clip(cosang, -1, 1))) * 60.0
        for k in np.where(sep <= radius_arcmin)[0]:
            t = dict(targets[k])
            t['source'] = 'GCVS (local)'
            stars.append(t)
    return stars


def _gcvs_lists(info):
    """[(file, (mtime, targets, (ra array, dec array), name index))] of the local GCVS list(s);
    each file is read once and again only when it changes."""
    from python.field_variables import parse_target_list
    out = []
    for f in info.get('files', []):
        try:
            mt = os.path.getmtime(f)
        except OSError:
            continue
        ent = _GCVS_CACHE.get(f)
        if not ent or ent[0] != mt:
            try:
                targets, _ = parse_target_list(f)
            except Exception as e:
                print(f"Local GCVS list {f} could not be read: {e}")
                continue
            targets = [t for t in targets if t.get('ra') is not None]
            arr = (np.array([t['ra'] for t in targets], float), np.array([t['dec'] for t in targets], float))
            names = {}
            for k, t in enumerate(targets):
                names.setdefault(name_key(t.get('name', '')), k)
            _GCVS_CACHE[f] = ent = (mt, targets, arr, names)
        out.append((f, ent))
    return out


def name_key(name):
    """Normalised star name for look-ups: case, spaces, underscores and a leading 'GCVS'
    do not matter ('HU Aqr' = 'hu aqr' = 'HU  AQR' = 'HU_Aqr'; 'V1500 Cyg' = 'V 1500 Cyg')."""
    s = re.sub(r'[\s_]+', '', str(name or '')).lower()
    return s[4:] if s.startswith('gcvs') and len(s) > 4 else s


def gcvs_lookup(name):
    """
    v4.30: star by name in the local GCVS list(s). Returns the target dict (name, ra, dec in
    degrees, type, period, ...; source 'GCVS (local)') or None (not in the list, no list
    configured, or local copies switched off).
    """
    _load_config()
    info = _STATE['catalogs'].get('GCVS')
    if not _STATE['use_local'] or not info or not str(name or '').strip():
        return None
    key = name_key(name)
    for f, ent in _gcvs_lists(info):
        k = ent[3].get(key)
        if k is not None:
            t = dict(ent[1][k])
            t['source'] = 'GCVS (local)'
            return t
    return None
