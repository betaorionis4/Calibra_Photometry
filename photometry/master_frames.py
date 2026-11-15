"""
Calibration frames: building master bias / dark / flat frames and calibrating
light frames - in files or on the fly (v4.27).

Building masters (build_masters):
    v4.29: the user gives one folder per type - a bias folder, a flat folder and
    (optionally) a dark folder. Every FITS file in such a folder is taken as a frame
    of that type (file names do not matter; a file whose header clearly says it is
    something else, e.g. IMAGETYP = 'Light Frame', is still used - the folder decides -
    and a warning is printed (v4.33; v4.29 left such files out). The
    masters are written into the same folder. Alternatively (v4.27) one mixed
    folder can be given, where the type comes from the FITS header (IMAGETYP /
    FRAME) or from the file/folder names (bias/offset, dark, flat, flat-dark).
    Frames are grouped by image size, binning, gain, offset, and
    filter (flats) or exposure time and temperature (darks). For every group a
    master is written as 32-bit float FITS into the folder of its raw frames:

        master_bias_<nx>x<ny>_bin1_g100_o50.fits
        master_dark_120s_m10C_<nx>x<ny>_bin1_g100_o50.fits
        master_darkflat_2.5s_<...>.fits
        master_flat_L_<nx>x<ny>_bin1_g100_o50.fits      (normalised, median = 1)

    Combination: per-pixel sigma-clipped mean (default; robust median/MAD clipping)
    or median, computed in horizontal strips so that memory stays small. Flats are
    corrected with the matching master flat-dark / dark (same exposure) or the
    master bias, each flat is normalised to its median before combining - so a
    Calibra master flat contains no bias any more and the bias is subtracted only
    once from every light. Masters are built in the order bias -> darks -> flats.
    The header of a master records the number of frames, the combination method
    and a hash of the input files (CALHASH; for flats it includes the hash of the
    subtracted bias/dark master); if a master with the same hash exists it is
    reused, if frames were added/changed (or the bias was rebuilt) it is rebuilt.

Calibrating (Calibrator):
    light_cal = (light - D) / F
    D = master bias, or - only if darks are switched on (option 'use_darks') - the
        master dark of the same exposure (+-5 %) and temperature (+-3 C) (it contains
        the bias), else master bias + (dark - bias) * t_light/t_dark (scaled dark).
    F = master flat of the light's filter (normalised). A master flat made by other
        software is assumed to be bias-subtracted already (option
        'flat_already_sub'); if that option is off and the flat is not normalised,
        the master bias is subtracted from it before normalising.
    Masters must match the light in image size and binning; gain/offset mismatches
    are reported. Pixels saturated in the raw frame keep a value >= the saturation
    limit so that photometry still recognises them.

On the fly (apply_on_the_fly): measuring functions (Light Curves, Field
Variables, Data Mining, pipeline) call it after reading a frame; with the option
switched on the frame is calibrated in memory (masters are loaded once and kept),
no calibrated files are written. Frames that were already calibrated (Calibra
'cal_' files, BIASFILE/FLATFILE/CALSTAT keywords, HISTORY entries of calibration
software) are left unchanged.
"""
import glob
import hashlib
import json
import os
import re
import time
import numpy as np
from astropy.io import fits

CONFIG_FILE = "calibration_config.json"
FRAME_TYPES = ('BIAS', 'DARK', 'DARKFLAT', 'FLAT')
FITS_EXT = ('.fits', '.fit', '.fts')

_KEYS = ('on_the_fly', 'folder', 'bias_folder', 'flat_folder', 'dark_folder', 'mode', 'manual', 'flat_already_sub',
         'sat_limit', 'write_files', 'use_darks')
_STATE = {'loaded': False, 'on_the_fly': False, 'folder': '', 'mode': 'auto',
          'bias_folder': '', 'flat_folder': '', 'dark_folder': '',
          'manual': {'bias': '', 'dark': '', 'flat_V': '', 'flat_B': '', 'flat_L': ''},
          'flat_already_sub': True, 'sat_limit': 63000.0, 'write_files': False, 'use_darks': False}


# ------------------------------------------------------------------ configuration
def _load():
    if _STATE['loaded']:
        return
    _STATE['loaded'] = True
    try:
        with open(CONFIG_FILE, encoding='utf-8') as f:
            cfg = json.load(f)
        for k in _KEYS:
            if k in cfg and k != 'manual':
                _STATE[k] = cfg[k]
        _STATE['manual'].update(cfg.get('manual', {}))
    except (OSError, ValueError):
        pass


def configure(save=True, **kw):
    """Set options: on_the_fly, folder (mixed), bias_folder, flat_folder, dark_folder, mode ('auto'|'manual'), manual (dict), flat_already_sub,
    sat_limit. Stored in calibration_config.json (also used by the pipeline without GUI)."""
    _load()
    for k, v in kw.items():
        if k == 'manual' and isinstance(v, dict):
            _STATE['manual'].update({kk: (vv or '') for kk, vv in v.items()})
        elif k in _STATE:
            _STATE[k] = v
    _CAL.reset()
    if save:
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump({k: _STATE[k] for k in _KEYS}, f, indent=2)
        except OSError as e:
            print(f"Calibration settings could not be saved: {e}")


def get_config():
    _load()
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in _STATE.items()}


# ------------------------------------------------------------------ frame scanning
_TYPE_WORDS = [('DARKFLAT', ('darkflat', 'dark_flat', 'dark-flat', 'flatdark', 'flat_dark', 'flat-dark', 'flatdarks')),
               ('BIAS', ('bias', 'offset', 'zero')),
               ('DARK', ('dark',)),
               ('FLAT', ('flat', 'skyflat', 'domeflat', 'twilight'))]


def frame_type(header, path='', use_names=True):
    """'BIAS', 'DARK', 'DARKFLAT', 'FLAT', 'LIGHT', 'MASTER' or None.
    use_names=False: only the FITS header (and 'master' in the file name) counts."""
    t = str(header.get('IMAGETYP', header.get('FRAME', header.get('IMAGETYPE', header.get('OBSTYPE', ''))))).strip().lower()
    if 'master' in t or os.path.basename(path).lower().startswith('master'):
        return 'MASTER'
    for typ, words in _TYPE_WORDS:
        if any(w in t.replace(' ', '') for w in words):
            return typ
    if 'light' in t or 'object' in t or 'science' in t:
        return 'LIGHT'
    if not use_names:
        return None
    name = (os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path)).lower()
    for typ, words in _TYPE_WORDS:
        if any(w in name for w in words):
            return typ
    return None


def _num(header, keys, default=None):
    for k in keys:
        v = header.get(k)
        try:
            if v is not None and str(v).strip() != '':
                return float(v)
        except (TypeError, ValueError):
            continue
    return default


def frame_info(path, header=None):
    if header is None:
        header = fits.getheader(path)
        if header.get('NAXIS', 0) < 2:
            try:
                header = fits.getheader(path, 1)
            except Exception:
                pass
    from photometry.filters import classify_band
    filt = str(header.get('FILTER', header.get('FILTER1', '')) or '').strip() or _filter_from_name(path)
    return {
        'path': path, 'type': frame_type(header, path),
        'shape': (int(header.get('NAXIS2', 0)), int(header.get('NAXIS1', 0))),
        'xbin': int(_num(header, ('XBINNING', 'BINNING'), 1) or 1), 'ybin': int(_num(header, ('YBINNING', 'BINNING'), 1) or 1),
        'gain': _num(header, ('GAIN', 'EGAIN_SET', 'ISOSPEED')), 'offset': _num(header, ('OFFSET', 'BLKLEVEL', 'PEDESTAL')),
        'exptime': _num(header, ('EXPTIME', 'EXPOSURE'), 0.0), 'temp': _num(header, ('CCD-TEMP', 'SET-TEMP', 'CCDTEMP')),
        'filter': filt, 'band': classify_band(filt) or (filt.upper() if filt else ''),
        'mtime': os.path.getmtime(path), 'size': os.path.getsize(path),
    }


_FILT_NAME = re.compile(r'(?:^|[_\-\s.])(lum|luminance|clear|Ha|OIII|SII|[LVBRGI])(?:mag)?(?=[_\-\s.]|\d|$)', re.IGNORECASE)


def _filter_from_name(path):
    """Filter from the file name when the header has none, e.g. 'flat_V_001.fits' -> 'V',
    'Flat-Lum-003.fit' -> 'L'. The folder name (e.g. .../flats/B/) is tried as well."""
    if not path:
        return ''
    stem = os.path.splitext(os.path.basename(path))[0]
    for cand in (stem, os.path.basename(os.path.dirname(path))):
        for m in _FILT_NAME.finditer(cand):
            f = m.group(1)
            if f.lower() in ('lum', 'luminance', 'clear'):
                return 'L'
            if len(f) == 1 and not f.isupper():
                continue                      # single letters only as capitals
            return f if len(f) > 1 else f.upper()
    return ''


def scan_folder(folder, max_depth=3, force_type=None, log=None):
    """Calibration frames (and existing masters) below folder.
    force_type ('BIAS', 'FLAT', 'DARK'): v4.29 - every FITS file in the folder is a frame of that
    type, whatever its name; files whose header names another type (e.g. a light frame) are used
    as well, with a warning (v4.33: the folder chosen by the user decides). Masters and Calibra 'cal_' files are recognised as before."""
    out = []
    base = os.path.abspath(folder).rstrip(os.sep).count(os.sep)
    for dp, dn, fn in os.walk(folder):
        if dp.count(os.sep) - base >= max_depth:
            dn[:] = []
        dn[:] = [d for d in dn if not d.startswith('.') and d.lower() not in ('calibrated',)]
        for f in sorted(fn):
            if not f.lower().endswith(FITS_EXT):
                continue
            p = os.path.join(dp, f)
            try:
                fi = frame_info(p)
            except Exception as e:
                (log or print)(f"  (skipped {f}: {e})")
                continue
            if force_type and fi['type'] != 'MASTER':
                if f.lower().startswith('cal_'):
                    continue
                try:
                    ht = frame_type(fits.getheader(p), p, use_names=False)
                except Exception:
                    ht = None
                ok = ht is None or ht == force_type or (force_type == 'DARK' and ht == 'DARKFLAT')
                if not ok:
                    (log or print)(f"  WARNING {f}: header says {ht}, but it is in the {force_type.lower()} folder - "
                                   f"used as {force_type.lower()}")
                fi['type'] = ht if (force_type == 'DARK' and ht == 'DARKFLAT') else force_type
            out.append(fi)
    return out


def _fmt(v, nd=0):
    if v is None:
        return 'x'
    s = f"{v:.{nd}f}" if nd else f"{v:g}"
    return s.replace('-', 'm').replace('.', 'p') if nd else s.replace('-', 'm')


def group_key(fi):
    """Frames that may be combined into one master."""
    base = (fi['type'], fi['shape'], fi['xbin'], fi['ybin'], fi['gain'], fi['offset'])
    if fi['type'] in ('DARK', 'DARKFLAT'):
        t = None if fi['temp'] is None else int(round(fi['temp'] / 2.0) * 2)
        return base + (round(fi['exptime'], 1), t)
    if fi['type'] == 'FLAT':
        return base + (fi['band'] or fi['filter'],)
    return base


def master_name(key):
    typ, shape, xb, yb, gain, off = key[:6]
    tail = f"{shape[1]}x{shape[0]}_bin{xb}" + (f"x{yb}" if yb != xb else "") + f"_g{_fmt(gain)}_o{_fmt(off)}"
    if typ in ('DARK', 'DARKFLAT'):
        exp, t = key[6], key[7]
        return f"master_{typ.lower()}_{exp:g}s_{'' if t is None else _fmt(t) + 'C_'}{tail}.fits"
    if typ == 'FLAT':
        filt = re.sub(r'[^A-Za-z0-9]+', '', str(key[6])) or 'nofilter'
        return f"master_flat_{filt}_{tail}.fits"
    return f"master_bias_{tail}.fits"


def _input_hash(frames):
    h = hashlib.sha1()
    for fi in sorted(frames, key=lambda x: os.path.basename(x['path'])):
        h.update(f"{os.path.basename(fi['path'])}|{fi['size']}|{int(fi['mtime'])}".encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ combining
def _read_rows(path, y0, y1, step=1):
    """Rows y0:y1 (every step-th row/column) of a frame as float32, memory-mapped; integer
    data with BZERO/BSCALE are scaled here (astropy cannot memory-map scaled data)."""
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as h:
        hdu = h[0] if h[0].header.get('NAXIS', 0) >= 2 else h[1]
        d = hdu.data
        if d.ndim == 3:
            d = d[0]
        out = np.asarray(d[y0:y1:step, ::step], dtype=np.float32)
        bscale = float(hdu.header.get('BSCALE', 1.0)); bzero = float(hdu.header.get('BZERO', 0.0))
    if bscale != 1.0:
        out *= bscale
    if bzero != 0.0:
        out += bzero
    return out


def combine_frames(paths, method='sigma', sigma=3.0, subtract=None, normalize=False, max_bytes=200e6,
                   progress=None):
    """
    Per-pixel combination of frames (same size) in horizontal strips.
    subtract: 2-D array subtracted from every frame first (bias / flat-dark).
    normalize: divide every frame by its median (flats).
    method: 'sigma' (mean after clipping at sigma * 1.4826*MAD around the median) or 'median'.
    Returns float32 array.
    """
    n = len(paths)
    with fits.open(paths[0], memmap=True, do_not_scale_image_data=True) as h:
        hdu = h[0] if h[0].header.get('NAXIS', 0) >= 2 else h[1]
        ny, nx = int(hdu.header['NAXIS2']), int(hdu.header['NAXIS1'])
    scale = np.ones(n, np.float32)
    if normalize:
        for k, p in enumerate(paths):         # median of each (bias-corrected) frame, from a subsample
            d = _read_rows(p, 0, ny, step=4)
            if subtract is not None:
                d = d - subtract[::4, ::4]
            m = float(np.median(d))
            scale[k] = m if m > 0 else np.nan
    rows = int(max(8, min(ny, max_bytes / (4.0 * n * nx))))
    out = np.empty((ny, nx), np.float32)
    for y0 in range(0, ny, rows):
        y1 = min(ny, y0 + rows)
        stack = np.empty((n, y1 - y0, nx), np.float32)
        for k, p in enumerate(paths):
            s = _read_rows(p, y0, y1)
            if subtract is not None:
                s = s - subtract[y0:y1]
            if normalize:
                s = s / scale[k]
            stack[k] = s
        med = np.median(stack, axis=0)
        if method == 'median' or n < 3:
            out[y0:y1] = med if n >= 3 or method == 'median' else stack.mean(axis=0)
        else:
            mad = np.median(np.abs(stack - med), axis=0) * 1.4826
            mad = np.where(mad > 0, mad, np.inf)
            good = np.abs(stack - med) <= sigma * mad
            cnt = good.sum(axis=0)
            s_ = np.where(good, stack, 0).sum(axis=0)
            out[y0:y1] = np.where(cnt > 0, s_ / np.maximum(cnt, 1), med)
        if progress:
            progress(y1 / ny)
    return out


def _write_master(path, data, key, frames, method, extra=None, calhash=None):
    typ = key[0]
    h = fits.Header()
    h['IMAGETYP'] = (f"MASTER {typ}", 'Calibra master frame')
    h['NCOMBINE'] = (len(frames), 'number of combined frames')
    h['COMBMETH'] = ('sigma-clipped mean' if method == 'sigma' else 'median', 'combination')
    h['XBINNING'] = key[2]; h['YBINNING'] = key[3]
    if key[4] is not None:
        h['GAIN'] = key[4]
    if key[5] is not None:
        h['OFFSET'] = key[5]
    if typ in ('DARK', 'DARKFLAT'):
        h['EXPTIME'] = (float(np.median([f['exptime'] for f in frames])), '[s]')
        temps = [f['temp'] for f in frames if f['temp'] is not None]
        if temps:
            h['CCD-TEMP'] = (round(float(np.mean(temps)), 2), '[C] mean of the frames')
    if typ == 'FLAT':
        h['FILTER'] = frames[0]['filter'] or str(key[6])
        h['FLATNORM'] = (True, 'median 1, bias/flat-dark subtracted')
    h['CALHASH'] = (calhash or _input_hash(frames), 'hash of the input frames')
    h['CREATOR'] = 'Calibra'
    for k, v in (extra or {}).items():
        h[k] = v
    for f in sorted(frames, key=lambda x: x['path'])[:200]:
        h.add_history(os.path.basename(f['path'])[:70])
    fits.PrimaryHDU(data=np.asarray(data, np.float32), header=h).writeto(path, overwrite=True)


def _sources(folder=None, bias_folder=None, flat_folder=None, dark_folder=None):
    """[(folder, forced type or None, max_depth)] of the existing folders, duplicates removed."""
    out, seen = [], set()
    for d, t, depth in ((bias_folder, 'BIAS', 2), (dark_folder, 'DARK', 2), (flat_folder, 'FLAT', 2), (folder, None, 3)):
        if d and os.path.isdir(d):
            key = (os.path.abspath(d), t)
            if key not in seen:
                seen.add(key)
                out.append((d, t, depth))
    return out


def build_masters(folder=None, method='sigma', sigma=3.0, min_frames=3, progress=None, log=print, force=False,
                  bias_folder=None, flat_folder=None, dark_folder=None):
    """
    Create (or reuse) master frames for all groups of raw calibration frames - in the bias / flat /
    dark folders (v4.29: every FITS file there is of that type) and/or in a mixed folder (type from
    header / names). Every master is written into the folder of its raw frames.
    Returns list of dicts {path, type, key, status ('built'|'reused'|'skipped'), n, note}.
    """
    frames, seen = [], set()
    for d, t, depth in _sources(folder, bias_folder, flat_folder, dark_folder):
        for fi in scan_folder(d, max_depth=depth, force_type=t, log=log):
            ap = os.path.abspath(fi['path'])
            if ap not in seen:
                seen.add(ap)
                frames.append(fi)
    raw = [f for f in frames if f['type'] in FRAME_TYPES]
    groups = {}
    for f in raw:
        groups.setdefault(group_key(f), []).append(f)
    order = {'BIAS': 0, 'DARKFLAT': 1, 'DARK': 2, 'FLAT': 3}
    results = []
    built = {}                    # key -> (path, data) for bias / darkflat (needed for flats)
    keys = sorted(groups, key=lambda k: (order[k[0]], str(k)))
    for gi, key in enumerate(keys):
        fr = sorted(groups[key], key=lambda f: f['path'])
        typ = key[0]
        out_dir = os.path.dirname(fr[0]['path'])
        name = master_name(key)
        out = os.path.join(out_dir, name)
        res = {'path': out, 'type': typ, 'key': key, 'n': len(fr), 'note': ''}
        if len(fr) < min_frames:
            res.update(status='skipped', note=f"only {len(fr)} frame(s) (minimum {min_frames})")
            results.append(res); log(f"{name}: skipped - {res['note']}")
            continue
        sub_path = None
        if typ == 'FLAT':
            sub_path = _find_for_flat(key, fr, built, results)
        h = _input_hash(fr)
        if sub_path:                  # flats depend on the subtracted master as well
            try:
                h = hashlib.sha1((h + '|' + str(fits.getheader(sub_path).get('CALHASH', os.path.basename(sub_path))))
                                 .encode()).hexdigest()[:16]
            except Exception:
                pass
        if not force and os.path.exists(out):
            try:
                if fits.getheader(out).get('CALHASH') == h:
                    res['status'] = 'reused'
                    results.append(res); log(f"{name}: up to date ({len(fr)} frames) - reused")
                    if typ in ('BIAS', 'DARKFLAT', 'DARK'):
                        built[key] = out
                    continue
            except Exception:
                pass
        t0 = time.time()
        sub, extra = None, {}
        if typ == 'FLAT':
            if sub_path:
                sub = fits.getdata(sub_path).astype(np.float32)
                extra['FLATCORR'] = (os.path.basename(sub_path)[:60], 'subtracted')
                res['note'] = f"corrected with {os.path.basename(sub_path)}"
            else:
                res['note'] = "WARNING: no master bias or flat-dark found - flats not bias-corrected"
                extra['FLATCORR'] = ('none', 'no bias/flat-dark available')
        log(f"{name}: combining {len(fr)} frames ...")
        data = combine_frames([f['path'] for f in fr], method=method, sigma=sigma, subtract=sub,
                              normalize=(typ == 'FLAT'),
                              progress=(lambda x, gi=gi: progress((gi + x) / len(keys), name)) if progress else None)
        if typ == 'FLAT':
            med = float(np.median(data[::4, ::4]))
            data = data / med if med > 0 else data
        _write_master(out, data, key, fr, method, extra, calhash=h)
        res['status'] = 'built'
        if typ in ('BIAS', 'DARKFLAT', 'DARK'):
            built[key] = out
        log(f"{name}: written ({time.time() - t0:.1f} s){' - ' + res['note'] if res['note'] else ''}")
        results.append(res)
    if not raw:
        log("No raw bias/dark/flat frames found (bias/flat/dark folders: every FITS file; mixed folder: "
            "type from IMAGETYP or from file/folder names).")
    return results


def _find_for_flat(flat_key, flats, built, results):
    """Master flat-dark (or dark) with the flats' exposure, else master bias - matching size/bin/gain/offset."""
    shape, xb, yb, gain, off = flat_key[1:6]
    exp = float(np.median([f['exptime'] for f in flats]))
    best = None
    for key, path in built.items():
        if key[1:4] != (shape, xb, yb):
            continue
        same_go = (key[4] == gain or key[4] is None or gain is None) and (key[5] == off or key[5] is None or off is None)
        if not same_go:
            continue
        if key[0] in ('DARKFLAT', 'DARK') and abs(key[6] - exp) <= max(0.05 * exp, 0.05):
            return path
        if key[0] == 'BIAS':
            best = path
    return best


# ------------------------------------------------------------------ master library and selection
def all_folders():
    """Configured calibration folders (bias, flat, dark, mixed) - where masters are looked for."""
    _load()
    return tuple(d for d in (_STATE['bias_folder'], _STATE['flat_folder'], _STATE['dark_folder'], _STATE['folder']) if d)


def find_masters(folder=None, *more):
    """All master frames (Calibra's, or other software's with IMAGETYP 'Master ...' / name 'master*')."""
    out, seen = [], set()
    infos = []
    for d in (folder,) + more:
        if d and os.path.isdir(d):
            infos += scan_folder(d)
    for fi in infos:
        if fi['type'] != 'MASTER' or os.path.abspath(fi['path']) in seen:
            continue
        seen.add(os.path.abspath(fi['path']))
        try:
            h = fits.getheader(fi['path'])
        except Exception:
            continue
        t = str(h.get('IMAGETYP', '')).upper() + ' ' + os.path.basename(fi['path']).upper()
        if 'DARKFLAT' in t.replace(' ', '').replace('_', '').replace('-', '') or 'FLATDARK' in t.replace('_', '').replace('-', ''):
            mt = 'DARKFLAT'
        elif 'FLAT' in t:
            mt = 'FLAT'
        elif 'DARK' in t:
            mt = 'DARK'
        elif 'BIAS' in t or 'OFFSET' in t:
            mt = 'BIAS'
        else:
            continue
        fi['mtype'] = mt
        out.append(fi)
    return out


def _matches(light, m, strict_go=False):
    if tuple(m['shape']) != tuple(light['shape']) or m['xbin'] != light['xbin'] or m['ybin'] != light['ybin']:
        return False
    if strict_go:
        for k in ('gain', 'offset'):
            if light[k] is not None and m[k] is not None and abs(light[k] - m[k]) > 1e-6:
                return False
    return True


def select_masters(light, masters):
    """Choose (bias, dark, flat, notes) for a light frame (frame_info dict) from find_masters()."""
    notes = []
    def pick(mtype, extra=None):
        cands = [m for m in masters if m['mtype'] == mtype and _matches(light, m)]
        if extra:
            cands = [m for m in cands if extra(m)]
        strict = [m for m in cands if _matches(light, m, strict_go=True)]
        if cands and not strict:
            notes.append(f"{mtype.lower()}: no master with the light's gain/offset - using {os.path.basename(cands[0]['path'])}")
        return (strict or cands or [None])[0], cands
    bias, _ = pick('BIAS')
    exp, temp = light['exptime'] or 0.0, light['temp']
    def dark_ok(m):
        ok = abs((m['exptime'] or 0) - exp) <= max(0.05 * exp, 0.05)
        if ok and temp is not None and m['temp'] is not None:
            ok = abs(m['temp'] - temp) <= 3.0
        return ok
    dark = None
    if _STATE.get('use_darks'):
        dark, _ = pick('DARK', dark_ok)
    if dark is None and _STATE.get('use_darks'):
        _, all_darks = pick('DARK')
        if all_darks and bias is not None:
            # longest dark with a similar temperature for scaling
            cand = [m for m in all_darks if temp is None or m['temp'] is None or abs(m['temp'] - temp) <= 3.0]
            if cand:
                dark = max(cand, key=lambda m: m['exptime'] or 0)
                dark = dict(dark, scaled=True)
                notes.append(f"dark scaled from {dark['exptime']:g} s to {exp:g} s")
    band = light['band']
    flat, _ = pick('FLAT', lambda m: (m['band'] or m['filter']).upper() == (band or light['filter']).upper())
    if flat is None:
        notes.append(f"no master flat for filter '{light['filter']}'")
    if bias is None and dark is None:
        notes.append("no master bias" + (" or dark" if _STATE.get('use_darks') else ""))
    return bias, dark, flat, notes


# ------------------------------------------------------------------ calibrating
_CAL_HIST = re.compile(r'calibrat|bias|dark|flat|master', re.IGNORECASE)


def skip_reason(header, path=''):
    """Why a light frame must not be calibrated ('' = calibrate):
    already calibrated (Calibra 'cal_' file, BIASFILE/FLATFILE/DARKFILE/CALSTAT keywords,
    HISTORY of calibration software)."""
    if os.path.basename(path).lower().startswith('cal_'):
        return "already calibrated (cal_ file)"
    for k in ('BIASFILE', 'FLATFILE', 'DARKFILE', 'CALSTAT', 'FLATCORR', 'BIASCORR', 'DARKCORR'):
        if k in header and str(header.get(k)).strip() not in ('', 'F', 'False'):
            return f"already calibrated ({k})"
    try:
        for h in header.get('HISTORY', []):
            if _CAL_HIST.search(str(h)):
                return "already calibrated (HISTORY)"
    except Exception:
        pass
    return ''


def stack_note(header):
    """Warning text for stacks (calibrating after stacking is only an approximation:
    the registration shifts the flat-field pattern between the sub-frames)."""
    for k in ('STACKCNT', 'NCOMBINE', 'NSTACK', 'SNAPSHOT'):
        try:
            n = int(header.get(k, 0) or 0)
        except (TypeError, ValueError):
            n = 0
        if n > 1:
            return (f"stack of {n} frames: calibration after stacking is only approximate "
                    "(better: calibrate the sub-frames before stacking)")
    return ''


def is_precalibrated(header, path=''):
    return bool(skip_reason(header, path))


class Calibrator:
    """Keeps master frames in memory and calibrates light frames."""

    def __init__(self, max_cached=8):
        self.cache = {}
        self.hdr = {}
        self.max_cached = max_cached
        self.masters = None
        self.warned = set()

    def _is_calibra_flat(self, path):
        """Calibra's own master flats are bias-corrected and normalised (FLATNORM)."""
        if path not in self.hdr:
            try:
                h = fits.getheader(path)
                self.hdr[path] = bool(h.get('FLATNORM', False)) or str(h.get('CREATOR', '')) == 'Calibra'
            except Exception:
                self.hdr[path] = False
        return self.hdr[path]

    def reset(self):
        self.hdr.clear()
        self.cache.clear()
        self.masters = None
        self.warned.clear()

    def _load(self, path, adu_scale=True):
        if path not in self.cache:
            if len(self.cache) >= self.max_cached:
                self.cache.pop(next(iter(self.cache)))
            d = fits.getdata(path).astype(np.float32)
            if d.ndim == 3:
                d = d[0]
            # masters stored normalised to 0..1 (e.g. Siril 32-bit) -> ADU (bias/dark only)
            if adu_scale and np.nanmax(d) <= 1.1 and 'FLAT' not in str(fits.getheader(path).get('IMAGETYP', '')).upper() \
                    and 'flat' not in os.path.basename(path).lower():
                d = d * 65535.0
            self.cache[path] = d
        return self.cache[path]

    def library(self):
        if self.masters is None:
            _load()
            self.masters = find_masters(*all_folders()) if _STATE['mode'] == 'auto' else []
        return self.masters

    def choose(self, light):
        """(bias_path, dark (path, scaled, t_dark), flat_path, notes) for a frame_info dict."""
        _load()
        if _STATE['mode'] == 'manual':
            m = _STATE['manual']
            flat = m.get(f"flat_{light['band']}") or ''
            notes = [] if flat else [f"no manual master flat for filter '{light['filter']}'"]
            dark = (m.get('dark'), False, None) if (m.get('dark') and _STATE.get('use_darks')) else None
            return (m.get('bias') or None), dark, (flat or None), notes
        bias, dark, flat, notes = select_masters(light, self.library())
        d = (dark['path'], bool(dark.get('scaled')), dark['exptime']) if dark else None
        return (bias['path'] if bias else None), d, (flat['path'] if flat else None), notes

    def calibrate(self, data, header, path=''):
        """Returns (calibrated float32 array or None, info dict)."""
        _load()
        light = frame_info(path, header) if path else None
        if light is None:
            return None, {'error': 'no path'}
        bias_p, dark, flat_p, notes = self.choose(light)
        info = {'bias': bias_p, 'dark': dark[0] if dark else None, 'flat': flat_p, 'notes': notes}
        if flat_p is None or (bias_p is None and dark is None):
            info['error'] = "; ".join(notes) or "masters missing"
            return None, info
        raw = np.asarray(data, dtype=np.float32)
        if bias_p and self._load(bias_p).shape != raw.shape:
            info['error'] = "master size differs from the frame"
            return None, info
        if dark is not None:
            dpath, scaled, t_dark = dark
            D = self._load(dpath)
            if scaled and bias_p:
                B = self._load(bias_p)
                f = (light['exptime'] or 0.0) / t_dark if t_dark else 0.0
                D = B + (D - B) * f
        else:
            D = self._load(bias_p)
        F = self._load(flat_p)
        if not _STATE['flat_already_sub'] and bias_p and not self._is_calibra_flat(flat_p) \
                and float(np.median(F[::8, ::8])) > 1.1:
            F = F - self._load(bias_p)        # flat from other software, not yet bias-subtracted
        med = float(np.median(F[::4, ::4]))
        Fn = F / med if med > 0 else F
        Fn = np.where(np.isfinite(Fn) & (Fn > 0.05), Fn, 1.0)
        cal = (raw - D) / Fn
        sat = float(_STATE['sat_limit'] or 0)
        if sat > 0:
            m = raw >= sat
            if np.any(m):
                cal[m] = np.maximum(cal[m], raw[m])     # saturated pixels stay recognisable
        return cal.astype(np.float32), info


_CAL = Calibrator()


def state_tag():
    """Short tag of the on-the-fly calibration state ('' if off) - part of measurement cache keys."""
    _load()
    if not _STATE['on_the_fly']:
        return ''
    if _STATE['mode'] == 'manual':
        parts = [v for v in _STATE['manual'].values() if v]
    else:
        parts = [m['path'] for m in _CAL.library()]
    sig = "|".join(f"{p}:{os.path.getmtime(p) if os.path.exists(p) else 0}" for p in sorted(parts))
    return "cal" + hashlib.sha1(f"{_STATE['mode']}|{sig}|{_STATE['sat_limit']}".encode()).hexdigest()[:10]


def apply_on_the_fly(data, header, path):
    """Calibrate a frame in memory if the option is on. Returns (data, calibrated_flag)."""
    _load()
    if not _STATE['on_the_fly']:
        return data, False
    why = skip_reason(header, path)
    if why:
        key = (os.path.dirname(path), why.split(' (')[0][:20])
        if key not in _CAL.warned:
            _CAL.warned.add(key)
            print(f"Calibration on the fly not applied to {os.path.basename(path)}: {why}.")
        return data, False
    sn = stack_note(header)
    if sn and ('stack', os.path.dirname(path)) not in _CAL.warned:
        _CAL.warned.add(('stack', os.path.dirname(path)))
        print(f"Note ({os.path.basename(path)}): {sn}.")
    try:
        cal, info = _CAL.calibrate(data, header, path)
    except Exception as e:
        cal, info = None, {'error': str(e)}
    if cal is None:
        key = (os.path.dirname(path), info.get('error'))
        if key not in _CAL.warned:
            _CAL.warned.add(key)
            print(f"Calibration on the fly NOT applied to {os.path.basename(path)}: {info.get('error')}")
        return data, False
    return cal, True


def calibrate_to_file(path, out_dir=None, overwrite=False):
    """Calibrate one file and write cal_<name> (32-bit float) to out_dir (default: 'calibrated'
    next to the raw file). Existing results made with the same masters are reused.
    Returns (out_path or None, message)."""
    _load()
    with fits.open(path) as h:
        hdu = h[0] if h[0].data is not None else h[1]
        data, header = hdu.data, hdu.header.copy()
    if data.ndim == 3:
        data = data[0]
    why = skip_reason(header, path)
    if why:
        return None, why + " - skipped"
    out_dir = out_dir or os.path.join(os.path.dirname(path), 'calibrated')
    out = os.path.join(out_dir, "cal_" + os.path.basename(path))
    light = frame_info(path, header)
    bias_p, dark, flat_p, notes = _CAL.choose(light)
    tag = hashlib.sha1("|".join(str(x) for x in (bias_p, dark, flat_p, _STATE['sat_limit'])).encode()).hexdigest()[:12]
    if not overwrite and os.path.exists(out):
        try:
            if fits.getheader(out).get('CALTAG') == tag and os.path.getmtime(out) >= os.path.getmtime(path):
                return out, "up to date - reused"
        except Exception:
            pass
    cal, info = _CAL.calibrate(data, header, path)
    if cal is None:
        return None, info.get('error', 'not calibrated')
    sn = stack_note(header)
    if sn:
        info['notes'] = info['notes'] + [sn]
    for k in ('BZERO', 'BSCALE', 'BLANK'):
        if k in header:
            del header[k]
    header['HISTORY'] = "Calibrated with Calibra (32-bit float)"
    header['BIASFILE'] = os.path.basename(info['bias'] or 'none')[:68]
    if info['dark']:
        header['DARKFILE'] = os.path.basename(info['dark'])[:68]
    header['FLATFILE'] = os.path.basename(info['flat'])[:68]
    header['CALSRC'] = (os.path.basename(path)[:68], 'Uncalibrated source file')
    header['CALTAG'] = (tag, 'Calibra masters used')
    if _STATE['sat_limit']:
        header['SATURATE'] = (float(_STATE['sat_limit']), 'raw saturation level (kept in calibrated data)')
    os.makedirs(out_dir, exist_ok=True)
    fits.PrimaryHDU(data=cal, header=header).writeto(out, overwrite=True)
    return out, "calibrated" + (" (" + "; ".join(info['notes']) + ")" if info['notes'] else "")
