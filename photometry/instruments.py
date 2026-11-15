"""
Instrument setups (v4.36).

One setup per telescope / camera / reducer / site combination (e.g. the 4" refractor in
Uppsala, the MWT in Raspay, the Vespera 2 with its OSC camera). A setup holds

  * a description of the instrument: telescope, aperture, focal length, reducer, camera,
    pixel size, sensor size, binning, colour (OSC) or mono, capture gain/offset;
  * the observing site: name, latitude, longitude, elevation, time zone;
  * the filters: name in the FITS header for V, B, L (and others) and their position in the
    filter wheel (for NINA sequences);
  * planning data: standard exposure times, bright/faint magnitude limits;
  * a snapshot of all Calibra settings that depend on the instrument (CCD gain, read noise,
    saturation, apertures, detection, default zero points, colour and extinction
    coefficients, calibration folders/masters, plate-solving field, ...).

Setups are stored as JSON files in the folder 'instruments' of the Calibra folder. The GUI
keeps the active setup up to date: when another setup is chosen, the current settings are
saved into the old one and the settings of the new one are applied.
"""
import json
import os
import re
import time

DIR = "instruments"

# Calibra settings (vars_dict keys) that belong to an instrument setup
SETTING_KEYS = (
    'filter_v_keyword', 'filter_b_keyword', 'filter_l_keyword',
    'ccd_gain', 'ccd_read_noise', 'ccd_dark_current', 'saturation_limit', 'sky_noise_factor',
    'detect_sigma', 'detect_fwhm', 'dao_sharplo', 'dao_sharphi', 'dao_roundlo', 'dao_roundhi',
    'centroid_initial_radius', 'centroid_tracking_shift', 'box_size',
    'use_flexible_aperture', 'aperture_fwhm_factor', 'annulus_inner_gap', 'annulus_width',
    'aperture_radius', 'annulus_inner', 'annulus_outer',
    'match_tolerance_arcsec', 'default_zp_v', 'default_zp_b', 'calib_snr_threshold',
    'catalog_search_radius', 'calib_peak_limit', 'extinction_kv', 'extinction_kb',
    'ts_lat', 'ts_lon', 'ts_coeff', 'ts_k', 'diff_tbv', 'diff_tvbv', 'diff_tbbv',
    'plate_radius', 'plate_fov', 'xy_x_min', 'xy_x_max', 'xy_y_min', 'xy_y_max',
    'cal_bias_folder', 'cal_flat_folder', 'cal_dark_folder', 'cal_frames_folder', 'cal_mode',
    'cal_use_darks', 'cal_on_the_fly', 'cal_combine_method', 'cal_min_frames',
    'flat_already_bias_subtracted', 'bias_path', 'flat_v_path', 'flat_b_path', 'flat_l_path', 'dark_path',
)

BANDS = ('V', 'B', 'L', 'R', 'G', 'I')
_FILTER_KEYS = {'V': 'filter_v_keyword', 'B': 'filter_b_keyword', 'L': 'filter_l_keyword'}


def template(name):
    return {
        'name': name, 'description': '', 'created': time.strftime('%Y-%m-%d'),
        'instrument': {'telescope': '', 'aperture_mm': None, 'focal_length_mm': None, 'reducer': '',
                       'camera': '', 'pixel_um': None, 'sensor_x': None, 'sensor_y': None, 'binning': 1,
                       'osc': False, 'gain': None, 'offset': None},
        'site': {'name': '', 'lat': None, 'lon': None, 'elevation_m': None, 'timezone': ''},
        'filters': {b: {'fits_name': '', 'nina_position': None} for b in BANDS},
        'planning': {'exposures_s': [30, 60, 120], 'mag_bright': None, 'mag_faint': None},
        'settings': {}, 'settings_saved': '',
    }


def _slug(name):
    return re.sub(r'[^A-Za-z0-9_.+-]+', '_', str(name).strip()).strip('_') or 'setup'


def path_for(name):
    return os.path.join(DIR, _slug(name) + ".json")


def list_names():
    if not os.path.isdir(DIR):
        return []
    out = []
    for f in sorted(os.listdir(DIR)):
        if f.endswith('.json'):
            try:
                with open(os.path.join(DIR, f), encoding='utf-8') as fh:
                    out.append(json.load(fh).get('name') or f[:-5])
            except (OSError, ValueError):
                continue
    return sorted(out, key=str.lower)


def _merge(base, extra):
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load(name):
    try:
        with open(path_for(name), encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return _merge(template(d.get('name', name)), d)       # fields added in later versions get defaults


def save(setup):
    os.makedirs(DIR, exist_ok=True)
    p = path_for(setup['name'])
    tmp = p + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(setup, f, indent=2, default=str)
    os.replace(tmp, p)
    return p


def delete(name):
    try:
        os.remove(path_for(name))
        return True
    except OSError:
        return False


def rename(old, new):
    s = load(old)
    if s is None or not str(new).strip():
        return None
    if os.path.exists(path_for(new)) and path_for(new) != path_for(old):
        raise ValueError(f"A setup named '{new}' exists already.")
    s['name'] = str(new).strip()
    save(s)
    if path_for(old) != path_for(new):
        delete(old)
    return s


# ------------------------------------------------------------------ settings snapshot
def snapshot(setup, values, sync_descriptive=True):
    """Store the current Calibra settings (dict vars_dict key -> value) in the setup. With
    sync_descriptive, site coordinates and filter names are kept in step with the
    descriptive fields (edits in the Settings tab of the active setup)."""
    setup['settings'] = {k: values[k] for k in SETTING_KEYS if k in values}
    setup['settings_saved'] = time.strftime('%Y-%m-%d %H:%M:%S')
    if not sync_descriptive:
        return setup
    for k, key in (('lat', 'ts_lat'), ('lon', 'ts_lon')):
        if key in values:
            try:
                setup['site'][k] = float(values[key])
            except (TypeError, ValueError):
                pass
    for b, key in _FILTER_KEYS.items():
        if key in values and str(values[key]).strip():
            setup['filters'].setdefault(b, {'fits_name': '', 'nina_position': None})['fits_name'] = str(values[key])
    return setup


def values_to_apply(setup):
    """vars_dict key -> value for activating the setup (its settings, with site coordinates
    and filter names from the descriptive fields where those are given)."""
    out = dict(setup.get('settings') or {})
    site = setup.get('site') or {}
    if site.get('lat') is not None:
        out['ts_lat'] = site['lat']
    if site.get('lon') is not None:
        out['ts_lon'] = site['lon']
    for b, key in _FILTER_KEYS.items():
        fn = ((setup.get('filters') or {}).get(b) or {}).get('fits_name')
        if fn:
            out[key] = fn
    return out


# ------------------------------------------------------------------ derived quantities
def _f(v):
    try:
        v = float(v)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def derived(setup):
    """{'pixel_scale': arcsec/px, 'fov_x_deg', 'fov_y_deg', 'focal_ratio'} - None where unknown."""
    ins = setup.get('instrument') or {}
    f, pix, b = _f(ins.get('focal_length_mm')), _f(ins.get('pixel_um')), _f(ins.get('binning')) or 1.0
    ap = _f(ins.get('aperture_mm'))
    scale = 206.265 * pix * b / f if (f and pix) else _f(ins.get('pixel_scale'))
    nx, ny = _f(ins.get('sensor_x')), _f(ins.get('sensor_y'))
    return {'pixel_scale': scale,
            'fov_x_deg': scale * nx / b / 3600.0 if (scale and nx) else None,
            'fov_y_deg': scale * ny / b / 3600.0 if (scale and ny) else None,
            'focal_ratio': f / ap if (f and ap) else None}


def summary(setup):
    d = derived(setup)
    ins, site = setup.get('instrument') or {}, setup.get('site') or {}
    parts = []
    tel = " ".join(x for x in (ins.get('telescope') or '', f"{ins['aperture_mm']:g} mm" if _f(ins.get('aperture_mm')) else '',
                               f"f={ins['focal_length_mm']:g} mm" if _f(ins.get('focal_length_mm')) else '',
                               f"(f/{d['focal_ratio']:.1f})" if d['focal_ratio'] else '') if x)
    if tel:
        parts.append(tel)
    cam = " ".join(x for x in (ins.get('camera') or '', "OSC" if ins.get('osc') else '',
                               f"{ins['pixel_um']:g} um" if _f(ins.get('pixel_um')) else '') if x)
    if cam:
        parts.append(cam)
    if d['pixel_scale']:
        parts.append(f"{d['pixel_scale']:.2f}\"/px")
    if d['fov_x_deg'] and d['fov_y_deg']:
        parts.append(f"field {d['fov_x_deg'] * 60:.0f}' x {d['fov_y_deg'] * 60:.0f}'")
    if site.get('name') or site.get('lat') is not None:
        txt = "site" + (" " + site['name'] if site.get('name') else '')
        if _f(site.get('lat')) is not None and _f(site.get('lon')) is not None:
            txt += " (%.2f, %.2f)" % (_f(site['lat']), _f(site['lon']))
        parts.append(txt.strip())
    return " | ".join(parts) or "(no instrument data yet)"


# ------------------------------------------------------------------ FITS headers
def _hv(header, keys, default=None):
    for k in keys:
        v = header.get(k)
        if v not in (None, ''):
            return v
    return default


def from_header(header):
    """Instrument / site fields that can be read from a FITS header (NINA, MaxIm, SharpCap,
    Vespera ...). Returns a partial setup dict {'instrument': {...}, 'site': {...}, 'filter': name}."""
    ins, site = {}, {}
    for key, hk in (('telescope', ('TELESCOP',)), ('camera', ('INSTRUME', 'CAMERA'))):
        v = _hv(header, hk)
        if v:
            ins[key] = str(v).strip()
    for key, hk in (('focal_length_mm', ('FOCALLEN', 'FOCAL')), ('aperture_mm', ('APTDIA', 'APERTURE')),
                    ('pixel_um', ('XPIXSZ', 'PIXSIZE1', 'PIXSIZE')), ('binning', ('XBINNING', 'BINNING')),
                    ('gain', ('GAIN',)), ('offset', ('OFFSET', 'BLKLEVEL'))):
        v = _f(_hv(header, hk))
        if v is not None:
            ins[key] = v
    b = ins.get('binning') or 1
    nx, ny = _f(header.get('NAXIS1')), _f(header.get('NAXIS2'))
    if nx and ny:
        ins['sensor_x'], ins['sensor_y'] = int(nx * b), int(ny * b)
    if ins.get('pixel_um') and _hv(header, ('XPIXSZ',)) is not None and b > 1:
        # NINA/MaxIm write XPIXSZ including the binning
        ins['pixel_um'] = ins['pixel_um'] / b
    if str(header.get('BAYERPAT', '') or '').strip():
        ins['osc'] = True
    # pixel scale from the WCS if the focal length is missing
    try:
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
        w = WCS(header)
        if w.has_celestial:
            ins['pixel_scale'] = float(proj_plane_pixel_scales(w.celestial)[0] * 3600.0)
    except Exception:
        pass
    for key, hk in (('lat', ('SITELAT', 'LAT-OBS', 'OBSLAT')), ('lon', ('SITELONG', 'LONG-OBS', 'OBSLONG')),
                    ('elevation_m', ('SITEELEV', 'ALT-OBS', 'OBSALT'))):
        v = _hv(header, hk)
        if v is None:
            continue
        fv = _f(v)
        if fv is None:
            try:
                from astropy.coordinates import Angle
                import astropy.units as u
                fv = float(Angle(str(v), unit=u.deg).deg)
            except Exception:
                fv = None
        if fv is not None:
            site[key] = fv
    return {'instrument': ins, 'site': site, 'filter': str(header.get('FILTER', '') or '').strip()}


def fill_from_header(setup, header, overwrite=False):
    """Copy what the header tells into the setup (empty fields only, unless overwrite).
    Returns the list of filled field names."""
    h = from_header(header)
    filled = []
    for sect in ('instrument', 'site'):
        for k, v in h[sect].items():
            if overwrite or setup[sect].get(k) in (None, '', 0):
                if k == 'pixel_scale' and _f(setup[sect].get('focal_length_mm')) and _f(setup[sect].get('pixel_um')):
                    continue
                setup[sect][k] = v
                filled.append(f"{sect}.{k}")
    return filled


def match_score(setup, header):
    """How well a FITS header fits a setup (0 = not at all). Sensor size, focal length / pixel
    scale, pixel size, telescope and camera names count."""
    h = from_header(header)['instrument']
    ins = setup.get('instrument') or {}
    score, checks = 0.0, 0
    for k in ('sensor_x', 'sensor_y'):
        if _f(ins.get(k)) and _f(h.get(k)):
            checks += 1
            score += 1.0 if abs(_f(ins[k]) - h[k]) < 2 else -1.0
    if _f(ins.get('focal_length_mm')) and _f(h.get('focal_length_mm')):
        checks += 1
        score += 1.5 if abs(_f(ins['focal_length_mm']) / h['focal_length_mm'] - 1) < 0.08 else -1.5
    else:
        sc = derived(setup)['pixel_scale']
        if sc and h.get('pixel_scale'):
            checks += 1
            score += 1.5 if abs(sc / h['pixel_scale'] - 1) < 0.1 else -1.5
    if _f(ins.get('pixel_um')) and _f(h.get('pixel_um')):
        checks += 1
        score += 0.5 if abs(_f(ins['pixel_um']) - h['pixel_um']) < 0.05 else -0.5
    for k in ('telescope', 'camera'):
        a, b = str(ins.get(k) or '').lower().strip(), str(h.get(k) or '').lower().strip()
        if a and b:
            checks += 1
            score += 0.7 if (a in b or b in a) else -0.3
    return score if checks else 0.0


def best_match(header, names=None):
    """(name, score) of the best fitting setup, or (None, 0)."""
    best, bs = None, 0.0
    for n in (names if names is not None else list_names()):
        s = load(n)
        if s is None:
            continue
        sc = match_score(s, header)
        if sc > bs:
            best, bs = n, sc
    return best, bs
