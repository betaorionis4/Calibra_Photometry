"""
APASS DR10 downloads from the AAVSO archive (v4.32).

The download page https://archive.aavso.org/download-apass-data sends a form to
cgi-bin/apass_dr10_download.pl (POST: ra, dec - sexagesimal or degrees -, radius in
degrees <= 2, outtype=1 for CSV). The answer is a CSV file (radeg, decdeg,
'Johnson_V (V)', 'Johnson_B (B)', ... ; 'NA' = no value) covering a SQUARE of
2 x radius around the centre. Calibra saves it unchanged in the APASS folder of the
local catalogues, where photometry.local_catalogs reads it like any other file.

Two ways to use it:
  * download_field()  - the button "Download APASS DR10..." (Settings, Light Curves,
                        Field Variables): one field, on request.
  * auto_download()   - called by calibration.fetch_online_catalog when APASS is the
                        reference catalogue, the field is not in the local files and the
                        option "Download APASS DR10 automatically" is on (default off).

AAVSO asks to use the service with moderation. Therefore: every field is downloaded
only once (it is local afterwards), automatic downloads wait at least MIN_INTERVAL
seconds between requests, and after a failure no automatic download is tried for
PAUSE_AFTER_FAILURE seconds (VizieR DR9 is used meanwhile).
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request

URL = "https://archive.aavso.org/cgi-bin/apass_dr10_download.pl"
PAGE = "https://archive.aavso.org/download-apass-data"
MAX_RADIUS = 2.0                 # deg, limit of the AAVSO service
MIN_RADIUS = 0.2                 # deg
MIN_INTERVAL = 10.0              # s between automatic downloads
PAUSE_AFTER_FAILURE = 1800.0     # s
LOG_FILE = os.path.join("photometry_refstars", "apass_dr10_downloads.json")
USER_AGENT = "Calibra photometry software (APASS DR10 field download)"

_RUNTIME = {'last': 0.0, 'failed_at': 0.0, 'failed_msg': ''}


def sexagesimal(ra_deg, dec_deg):
    """('HH:MM:SS.ss', '+DD:MM:SS.s') for the form."""
    ra = (float(ra_deg) % 360.0) / 15.0
    h = int(ra); m = int((ra - h) * 60); s = ((ra - h) * 60 - m) * 60
    if s >= 59.995:
        s = 0.0; m += 1
    if m >= 60:
        m = 0; h = (h + 1) % 24
    d = abs(float(dec_deg)); sign = '-' if dec_deg < 0 else '+'
    dd = int(d); dm = int((d - dd) * 60); ds = ((d - dd) * 60 - dm) * 60
    if ds >= 59.95:
        ds = 0.0; dm += 1
    if dm >= 60:
        dm = 0; dd += 1
    return f"{h:02d}:{m:02d}:{s:05.2f}", f"{sign}{dd:02d}:{dm:02d}:{ds:04.1f}"


def radius_for_field(radius_arcmin, margin=1.2):
    """Download radius [deg] for a field of radius radius_arcmin (circle inside the square)."""
    r = float(radius_arcmin) / 60.0 * margin
    return round(min(MAX_RADIUS, max(MIN_RADIUS, r)), 3)


def target_folder():
    """APASS folder of the local catalogues (created as <root>/APASS if there is none yet);
    without a catalogue folder: photometry_refstars/APASS_DR10 in the Calibra folder."""
    from photometry import local_catalogs as lc
    cfg = lc.get_config()
    info = cfg['catalogs'].get('APASS')
    if info and os.path.isdir(info.get('path', '')):
        return info['path']
    root = cfg.get('root')
    if root and os.path.isdir(root):
        return os.path.join(root, 'APASS')
    return os.path.join("photometry_refstars", "APASS_DR10")


def _load_log():
    try:
        with open(LOG_FILE, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _save_log(entries):
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(entries[-2000:], f, indent=1)
    except OSError:
        pass


def already_downloaded(ra_deg, dec_deg, radius_deg):
    """Path of an earlier download whose square contains the requested square, else None."""
    import numpy as np
    for e in _load_log():
        if not os.path.exists(e.get('path', '')):
            continue
        cosd = max(np.cos(np.radians(dec_deg)), 1e-3)
        dra = abs(((ra_deg - e['ra'] + 180.0) % 360.0) - 180.0) * cosd
        if dra + radius_deg <= e['radius'] * cosd + 1e-6 and abs(dec_deg - e['dec']) + radius_deg <= e['radius'] + 1e-6:
            return e['path']
    return None


def _http_post(params, timeout):
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(URL, data=data, headers={'User-Agent': USER_AGENT, 'Referer': PAGE})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get('Content-Disposition', '') or ''


def download_field(ra_deg, dec_deg, radius_deg, folder=None, timeout=300, log=print, force=False):
    """
    Download the APASS DR10 stars in the square of half-width radius_deg [deg] around
    (ra_deg, dec_deg) and save the CSV unchanged in folder (default: target_folder()).
    Returns (path, n_stars, message). Raises RuntimeError on failure.
    """
    from photometry import local_catalogs as lc
    radius_deg = min(MAX_RADIUS, max(0.01, float(radius_deg)))
    if not force:
        old = already_downloaded(ra_deg, dec_deg, radius_deg)
        if old:
            return old, None, f"already downloaded: {os.path.basename(old)}"
    ra_s, dec_s = sexagesimal(ra_deg, dec_deg)
    log(f"APASS DR10: downloading RA {ra_s}, Dec {dec_s}, radius {radius_deg:g} deg from archive.aavso.org "
        "(can take a minute in rich fields) ...")
    t0 = time.time()
    try:
        body, disp = _http_post({'ra': ra_s, 'dec': dec_s, 'radius': f"{radius_deg:g}", 'outtype': '1'}, timeout)
    except Exception as e:
        raise RuntimeError(f"APASS DR10 download failed: {e}")
    text = body.decode('utf-8', errors='replace')
    first = text.lstrip().split('\n', 1)[0].lower()
    if not first.startswith('radeg') or 'decdeg' not in first:
        snippet = re.sub(r'<[^>]+>', ' ', text[:400])
        raise RuntimeError("APASS DR10: unexpected answer from the server (no CSV): " + ' '.join(snippet.split())[:200])
    n = max(0, sum(1 for line in text.splitlines()[1:] if line.strip()))
    m = re.search(r'filename\s*=\s*"?([^";]+)"?', disp)
    name = os.path.basename(m.group(1).strip()) if m else ''
    if not name.lower().endswith('.csv'):
        name = f"apass_{ra_deg:.4f}_{dec_deg:.4f}_{radius_deg:g}_dr10.csv"
    folder = folder or target_folder()
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(text)
    entries = [e for e in _load_log() if os.path.abspath(e.get('path', '')) != os.path.abspath(path)]
    entries.append({'ra': float(ra_deg), 'dec': float(dec_deg), 'radius': float(radius_deg), 'path': path,
                    'n': n, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})
    _save_log(entries)
    lc.register_folder('APASS', folder)
    msg = f"{n} stars saved as {name} ({time.time() - t0:.0f} s)"
    log("APASS DR10: " + msg)
    return path, n, msg


def auto_download(ra_deg, dec_deg, radius_arcmin, log=print):
    """Automatic download for a field missing in the local files (option on, online allowed).
    Returns True if a file is now available locally."""
    from photometry import local_catalogs as lc
    if not (lc.apass_auto() and lc.online_allowed() and lc.get_config().get('use_local', True)):
        return False
    now = time.time()
    if now - _RUNTIME['failed_at'] < PAUSE_AFTER_FAILURE:
        return False
    wait = MIN_INTERVAL - (now - _RUNTIME['last'])
    if wait > 0:
        time.sleep(wait)
    _RUNTIME['last'] = time.time()
    try:
        download_field(ra_deg, dec_deg, radius_for_field(radius_arcmin), log=log)
        return True
    except Exception as e:
        _RUNTIME['failed_at'] = time.time()
        _RUNTIME['failed_msg'] = str(e)
        log(f"{e} - using VizieR (APASS DR9) instead; no further automatic APASS DR10 download for "
            f"{PAUSE_AFTER_FAILURE / 60:.0f} min.")
        return False


def fields_from_images(paths, group_deg=0.25, margin=1.1):
    """
    Fields to download for a set of plate-solved images: images whose centres lie within
    group_deg of each other form one field. Returns (fields, no_wcs) with fields =
    [(ra, dec, radius_deg, n_images)] (radius = largest centre-to-corner distance x margin,
    at most MAX_RADIUS) and no_wcs = images without a celestial WCS.
    """
    import numpy as np
    from photometry.field_variables import image_footprints

    def vec(ra, dec):
        ra, dec = np.radians(ra), np.radians(dec)
        return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])

    def sep(v1, v2):
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))
    imgs, no_wcs = [], []
    for p, w, shape in image_footprints(paths):
        if w is None:
            no_wcs.append(p)
            continue
        ny, nx = shape
        pts = w.pixel_to_world_values(np.array([(nx - 1) / 2, 0, nx - 1, 0, nx - 1]),
                                      np.array([(ny - 1) / 2, 0, 0, ny - 1, ny - 1]))
        c = vec(pts[0][0], pts[1][0])
        corners = [vec(pts[0][k], pts[1][k]) for k in range(1, 5)]
        imgs.append((c, corners))
    groups = []
    for c, corners in imgs:
        for g in groups:
            if sep(g['c'] / np.linalg.norm(g['c']), c) <= group_deg:
                g['c'] = g['c'] + c; g['corners'] += corners; g['n'] += 1
                break
        else:
            groups.append({'c': c.copy(), 'corners': list(corners), 'n': 1})
    fields = []
    for g in groups:
        c = g['c'] / np.linalg.norm(g['c'])
        r = max(sep(c, k) for k in g['corners']) * margin
        ra = float(np.degrees(np.arctan2(c[1], c[0])) % 360.0)
        dec = float(np.degrees(np.arcsin(c[2])))
        fields.append((ra, dec, round(min(MAX_RADIUS, max(MIN_RADIUS, r)), 3), g['n']))
    return fields, no_wcs
