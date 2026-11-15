"""
Observation planner (v4.37).

Which variable stars can be observed from a site in a given night and time window,
above a minimum altitude, in a chosen azimuth sector, away from the Moon, with
magnitudes suited to the instrument - from the local GCVS list, a priority list or a
list of open clusters. Everything is computed locally (no internet): positions of Sun
and Moon from low-precision formulas (Astronomical Almanac; Sun ~0.01 deg, Moon ~0.3 deg),
sidereal time (IAU 1982), precession J2000 -> date (IAU 1976). Accurate to a few
arcminutes - plenty for planning. Refraction is ignored.
"""
import csv
import datetime as _dt
import math
import os
import re

import numpy as np

PRIORITY_FILE = os.path.join("photometry_refstars", "priority_targets.csv")
CLUSTER_FILE = os.path.join("photometry_refstars", "open_clusters.csv")

# Approximate J2000 centres and radii of the open clusters of interest (radius in arcmin,
# about half the apparent diameter). Editable in photometry_refstars/open_clusters.csv.
DEFAULT_CLUSTERS = [
    ("IC 1805", "Melotte 15, Heart Nebula (Cas)", "02 32 42", "+61 27 00", 11),
    ("NGC 1912", "M38 (Aur)", "05 28 42", "+35 51 18", 11),
    ("NGC 2355", "Gem", "07 16 59", "+13 45 00", 5),
    ("Melotte 22", "Pleiades, M45 (Tau)", "03 47 24", "+24 07 00", 60),
    ("NGC 6834", "Cyg", "19 52 12", "+29 25 00", 3),
    ("NGC 188", "Caldwell 1 (Cep)", "00 48 26", "+85 15 18", 7),
    ("NGC 869", "h Persei", "02 19 00", "+57 08 00", 9),
    ("NGC 884", "chi Persei", "02 22 18", "+57 08 12", 9),
    ("NGC 457", "Owl / ET cluster (Cas)", "01 19 32", "+58 17 27", 7),
    ("NGC 663", "Cas", "01 46 09", "+61 14 06", 8),
]

SECTORS = {          # azimuth ranges (deg, from North through East)
    'any': (0, 360), 'N': (315, 45), 'NE': (0, 90), 'E': (45, 135), 'SE': (90, 180), 'S': (135, 225),
    'SW': (180, 270), 'W': (225, 315), 'NW': (270, 360),
    'east half': (0, 180), 'south half': (90, 270), 'west half': (180, 360), 'north half': (270, 90),
}


# ------------------------------------------------------------------ time
def jd_from_datetime(t):
    """Julian date of an aware datetime (or naive = UTC)."""
    if t.tzinfo is not None:
        t = t.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return (t - _dt.datetime(2000, 1, 1, 12)).total_seconds() / 86400.0 + 2451545.0


def get_tz(name):
    """tzinfo for an IANA name ('Europe/Stockholm'); fallback: the computer's local zone."""
    if name:
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(name)
        except Exception:
            pass
        m = re.fullmatch(r'(?:UTC|GMT)?\s*([+-])(\d{1,2})(?::?(\d{2}))?', str(name).strip(), re.IGNORECASE)
        if m:
            off = int(m.group(2)) * 60 + int(m.group(3) or 0)
            return _dt.timezone(_dt.timedelta(minutes=off if m.group(1) == '+' else -off))
    return _dt.datetime.now().astimezone().tzinfo


def parse_hhmm(s):
    s = str(s).strip().replace('.', ':')
    m = re.fullmatch(r'(\d{1,2})(?::(\d{2}))?', s)
    if not m:
        raise ValueError(f"time '{s}' - use HH:MM")
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    if not (0 <= h <= 24 and 0 <= mi < 60):
        raise ValueError(f"time '{s}' out of range")
    return h, mi


def time_grid(date, start, end, tz, step_min=10):
    """Local datetimes from date+start to date(+1)+end every step_min minutes (end after
    midnight if it is earlier than start, e.g. 18:00 -> 02:00). Returns (list of aware
    datetimes, numpy array of JD)."""
    if isinstance(date, str):
        date = _dt.date.fromisoformat(date.strip())
    h0, m0 = parse_hhmm(start)
    h1, m1 = parse_hhmm(end)
    t0 = _dt.datetime.combine(date, _dt.time(0, 0)) + _dt.timedelta(hours=h0, minutes=m0)
    t1 = _dt.datetime.combine(date, _dt.time(0, 0)) + _dt.timedelta(hours=h1, minutes=m1)
    if t1 <= t0:
        t1 += _dt.timedelta(days=1)
    n = max(2, int(round((t1 - t0).total_seconds() / 60.0 / step_min)) + 1)
    times = [(t0 + (t1 - t0) * k / (n - 1)).replace(tzinfo=tz) for k in range(n)]
    return times, np.array([jd_from_datetime(t) for t in times])


# ------------------------------------------------------------------ astronomy
def gmst_deg(jd):
    d = jd - 2451545.0
    T = d / 36525.0
    return (280.46061837 + 360.98564736629 * d + 0.000387933 * T ** 2 - T ** 3 / 38710000.0) % 360.0


def precess(ra_deg, dec_deg, jd):
    """J2000 -> mean equator of date (IAU 1976 precession)."""
    T = (np.mean(jd) - 2451545.0) / 36525.0
    a = np.radians
    zeta = a((2306.2181 * T + 0.30188 * T ** 2 + 0.017998 * T ** 3) / 3600.0)
    z = a((2306.2181 * T + 1.09468 * T ** 2 + 0.018203 * T ** 3) / 3600.0)
    th = a((2004.3109 * T - 0.42665 * T ** 2 - 0.041833 * T ** 3) / 3600.0)
    ra, dec = a(np.asarray(ra_deg, float)), a(np.asarray(dec_deg, float))
    A = np.cos(dec) * np.sin(ra + zeta)
    B = np.cos(th) * np.cos(dec) * np.cos(ra + zeta) - np.sin(th) * np.sin(dec)
    C = np.sin(th) * np.cos(dec) * np.cos(ra + zeta) + np.cos(th) * np.sin(dec)
    return (np.degrees(np.arctan2(A, B) + z) % 360.0), np.degrees(np.arcsin(np.clip(C, -1, 1)))


def sun_radec(jd):
    n = np.asarray(jd, float) - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360)
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 0.0000004 * n)
    ra = np.degrees(np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))) % 360
    dec = np.degrees(np.arcsin(np.sin(eps) * np.sin(lam)))
    return ra, dec


def moon_radec(jd):
    """Geocentric Moon (low precision, ~0.3 deg); returns (ra, dec, ecliptic longitude)."""
    T = (np.asarray(jd, float) - 2451545.0) / 36525.0
    s = lambda x: np.sin(np.radians(x))
    lam = (218.32 + 481267.881 * T + 6.29 * s(135.0 + 477198.87 * T) - 1.27 * s(259.3 - 413335.36 * T)
           + 0.66 * s(235.7 + 890534.22 * T) + 0.21 * s(269.9 + 954397.74 * T) - 0.19 * s(357.5 + 35999.05 * T)
           - 0.11 * s(186.5 + 966404.03 * T)) % 360
    bet = (5.13 * s(93.3 + 483202.02 * T) + 0.28 * s(228.2 + 960400.89 * T) - 0.28 * s(318.3 + 6003.15 * T)
           - 0.17 * s(217.6 - 407332.21 * T))
    eps = np.radians(23.439)
    l, b = np.radians(lam), np.radians(bet)
    x = np.cos(b) * np.cos(l)
    y = np.cos(eps) * np.cos(b) * np.sin(l) - np.sin(eps) * np.sin(b)
    zc = np.sin(eps) * np.cos(b) * np.sin(l) + np.cos(eps) * np.sin(b)
    return np.degrees(np.arctan2(y, x)) % 360, np.degrees(np.arcsin(zc)), lam


def moon_illumination(jd):
    """Illuminated fraction of the Moon (0 = new, 1 = full)."""
    sra, sdec = sun_radec(jd)
    mra, mdec, _ = moon_radec(jd)
    return (1 - np.cos(np.radians(separation(sra, sdec, mra, mdec)))) / 2.0


def separation(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def altaz(ra_deg, dec_deg, jd, lat, lon):
    """Altitude and azimuth (deg; azimuth from North through East) of positions of date.
    ra/dec: shape (N,) ; jd: shape (T,) -> arrays (N, T)."""
    ra = np.radians(np.atleast_1d(ra_deg))[:, None]
    dec = np.radians(np.atleast_1d(dec_deg))[:, None]
    lst = np.radians((gmst_deg(np.atleast_1d(jd)) + lon) % 360)[None, :]
    H = lst - ra
    phi = math.radians(lat)
    sin_alt = np.sin(dec) * math.sin(phi) + np.cos(dec) * math.cos(phi) * np.cos(H)
    alt = np.arcsin(np.clip(sin_alt, -1, 1))
    az = np.arctan2(-np.cos(dec) * np.sin(H), np.sin(dec) * math.cos(phi) - np.cos(dec) * math.sin(phi) * np.cos(H))
    return np.degrees(alt), np.degrees(az) % 360


def in_sector(az, sector):
    """Boolean array: azimuth inside the sector ((a0, a1) in deg, may wrap through North)."""
    a0, a1 = sector
    if (a0, a1) == (0, 360):
        return np.ones_like(az, bool)
    return (az >= a0) & (az <= a1) if a0 <= a1 else (az >= a0) | (az <= a1)


def parse_sector(text):
    """'any', 'E', 'SE', 'east half', or '60-200' -> (a0, a1)."""
    t = str(text or 'any').strip()
    for k, v in SECTORS.items():
        if t.lower() == k.lower() or t.lower().startswith(k.lower() + ' '):
            return v
    m = re.fullmatch(r'(\d+(?:\.\d*)?)\s*[-–]\s*(\d+(?:\.\d*)?)', t)
    if m:
        return float(m.group(1)) % 360, float(m.group(2)) % 360 if float(m.group(2)) != 360 else 360.0
    raise ValueError(f"azimuth sector '{t}' - use any, N, NE, E, SE, S, SW, W, NW, 'east half' ... or e.g. 60-200")


# ------------------------------------------------------------------ night overview
def night_info(times, jd, lat, lon, dark_sun_alt=-12.0):
    """Sun altitude, dark mask, Moon position/altitude/illumination for the time grid."""
    sra, sdec = sun_radec(jd)
    s_alt = np.array([altaz(sra[k], sdec[k], jd[k], lat, lon)[0][0, 0] for k in range(len(jd))])
    mra, mdec, _ = moon_radec(jd)
    m_alt = np.array([altaz(mra[k], mdec[k], jd[k], lat, lon)[0][0, 0] for k in range(len(jd))])
    return {'sun_alt': s_alt, 'dark': s_alt <= dark_sun_alt, 'moon_ra': mra, 'moon_dec': mdec, 'moon_alt': m_alt,
            'moon_illum': float(np.mean(moon_illumination(jd)))}


def twilight_times(date, tz, lat, lon, sun_alt=-12.0):
    """(evening, morning) local times when the Sun passes sun_alt, for the night after date.
    None if it does not happen (e.g. bright summer nights in Uppsala)."""
    times, jd = time_grid(date, "12:00", "12:00", tz, step_min=2)
    sra, sdec = sun_radec(jd)
    alt = np.array([altaz(sra[k], sdec[k], jd[k], lat, lon)[0][0, 0] for k in range(len(jd))])
    below = alt <= sun_alt
    if not below.any():
        return None, None
    i0 = int(np.argmax(below))
    i1 = len(below) - 1 - int(np.argmax(below[::-1]))
    return times[i0], times[min(i1 + 1, len(times) - 1)]


# ------------------------------------------------------------------ targets
def _f(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def type_matches(t, patterns):
    """GCVS type (e.g. 'EA/DM', 'DSCT', 'M') against user patterns ('EA EW', 'DSCT*', 'M')."""
    if not patterns:
        return True
    t = str(t or '').upper()
    parts = [p for p in re.split(r'[/|+:\s]+', t) if p]
    for p in patterns:
        p = p.upper().strip()
        if not p:
            continue
        if p.endswith('*'):
            if t.startswith(p[:-1]) or any(x.startswith(p[:-1]) for x in parts):
                return True
        elif p == t or p in parts:
            return True
    return False


def mag_ok(t, bright=None, faint=None):
    """Brightness range of the star overlaps with what the instrument can measure.
    Stars without magnitudes are kept."""
    mx, mn = _f(t.get('max')), _f(t.get('min'))
    if mx is None and mn is None:
        return True
    brightest = mx if mx is not None else mn
    faintest = mn if mn is not None else mx
    if faint is not None and brightest > faint:
        return False
    if bright is not None and faintest < bright:
        return False
    return True


def filter_targets(targets, bright=None, faint=None, types=None, pmin=None, pmax=None, amp_min=None):
    out = []
    for t in targets:
        if t.get('ra') is None or t.get('dec') is None:
            continue
        if not mag_ok(t, bright, faint) or not type_matches(t.get('type'), types):
            continue
        p = _f(t.get('period'))
        if (pmin is not None or pmax is not None) and p is None:
            continue
        if pmin is not None and p < pmin or pmax is not None and p > pmax:
            continue
        if amp_min is not None:
            mx, mn = _f(t.get('max')), _f(t.get('min'))
            if mx is None or mn is None or (mn - mx) < amp_min:
                continue
        out.append(t)
    return out


def plan(targets, times, jd, lat, lon, alt_min=30.0, sector=(0, 360), moon_min=None, dark_sun_alt=-12.0,
         night=None):
    """
    Observability of targets (dicts with ra, dec in J2000 degrees) on the time grid.
    Returns (rows, night) with one row per target that is observable at least once:
    {'target', 'hours', 'first', 'last', 'best', 'best_alt', 'best_az', 'max_alt', 'moon_sep',
     'up_elsewhere' (hours above alt_min outside the sector, dark), 'alt' (array), 'az' (array),
     'ok' (bool array)}. Targets that are never observable in the window are in rows with
    hours = 0 only if they are above alt_min elsewhere (status for priority lists).
    """
    if night is None:
        night = night_info(times, jd, lat, lon, dark_sun_alt)
    if not targets:
        return [], night
    ra = np.array([float(t['ra']) for t in targets])
    dec = np.array([float(t['dec']) for t in targets])
    ra_d, dec_d = precess(ra, dec, jd)
    alt, az = altaz(ra_d, dec_d, jd, lat, lon)
    dark = night['dark'][None, :]
    up = (alt >= alt_min) & dark
    ok = up & in_sector(az, sector)
    msep = separation(ra_d[:, None], dec_d[:, None], night['moon_ra'][None, :], night['moon_dec'][None, :])
    moon_up = (night['moon_alt'] > 0)[None, :]
    if moon_min is not None:
        ok &= ~moon_up | (msep >= moon_min)
    dt_h = (jd[1] - jd[0]) * 24.0 if len(jd) > 1 else 0.0
    # meridian passage (upper culmination) nearest to the middle of the window
    jd_mid = 0.5 * (jd[0] + jd[-1])
    ha_mid = (gmst_deg(jd_mid) + lon - ra_d) % 360.0
    ha_mid = np.where(ha_mid > 180, ha_mid - 360, ha_mid)
    transit_jd = jd_mid - ha_mid / 360.98564736629
    rows = []
    for i, t in enumerate(targets):
        n_ok = int(ok[i].sum())
        n_up = int(up[i].sum())
        if n_ok == 0 and n_up == 0:
            continue
        sel = ok[i] if n_ok else up[i]
        k_best = int(np.argmax(np.where(sel, alt[i], -99)))
        idx = np.where(sel)[0]
        rows.append({'target': t, 'hours': n_ok * dt_h, 'up_elsewhere': (n_up - n_ok) * dt_h if n_ok == 0 else 0.0,
                     'first': times[idx[0]], 'last': times[idx[-1]], 'best': times[k_best],
                     'best_alt': float(alt[i, k_best]), 'best_az': float(az[i, k_best]),
                     'max_alt': float(np.max(np.where(night['dark'], alt[i], -99))),
                     'moon_sep': float(np.min(msep[i])), 'alt': alt[i], 'az': az[i], 'ok': ok[i],
                     'transit': times[0] + _dt.timedelta(days=float(transit_jd[i] - jd[0])),
                     'transit_in_window': bool(jd[0] <= transit_jd[i] <= jd[-1])})
    rows.sort(key=lambda r: (-r['hours'], -r['best_alt']))
    return rows, night


_GREEK = ['alf', 'bet', 'gam', 'del', 'eps', 'zet', 'eta', 'the', 'iot', 'kap', 'lam', 'mu', 'nu', 'xi', 'omi',
          'pi', 'rho', 'sig', 'tau', 'ups', 'phi', 'chi', 'psi', 'ome']


def name_sort_key(name):
    """Sort GCVS names by constellation first, then in the GCVS order of the designation:
    Bayer letters (alf ... ome), R ... Z, RR ... ZZ, AA ... QZ, V335 ... (numbers).
    Names without a constellation (e.g. 'NGC 869') come after, alphabetically."""
    n = str(name or '').strip()
    m = re.fullmatch(r'(.+?)\s+([A-Za-z]{3})', n)
    if not m:
        return (1, n.lower(), 0, 0, '')
    des, con = m.group(1).strip(), m.group(2).capitalize()
    d = des.lower()
    g = re.fullmatch(r'([a-z]+)(\d*)', d)
    if g and g.group(1) in _GREEK:
        return (0, con, 0, _GREEK.index(g.group(1)) * 100 + int(g.group(2) or 0), d)
    if re.fullmatch(r'[A-Z]', des):
        return (0, con, 1, ord(des) - ord('R'), d)
    if re.fullmatch(r'[A-Z]{2}', des):
        a, b = des
        grp = 0 if a >= 'R' else 1                      # RR..ZZ before AA..QZ
        return (0, con, 2 + grp, (ord(a) - 65) * 26 + (ord(b) - 65), d)
    v = re.fullmatch(r'V\s*(\d+)', des, re.IGNORECASE)
    if v:
        return (0, con, 4, int(v.group(1)), d)
    return (0, con, 5, 0, d)


def compass(az):
    names = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return names[int((az % 360) / 22.5 + 0.5) % 16]


# ------------------------------------------------------------------ lists
def load_priority():
    """Priority targets (same columns as target lists + 'notes', 'added')."""
    if not os.path.exists(PRIORITY_FILE):
        return []
    from photometry.field_variables import parse_target_list
    try:
        targets, _ = parse_target_list(PRIORITY_FILE)
    except Exception:
        return []
    notes = {}
    try:
        with open(PRIORITY_FILE, encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                notes[(r.get('name') or '').strip()] = (r.get('notes') or '', r.get('added') or '')
    except Exception:
        pass
    for t in targets:
        t['notes'], t['added'] = notes.get(t.get('name', ''), ('', ''))
    return targets


def save_priority(targets):
    os.makedirs(os.path.dirname(PRIORITY_FILE), exist_ok=True)
    with open(PRIORITY_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['name', 'ra', 'dec', 'type', 'period', 'max', 'min', 'epoch', 'notes', 'added'])
        for t in targets:
            w.writerow([t.get('name', ''), f"{float(t['ra']):.6f}", f"{float(t['dec']):.6f}", t.get('type') or '',
                        '' if _f(t.get('period')) is None else t['period'], '' if _f(t.get('max')) is None else t['max'],
                        '' if _f(t.get('min')) is None else t['min'], '' if _f(t.get('epoch')) is None else t['epoch'],
                        t.get('notes', ''), t.get('added', '')])


def add_priority(new, notes=''):
    """Add targets to the priority list (by name; existing entries are kept). Returns number added."""
    from photometry.local_catalogs import name_key
    cur = load_priority()
    have = {name_key(t.get('name', '')) for t in cur}
    n = 0
    for t in new:
        if name_key(t.get('name', '')) in have:
            continue
        t = dict(t)
        t.setdefault('notes', notes)
        t['added'] = _dt.date.today().isoformat()
        cur.append(t); have.add(name_key(t.get('name', ''))); n += 1
    save_priority(cur)
    return n


def remove_priority(names):
    from photometry.local_catalogs import name_key
    drop = {name_key(n) for n in names}
    cur = [t for t in load_priority() if name_key(t.get('name', '')) not in drop]
    save_priority(cur)
    return cur


def load_clusters():
    """Open clusters [{'name', 'alias', 'ra', 'dec', 'radius_arcmin'}]; the file is created
    with the default list if it does not exist (and can be edited / extended)."""
    from astropy.coordinates import Angle
    import astropy.units as u
    if not os.path.exists(CLUSTER_FILE):
        os.makedirs(os.path.dirname(CLUSTER_FILE), exist_ok=True)
        with open(CLUSTER_FILE, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['name', 'alias', 'ra', 'dec', 'radius_arcmin'])
            for row in DEFAULT_CLUSTERS:
                w.writerow(row)
    out = []
    with open(CLUSTER_FILE, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            try:
                ra_t, de_t = str(r['ra']).strip(), str(r['dec']).strip()
                ra = float(ra_t) if re.fullmatch(r'[+-]?\d+(\.\d*)?', ra_t) else Angle(ra_t, unit=u.hourangle).deg
                de = float(de_t) if re.fullmatch(r'[+-]?\d+(\.\d*)?', de_t) else Angle(de_t, unit=u.deg).deg
                out.append({'name': r['name'].strip(), 'alias': (r.get('alias') or '').strip(), 'ra': float(ra),
                            'dec': float(de), 'radius_arcmin': float(r.get('radius_arcmin') or 10)})
            except Exception:
                continue
    return out


def cluster_variables(cluster, catalog_targets, radius_factor=1.5, field_radius_arcmin=None):
    """Variables of a target list (e.g. the local GCVS list) within radius_factor x the
    cluster radius (or within field_radius_arcmin if larger)."""
    r = cluster['radius_arcmin'] * radius_factor
    if field_radius_arcmin:
        r = max(r, field_radius_arcmin)
    out = []
    for t in catalog_targets:
        if t.get('ra') is None:
            continue
        d = separation(cluster['ra'], cluster['dec'], float(t['ra']), float(t['dec'])) * 60.0
        if d <= r:
            tt = dict(t); tt['dist_arcmin'] = float(d)
            out.append(tt)
    out.sort(key=lambda t: t['dist_arcmin'])
    return out


def gcvs_targets():
    """All stars of the local GCVS list(s) (see local_catalogs)."""
    from photometry import local_catalogs as lc
    cfg = lc.get_config()
    info = cfg['catalogs'].get('GCVS')
    if not info or not cfg.get('use_local', True):
        return []
    out = []
    for f, ent in lc._gcvs_lists(info):
        out += [dict(t) for t in ent[1]]
    return out
