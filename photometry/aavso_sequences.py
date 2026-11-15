"""
Stored AAVSO comparison-star sequences (v4.35).

Every sequence fetched from the AAVSO VSP (photometry.calibration.fetch_aavso_chart_stars)
is kept as one JSON file per variable in photometry_refstars/aavso_sequences/, together
with the field size, the download date and the cross-match with the reference catalogue.
'Get AAVSO Ref Stars' uses the stored file instead of asking AAVSO again; 'Refresh from
AAVSO' fetches it anew. Names are matched loosely ('HU Aqr' = 'hu aqr' = 'HU_Aqr').
"""
import json
import os
import re
import time

DIR = os.path.join("photometry_refstars", "aavso_sequences")
OLD_DAYS = 365


def _key(target):
    from photometry.local_catalogs import name_key
    return name_key(target)


def path_for(target):
    k = re.sub(r'[^a-z0-9+\-.]+', '_', _key(target)) or 'unnamed'
    return os.path.join(DIR, f"{k}.json")


def load(target):
    """Stored sequence {'target', 'fetched', 'fov_arcmin', 'catalog', 'stars', ...} or None."""
    p = path_for(target)
    try:
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or not d.get('stars') or _key(d.get('target', '')) != _key(target):
        return None
    return d


def save(target, stars, fov_arcmin, catalog=None):
    """Store a sequence (stars incl. catalogue cross-match). Returns the file path."""
    os.makedirs(DIR, exist_ok=True)
    p = path_for(target)
    d = {'target': target, 'fetched': time.strftime('%Y-%m-%d %H:%M:%S'),
         'fov_arcmin': float(fov_arcmin), 'catalog': catalog or '', 'n': len(stars), 'stars': stars}
    tmp = p + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1, default=float)
    os.replace(tmp, p)
    return p


def update_crossmatch(target, stars, catalog):
    """Store a new catalogue cross-match for an existing sequence (keeps the fetch date)."""
    d = load(target)
    if not d:
        return None
    d['stars'] = stars
    d['catalog'] = catalog
    with open(path_for(target), 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1, default=float)
    return path_for(target)


def age_days(d):
    try:
        t = time.mktime(time.strptime(d['fetched'], '%Y-%m-%d %H:%M:%S'))
        return (time.time() - t) / 86400.0
    except Exception:
        return None


def list_stored():
    """[(target, fetched, n_stars, path)] of all stored sequences."""
    out = []
    if not os.path.isdir(DIR):
        return out
    for f in sorted(os.listdir(DIR)):
        if f.endswith('.json'):
            try:
                with open(os.path.join(DIR, f), encoding='utf-8') as fh:
                    d = json.load(fh)
                out.append((d.get('target', f[:-5]), d.get('fetched', ''), len(d.get('stars', [])), os.path.join(DIR, f)))
            except (OSError, ValueError):
                continue
    return out
