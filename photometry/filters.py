"""
Central filter classification for Calibra (v4.15).

Before v4.15 the B/V decision was made in five different ways across the code
(e.g. ``'B' in FILTER``), so a filter called "Bessell V" or "Baader V" was
treated as B, and a file could be counted as B in one tab and as V in another.
All code should now call :func:`classify_filter`.
"""
import re

_SPLIT = re.compile(r'[^A-Z0-9]+')


def classify_filter(filt, v_key="Vmag", b_key="Bmag"):
    """
    Map a FITS FILTER value to 'V', 'B' or None (unknown / other band).

    Rules, in order (case-insensitive):
      1. Exact match with the user's V / B keyword (Settings -> Filter mapping).
      2. The filter is just 'V' or 'B'.
      3. The keyword (>= 2 characters) is contained in the filter string,
         e.g. keyword 'Vmag' matches 'Vmag_2'.
      4. A separate word 'V' or 'B' in the filter string, e.g. 'Bessell V',
         'Baader V', 'Johnson-B'. (A lone letter 'B' inside 'Bessell' does NOT count.)
      5. Names starting with 'JOHNSON'/'BESSELL' followed by V or B (e.g. 'JohnsonV').
    """
    f = str(filt or '').strip().upper()
    if not f:
        return None
    v = str(v_key or '').strip().upper()
    b = str(b_key or '').strip().upper()

    # 1. exact keyword
    if v and f == v:
        return 'V'
    if b and f == b:
        return 'B'
    # 2. single letter
    if f in ('V', 'B'):
        return f
    # 3. keyword contained (only meaningful for keywords of >= 2 characters)
    v_in = bool(v) and len(v) >= 2 and v in f
    b_in = bool(b) and len(b) >= 2 and b in f
    if v_in and not b_in:
        return 'V'
    if b_in and not v_in:
        return 'B'
    if v_in and b_in:  # both contained: prefer the longer (more specific) keyword
        return 'V' if len(v) >= len(b) else 'B'
    # 4. separate word V / B
    tokens = [t for t in _SPLIT.split(f) if t]
    has_v, has_b = 'V' in tokens, 'B' in tokens
    if has_v and not has_b:
        return 'V'
    if has_b and not has_v:
        return 'B'
    # 5. JohnsonV / BessellB style
    for prefix in ('JOHNSON', 'BESSELL', 'BESSEL'):
        if f.startswith(prefix) and f[len(prefix):] in ('V', 'B'):
            return f[len(prefix):]
    return None


def filter_band(filt, v_key="Vmag", b_key="Bmag", default='V'):
    """Like classify_filter but returns *default* for unknown filters."""
    band = classify_filter(filt, v_key, b_key)
    return band if band else default


_L_NAMES = ('L', 'LMAG', 'L_MAG', 'LUM', 'LUMINANCE', 'LUMINANCEFILTER', 'LRGB_L', 'IR-CUT', 'UVIR', 'UV-IR', 'UVIRCUT',
            'UVIR-CUT', 'UV-IR-CUT', 'UVIRCUTFILTER')


def classify_band(filt, v_key="Vmag", b_key="Bmag", l_key="L"):
    """
    v4.23: like classify_filter, but also recognises the luminance (L) filter:
    'V', 'B', 'L' or None. L = the user's L keyword (Settings), or names such as
    'L', 'Lum', 'Luminance', 'UV/IR-cut' (also as a separate word, e.g. 'Baader L').
    classify_filter itself is unchanged (it is used where only B/V make sense).
    """
    band = classify_filter(filt, v_key, b_key)
    if band:
        return band
    f = str(filt or '').strip().upper()
    if not f:
        return None
    lk = str(l_key or '').strip().upper()
    if lk and (f == lk or (len(lk) >= 2 and lk in f)):
        return 'L'
    compact = re.sub(r'[^A-Z0-9_-]', '', f)
    if compact in _L_NAMES or f in _L_NAMES:
        return 'L'
    tokens = [t for t in _SPLIT.split(f) if t]
    if any(t in ('L', 'LUM', 'LUMINANCE') for t in tokens):
        return 'L'
    return None
